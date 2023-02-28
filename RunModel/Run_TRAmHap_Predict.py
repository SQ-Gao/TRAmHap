import argparse
import os,sys,glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn

plt.switch_backend('agg')

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

torch.manual_seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('-S', "--sample", required=False,  help="Sample.")
parser.add_argument('-I', "--datapath", required=True, help="Data path to be predict.")
parser.add_argument('-O', "--outFolder", required=True, help="Output folder.")
parser.add_argument('-M', "--modelFolder", required=True, help="A folder with train model.")
parser.add_argument('-K', "--flag", default="all", help="Window size of input gene.")
parser.add_argument('-T', "--statistics", default="all", help="which statistics to train, for example '0,2' seperate by ,")
args = parser.parse_args()

if not os.path.exists(args.outFolder):
    os.makedirs(args.outFolder)

if  args.statistics =='all':
    stats_select = [0,1,2]
else:
    stats_select= [int(_) for _ in args.statistics.split(sep = ',')]

    
try:
    five_k_flag = int(args.flag)
except:
    five_k_flag = args.flag # 5 , 50 or "all"
    
savepath = args.outFolder + '/'

class TorchCRNN(nn.Module):
    def __init__(self, inputflag='all',stats_flag = [0,1,2]):
        super(TorchCRNN, self).__init__()
        self.flag = inputflag
        self.stats_flag = stats_flag

        self.s0 = len(self.stats_flag)

        if (inputflag == 5) or (inputflag == 50):
            self.s0 = self.s0
        else:
            self.s0 = self.s0 *2
        self.to_1d = nn.Sequential(
            nn.Conv2d(1, 128, (self.s0, 2), padding=0, stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )

        self.conv_w3 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 3), padding=(0, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (1, 3), padding=(0, 1), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2), padding=(0, 0)),
            nn.Conv2d(256, 512, (1, 3), padding=(0, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (1, 3), padding=(0, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (1, 3), padding=(0, 1), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 2), (1, 2), padding=(0, 1)),
            nn.Conv2d(512, 1024, (1, 3), padding=(0, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, (1, 3), padding=(0, 1), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        self.conv_w5 = nn.Sequential(
            nn.Conv2d(128, 256, (1, 5), padding=(0, 2), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (1, 5), padding=(0, 2), stride=1),
            nn.ReLU(),
            nn.MaxPool2d((1, 2), (1, 2), padding=(0, 0)),
            nn.Conv2d(256, 512, (1, 5), padding=(0, 2), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (1, 5), padding=(0, 2), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (1, 5), padding=(0, 2), stride=1),
            nn.ReLU(),
            nn.AvgPool2d((1, 2), (1, 2), padding=(0, 1)),
            nn.Conv2d(512, 1024, (1, 5), padding=(0, 2), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, (1, 5), padding=(0, 2), stride=1),
            nn.ReLU(),
            nn.BatchNorm2d(1024),
        )

        self.lstm_layer1 = nn.Sequential(
            nn.LSTM(1024, 512, batch_first=True)
        )

        self.lstm_layer2 = nn.Sequential(
            nn.LSTM(512, 256, batch_first=True)
        )

        self.dense_layer = nn.Sequential(
            nn.Linear(10 * 256, 1000),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1000, 500),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(100, 1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.to_1d(x)  # b*256*1*19
        x = torch.cat((self.conv_w3(x), self.conv_w5(x)), 3)
        x = torch.squeeze(x, 2).transpose(1, 2)
        x, (h, c) = self.lstm_layer1(x)
        x, (h, c) = self.lstm_layer2(x)
        x = x.reshape(-1, 10 * 256)
        x = self.dense_layer(x)
        return x

def getx(x, flag_get='all',stats = stats_select):
    if flag_get == 5:
        return x[:, [_+1 for _ in stats], :]
    elif flag_get == 50:
        return x[:, [_+4 for _ in stats], :]
    elif flag_get == 'all':
        return x[:,  [_+1 for _ in stats] + [_+4 for _ in stats], :]

def datasetsplit(x, test_ratio=0, shuffle=True, flag_get='all',stats = stats_select):
    if test_ratio == 0:
        x= torch.from_numpy(getx(x, flag_get,stats).astype(float)).unsqueeze(-1).transpose(2, 3).transpose(1,2)
        data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=64, shuffle=False)
        return data

x_test = np.load(args.datapath, allow_pickle=True)


model = TorchCRNN(five_k_flag,stats_select).to(device)
model.load_state_dict(torch.load(args.modelFolder))
db_test= datasetsplit(x_test, test_ratio=0, flag_get=five_k_flag,stats=stats_select)
x_index = [_[0] for _ in x_test[:, 0, 0]]

try:
    del res_pred
    print('delete')
except:
    pass

true = []
with torch.no_grad():
    model.eval()
    for idx, x in enumerate(db_test):
        x = x[0].squeeze(-1).type(torch.FloatTensor).to(device)
        pred = model(x)
        try:
            res_pred = np.concatenate((res_pred, np.array(pred.cpu())), axis=0)
        except:
            res_pred = np.array(pred.cpu())
            print('respred created')
            
result = pd.DataFrame(res_pred)
result.index = x_index
result.columns = ['log2(TPM+1)']
print(result)
result.to_csv(savepath + args.sample + '_result.csv')