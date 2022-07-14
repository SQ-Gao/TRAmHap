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
# config = Configs.config()

parser = argparse.ArgumentParser()
parser.add_argument('-S', "--sample", required=True,  help="Sample.")
parser.add_argument('-I', "--dataFolder", required=True, help="A folder with processed data.")
parser.add_argument('-O', "--outFolder", required=True, help="Output folder.")
parser.add_argument('-M', "--modelFolder", required=True, help="A folder with model save.")
parser.add_argument('-E', "--epoch", default=90, help="Number of epoch.")
parser.add_argument('-K', "--flag", default="all", help="Window size of inpur gene.")
parser.add_argument('-T', "--statistics", default="all", help="which statistics to train, for example '0,2' seperate by ,")

args = parser.parse_args()

# create output folder
#if not os.path.exists(args.outFolder):
#    sys.exit('Output folder not found.')

if not os.path.exists(args.outFolder):
    os.makedirs(args.outFolder)
if not os.path.exists(args.modelFolder):
    os.makedirs(args.modelFolder)


'''
新加的
'''
if  args.statistics =='all':
    stats_select = [0,1,2]
else:
    stats_select= [int(_) for _ in args.statistics.split(sep = ',')]


name_select = args.sample
model_save_path = args.modelFolder + '/model_' + name_select + '.pkl'
datapath = args.dataFolder + '/'
fig_savepath = args.outFolder + '/'
ep = int(args.epoch) #epoch
lr = 1e-4 #leariningrate

try:
    five_k_flag = int(args.flag)
except:
    five_k_flag = args.flag # 5 或者 50 或者 "all"

gooddata = []
totaldataname = os.listdir(datapath)
nameset = set([i[2:] for i in os.listdir(datapath)])
for name in nameset:
    if ('x_' + name in totaldataname) and ('y_' + name in totaldataname):
        gooddata += ['x_' + name, 'y_' + name]
    else:
        print(name + ' bad data')
x_train_list = []
y_train_list = []
x_test_list = []
y_test_list = []

try:
    del x_test, x_train, y_test, y_train
except:
    pass

for i in gooddata:
    if name_select in i:
        if i[0] == 'x':
            x_test_list.append(i)
        elif i[0] == 'y':
            y_test_list.append(i)
    else:
        if i[0] == 'x':
            x_train_list.append(i)
        elif i[0] == 'y':
            y_train_list.append(i)

for i in x_train_list:
    tmp_x = np.load(datapath + i, allow_pickle=True)
    try:
        x_train = np.concatenate([x_train, tmp_x], axis=0)
    except:
        print('create xtrain')
        x_train = tmp_x

for i in x_test_list:
    tmp_x = np.load(datapath + i, allow_pickle=True)
    try:
        x_test = np.concatenate([x_test, tmp_x], axis=0)
    except:
        print('create xtest')
        x_test = tmp_x

for i in y_train_list:
    tmp_y = np.load(datapath + i, allow_pickle=True)
    try:
        y_train = np.concatenate([y_train, tmp_y], axis=0)
    except:
        print('create ytrain')
        y_train = tmp_y

for i in y_test_list:
    tmp_y = np.load(datapath + i, allow_pickle=True)
    try:
        y_test = np.concatenate([y_test, tmp_y], axis=0)
    except:
        print('create ytest')
        y_test = tmp_y


def evaluate(model, loader, c=nn.MSELoss()):
    model.eval()
    batch_num = 0
    loss = 0
    for x_e, y_e in loader:
        x_e, y_e = x_e.squeeze(-1).type(torch.FloatTensor).to(device), y_e.type(torch.FloatTensor).to(device)
        criteon = c
        with torch.no_grad():
            pred_e = model(x_e)
        loss += criteon(pred_e, y_e).item()
        batch_num += 1
    return loss / batch_num

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
            nn.Linear(100, 3),
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

def datasetsplit(x, y, test_ratio=0.2, shuffle=True, flag_get='all',stats = stats_select):
    if test_ratio == 0:
        x, y_float = torch.from_numpy(getx(x, flag_get,stats).astype(float)).unsqueeze(-1).transpose(2, 3).transpose(1,
                                                                                                               2), torch.from_numpy(
            y[:, 1:].astype(float))
        data = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y_float), batch_size=64, shuffle=False)
        return data, y
    else:
        np.random.seed(888)
        assert (x.shape[0] == y.shape[0])
        dlen = x.shape[0]
        shuffled_indices = np.random.permutation(dlen)
        test_set_size = int(dlen * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        # valid_indices = shuffled_indices[test_set_size:test_set_size+valid_set_size]
        train_indices = shuffled_indices[test_set_size:]

        xtrain, ytrain = torch.from_numpy(getx(x[[train_indices]], flag_get,stats).astype(float)).unsqueeze(-1).transpose(2,
                                                                                                                    3).transpose(
            1, 2), torch.from_numpy(y[[train_indices]][:, 1:].astype(float))
        # xvalid,yvalid = torch.from_numpy(x[[valid_indices]][:,1:,:].astype(float)).unsqueeze(-1).transpose(2,3).transpose(1,2),torch.from_numpy(y[[valid_indices]][:,1:].astype(float))
        xtest, ytest = torch.from_numpy(getx(x[[test_indices]], flag_get,stats).astype(float)).unsqueeze(-1).transpose(2,
                                                                                                                 3).transpose(
            1, 2), torch.from_numpy(y[[test_indices]][:, 1:].astype(float))
        train = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtrain, ytrain), batch_size=64, shuffle=True)
        # valid = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xvalid,yvalid),batch_size = 128,shuffle = True)
        test = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xtest, ytest), batch_size=128,
                                           shuffle=shuffle)
        print('Finish with train shape:', len(train.dataset), ',test shape:', len(test.dataset))
        return train, test


db_train, db_valid = datasetsplit(x_train, y_train, flag_get=five_k_flag)

lr_0 = lr
decay_param = 0.98
model = TorchCRNN(five_k_flag,stats_select).to(device)
# optimizer = torch.optim.SGD(model.parameters(),lr = lr_0)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_0, betas=[0.5, 0.999])

Exp_lr = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_param)
criteon = nn.MSELoss()

best_mse = 10
train_loss = []
valid_loss = []
for epoch in range(ep):
    step_loss = 0
    step_num = 0
    for step, (x, y) in enumerate(db_train):
        model.train()
        x, y = x.squeeze(-1).type(torch.FloatTensor).to(device), y.type(torch.FloatTensor).to(device)
        pred = model(x)
        loss = criteon(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss += loss.item()

        step_num += 1
        if step % 100 == 0:
            print('epoch:', epoch, ',step:', step, ',mean_step_loss:', step_loss / step_num)
            step_loss = 0
            step_num = 0
    train_loss.append(loss.item())
    test_mse = evaluate(model, db_valid)
    valid_loss.append(test_mse)
    if test_mse < best_mse:
        best_mse = test_mse
        torch.save(model.state_dict(), model_save_path)
        print('********best model generated!********')
    print('------- epoch:', epoch, ',test error:', test_mse, '-------')
    Exp_lr.step()

model = TorchCRNN(five_k_flag,stats_select).to(device)
model.load_state_dict(torch.load(model_save_path))
db_test, y0 = datasetsplit(x_test, y_test, test_ratio=0, flag_get=five_k_flag,stats=stats_select)

try:
    del res_true, res_pred
    print('delete')
except:
    pass

true = []
with torch.no_grad():
    model.eval()
    for idx, (x, y) in enumerate(db_test):
        x, y = x.squeeze(-1).type(torch.FloatTensor).to(device), y
        pred = model(x)
        try:
            res_pred = np.concatenate((res_pred, np.array(pred.cpu())), axis=0)
        except:
            res_pred = np.array(pred.cpu())
            print('respred created')
        try:
            res_true = np.concatenate((res_true, np.array(y)), axis=0)
        except:
            res_true = np.array(y)
            print('resture created')

res_true = pd.DataFrame(res_true)
res_pred = pd.DataFrame(res_pred)

plt.plot(valid_loss)
plt.savefig(fig_savepath + name_select + "_valid_loss.jpg")
plt.close()


result = pd.concat([res_true, res_pred], axis=1)
result.columns = ['T_H3K27ac', 'T_H3K4me3', 'T_log2(TPM+1)', 'P_pred1', 'P_pred2', 'P_pred3']
# res = []
title = ['H3K27ac', 'H3K4me3', 'log2(TPM+1)']
plt.figure(figsize=(15, 15))
plt.title('before transfer')
for i in range(3):
    y_p_i = np.array(result.iloc[:, i + 3])
    y_r_i = np.array(result.iloc[:, i])
    lineh = max(y_p_i)
    place = int(311 + i)
    plt.subplot(place)
    plt.scatter(y_p_i, y_r_i, alpha=0.5, c='lightpink')
    plt.plot([0, lineh], [0, lineh], c='lightskyblue')
    plt.xlabel('pred val')
    plt.ylabel('true val')
    plt.title(title[i] + ' corr:' + str(np.round(np.corrcoef(y_p_i, y_r_i)[0][1], 2)))
    plt.savefig(fig_savepath + name_select + '.pdf')

result.index = y0[:, 0]
result = result.reset_index()
result.to_csv(fig_savepath + name_select + '_result.csv', index=False)