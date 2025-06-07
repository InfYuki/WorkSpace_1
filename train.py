import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from Data_process import DC_labels_tensor,device
from feature_extract.CE import EK_tensor
from feature_extract.Word2vec import w2c_tensor
from feature_extract.token_encoding import K_num_tensor

from feature_extract.BERT import bert_tensor

from model import model
from utils import cal_score,Dataset2
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_seed(42)

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    pred_list = []
    label_list = []

    for features1,features2,features3, labels in tqdm(train_loader):
        features1 = torch.tensor(features1, dtype=torch.float)
        features2= torch.tensor(features2, dtype=torch.float)
        features3 = torch.tensor(features3, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)
        features1 = features1.to(device)
        features2 = features2.to(device)
        features3 = features3.to(device)
        labels = labels.to(device)
        outputs = model(features1,features2,features3).to(device)
        loss = criterion(outputs.squeeze(), labels )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = torch.where(outputs > 0.5, torch.tensor(1., device=device), torch.tensor(0., device=device))
        pred_list.extend(outputs.squeeze().cpu().detach().numpy())
        label_list.extend(labels.squeeze().cpu().detach().numpy())

    print("train_loss", loss)

    score = cal_score(label_list, pred_list)
    return score

def vail(model, vail_loader, criterion, device):
    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for features1,features2,features3, labels in tqdm(vail_loader):
            features1 = torch.tensor(features1, dtype=torch.float)
            features2 = torch.tensor(features2, dtype=torch.float)
            features3 = torch.tensor(features3, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.float)
            features1 = features1.to(device)
            features2 = features2.to(device)
            features3 = features3.to(device)
            labels = labels.to(device)
            outputs = model(features1,features2,features3).to(device)
            loss = criterion(outputs.squeeze(), labels)
            outputs = torch.where(outputs > 0.5, torch.tensor(1., device=device), torch.tensor(0., device=device))

            pred_list.extend(outputs.squeeze().cpu().detach().numpy())
            label_list.extend(labels.squeeze().cpu().detach().numpy())
        print("test_loss", loss)

        score = cal_score(label_list, pred_list)

    return score

''' 交叉验证'''
batch_size =128
criterion = nn.BCEWithLogitsLoss()

#DC_tensor =w2c_tensor
DC_tensor = bert_tensor
DC2_tensor=EK_tensor
DC3_tensor=K_num_tensor

kf =StratifiedKFold (n_splits=10, shuffle=True, random_state=42)
learning_rate = 0.0005
for fold, (train_indices, val_indices) in enumerate(kf.split(DC_tensor, DC_labels_tensor)):
    print(f'第{fold+1}折：',fold+1)
    num_val = 0
    num_train = 0
    best_score = 0.0

    model3 = model(out_channels=16,kernel_size=3,stride=1,hidden_size=12).to(device) # 12:平均0.77
    optimizer = torch.optim.Adam(model3.parameters(), lr=learning_rate, weight_decay=5e-05)  # 5e-04:0.759
    scheduler = lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)
    train_features1, val_features1 = DC_tensor[train_indices], DC_tensor[val_indices]
    train_features2, val_features2 = DC2_tensor[train_indices], DC2_tensor[val_indices]
    train_features3, val_features3 = DC3_tensor[train_indices], DC3_tensor[val_indices]
    train_labels, val_labels = DC_labels_tensor[train_indices], DC_labels_tensor[val_indices]

    train_dataset = Dataset2(train_features1,train_features2,train_features3, train_labels)
    val_dataset = Dataset2(val_features1,val_features2,val_features3, val_labels)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,drop_last=True,)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True,drop_last=True,)

    all_train_score = []
    all_val_score = []

    for epoch in range(100):
        scheduler.step()

        # 训练
        print('------------第{}轮训练开始---------------'.format(epoch + 1))
        train_score = train_model(model3, train_loader, criterion, optimizer, device)
        print('Learning Rate:',optimizer.param_groups[0]['lr'])
        print('\n')
        all_train_score.append(train_score)



        num_train += train_score

        # 测试
        print('------------第{}轮验证开始---------------'.format(epoch + 1))
        vail_score = vail(model3, val_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}, Learning Rate: {optimizer.param_groups[0]['lr']},batchsize:{batch_size}")
        print("vail_score:", vail_score)
        print('\n')

        if vail_score > best_score:
            best_score = vail_score
            best_model_state_dict = model3.state_dict()

torch.save(best_model_state_dict, "model_save.pth")
