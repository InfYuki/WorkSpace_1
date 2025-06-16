import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from Data_process import DC_labels_tensor, device
from feature_extract.CE import EK_tensor
from feature_extract.Word2vec import w2c_tensor
from feature_extract.token_encoding import K_num_tensor
from feature_extract.BERT import bert_tensor
from feature_extract.BDGraph import get_graph_datasets
from torch_geometric.data import DataLoader

from model2 import model
from utils import cal_score, Dataset2
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


def train_model(model, train_loader, graph_loader, criterion, optimizer, device):
    model.train()
    pred_list = []
    label_list = []

    for (features1, features3, labels), graph_batch in zip(train_loader, graph_loader):
        features1 = torch.tensor(features1, dtype=torch.float)
        features3 = torch.tensor(features3, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.float)

        features1 = features1.to(device)
        graph_batch = graph_batch.to(device)
        features3 = features3.to(device)
        labels = labels.to(device)

        outputs = model(features1, graph_batch, features3).to(device)
        loss = criterion(outputs.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        outputs = torch.where(outputs > 0.5, torch.tensor(1., device=device), torch.tensor(0., device=device))
        pred_list.extend(outputs.squeeze().cpu().detach().numpy())
        label_list.extend(labels.squeeze().cpu().detach().numpy())

    print("train_loss", loss)

    score = cal_score(label_list, pred_list)
    return score


def vail(model, vail_loader, graph_loader, criterion, device):
    model.eval()
    pred_list = []
    label_list = []

    with torch.no_grad():
        for (features1, features3, labels), graph_batch in zip(vail_loader, graph_loader):
            features1 = torch.tensor(features1, dtype=torch.float)
            features3 = torch.tensor(features3, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.float)

            features1 = features1.to(device)
            graph_batch = graph_batch.to(device)
            features3 = features3.to(device)
            labels = labels.to(device)

            outputs = model(features1, graph_batch, features3).to(device)
            loss = criterion(outputs.squeeze(), labels)
            outputs = torch.where(outputs > 0.5, torch.tensor(1., device=device), torch.tensor(0., device=device))

            pred_list.extend(outputs.squeeze().cpu().detach().numpy())
            label_list.extend(labels.squeeze().cpu().detach().numpy())
        print("test_loss", loss)

        score = cal_score(label_list, pred_list)

    return score


''' 交叉验证'''
batch_size = 128
criterion = nn.BCEWithLogitsLoss()

# 获取图数据集
train_graph_dataset, test_graph_dataset = get_graph_datasets()

# 使用BERT特征作为x1
DC_tensor = bert_tensor
# 使用图数据作为x2 (已经通过get_graph_datasets获取)
# 使用K-mer编码作为x3
DC3_tensor = K_num_tensor

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
learning_rate = 0.0005

for fold, (train_indices, val_indices) in enumerate(kf.split(DC_tensor, DC_labels_tensor)):
    print(f'第{fold + 1}折：', fold + 1)
    num_val = 0
    num_train = 0
    best_score = 0.0

    # 创建模型
    model3 = model(out_channels=16, kernel_size=3, stride=1, hidden_size=12).to(device)
    optimizer = torch.optim.Adam(model3.parameters(), lr=learning_rate, weight_decay=5e-05)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)

    # 分割特征和标签
    train_features1, val_features1 = DC_tensor[train_indices], DC_tensor[val_indices]
    train_features3, val_features3 = DC3_tensor[train_indices], DC3_tensor[val_indices]
    train_labels, val_labels = DC_labels_tensor[train_indices], DC_labels_tensor[val_indices]

    # 分割图数据集
    train_graph_subset = torch.utils.data.Subset(train_graph_dataset, train_indices)
    val_graph_subset = torch.utils.data.Subset(train_graph_dataset, val_indices)

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        list(zip(train_features1, train_features3, train_labels)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = torch.utils.data.DataLoader(
        list(zip(val_features1, val_features3, val_labels)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    # 创建图数据加载器
    train_graph_loader = DataLoader(
        train_graph_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    val_graph_loader = DataLoader(
        val_graph_subset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    all_train_score = []
    all_val_score = []

    for epoch in range(150):
        scheduler.step()

        # 训练
        print('------------第{}轮训练开始---------------'.format(epoch + 1))
        train_score = train_model(model3, train_loader, train_graph_loader, criterion, optimizer, device)
        print('Learning Rate:', optimizer.param_groups[0]['lr'])
        print('\n')
        all_train_score.append(train_score)

        num_train += train_score

        # 测试
        print('------------第{}轮验证开始---------------'.format(epoch + 1))
        vail_score = vail(model3, val_loader, val_graph_loader, criterion, device)
        print(
            f"Epoch {epoch + 1}, Learning Rate: {optimizer.param_groups[0]['lr']}, batchsize:{batch_size}")
        print("vail_score:", vail_score)
        print('\n')

        if vail_score > best_score:
            best_score = vail_score
            best_model_state_dict = model3.state_dict()

torch.save(best_model_state_dict, "model_save.pth")