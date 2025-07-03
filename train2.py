import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
from Data_process import DC_labels_tensor, device
#from feature_extract.CE import EK_tensor
#from feature_extract.Word2vec import w2c_tensor

from feature_extract.BERT import Bert_out
from feature_extract.BDGraph import get_graph_datasets
from feature_extract.Bio_feature import Bio_feature_out

from torch_geometric.data import DataLoader

from model2 import model
from utils import cal_score, Dataset2
import numpy as np
import random
import warnings

import argparse

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

    return score, loss


def main_worker(args):

    ''' 交叉验证'''
    batch_size = args.batch_size
    criterion = nn.BCEWithLogitsLoss()

    # 获取图数据集
    train_graph_dataset, test_graph_dataset = get_graph_datasets(args.dataset)

    # 使用BERT特征作为x1
    DC_tensor = Bert_out(args.dataset)
    # 使用图数据作为x2 (已经通过get_graph_datasets获取)
    # 使用Bio_feature作为x3
    DC3_tensor = Bio_feature_out(args.dataset)
    #DC3_tensor = EK_tensor

    kf = StratifiedKFold(n_splits=args.KFold, shuffle=True, random_state=args.seed)

    for fold, (train_indices, val_indices) in enumerate(kf.split(DC_tensor, DC_labels_tensor)):
        print(f'第{fold + 1}折：', fold + 1)
        num_val = 0
        num_train = 0
        best_score = 0.0
        best_epoch = 0

        # 早停相关变量
        patience = args.patience  # 早停耐心值
        counter = 0               # 计数器
        best_val_loss = float('inf')  # 最佳验证损失
        early_stop = False        # 早停标志

        # 创建模型
        model3 = model(out_channels=args.out_channels, kernel_size=args.kernel_size, stride=args.stride, hidden_size=args.hidden_size).to(device)
        optimizer = torch.optim.Adam(model3.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

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
        all_val_loss = []

        for epoch in range(args.epoch):
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
            vail_score, val_loss = vail(model3, val_loader, val_graph_loader, criterion, device)
            print(
                f"Epoch {epoch + 1}, Learning Rate: {optimizer.param_groups[0]['lr']}, batchsize:{batch_size}")
            print("vail_score:", vail_score)
            print('\n')


            all_val_score.append(vail_score)
            all_val_loss.append(val_loss)

            # 保存最佳模型
            if vail_score > best_score:
                best_score = vail_score
                best_epoch = epoch + 1
                best_model_state_dict = model3.state_dict()
                torch.save(best_model_state_dict, "model_save.pth")
                counter = 0  # 重置计数器
            else:
                counter += 1  # 增加计数器

            # 早停检查
            if val_loss < best_val_loss:
                best_val_loss = val_loss


            if counter >= patience:
                print(f"Early stopping triggered! No improvement for {patience} consecutive epochs.")
                print(f"Best validation score: {best_score} at epoch {best_epoch}")
                early_stop = True
                break

            torch.cuda.empty_cache()

        # 打印最佳结果
        if not early_stop:
            print(f"Training completed for all {args.epoch} epochs.")
        print(f"Best validation score: {best_score} at epoch {best_epoch}")

    #torch.save(best_model_state_dict, "model_save.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--runs', type=int, default=10) #5 for penn
    parser.add_argument('--dataset', default='Dataset_mouse')
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--KFold', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    # 创建模型
    #model3 = model(out_channels=16, kernel_size=3, stride=1, hidden_size=12).to(device)
    parser.add_argument('--out_channels', type=int, default=16)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=12)
    #optimizer = torch.optim.Adam(model3.parameters(), lr=learning_rate, weight_decay=5e-05)
    parser.add_argument('--learning_rate', type=int, default=0.001)
    parser.add_argument('--weight_decay', type=int, default=5e-04)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.5)
    parser.add_argument('--step_size', type=int, default=35)
    parser.add_argument('--gamma', type=int, default=0.5)

    # 早停参数
    parser.add_argument('--patience', type=int, default=80)            # 添加早停耐心参数


    args = parser.parse_args()

    main_worker(args)

