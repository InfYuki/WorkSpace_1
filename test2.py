import torch
import numpy as np
from tqdm import tqdm
from model2 import model
from utils import cal_score, Model_Evaluate
from Data_process import device
from feature_extract.BERT import bert_test_tensor
from feature_extract.token_encoding import K_num_test_tensor
from feature_extract.BDGraph import get_graph_datasets
from torch_geometric.data import DataLoader
import warnings

warnings.filterwarnings('ignore')


def test():
    # 加载测试数据
    x1_test = bert_test_tensor  # BERT特征作为x1
    x3_test = K_num_test_tensor  # K-mer编码作为x3

    # 加载图数据
    _, test_graph_dataset = get_graph_datasets()

    # 加载标签
    test_label_positive_path = 'data/Dataset_mouse/npy/test_label_positive.npy'
    test_label_negative_path = 'data/Dataset_mouse/npy/test_label_negative.npy'

    test_label_positive = np.load(test_label_positive_path)
    test_label_negative = np.load(test_label_negative_path)
    test_labels = np.concatenate([test_label_positive, test_label_negative], axis=0)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.float)

    # 创建数据加载器
    batch_size = 128
    test_loader = torch.utils.data.DataLoader(
        list(zip(x1_test, x3_test, test_labels_tensor)),
        batch_size=batch_size,
        shuffle=False
    )

    # 创建图数据加载器
    test_graph_loader = DataLoader(
        test_graph_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    # 加载模型
    model_path = "model_save.pth"
    model_state_dict = torch.load(model_path)

    # 初始化模型
    test_model = model(out_channels=16, kernel_size=3, stride=1, hidden_size=12).to(device)
    test_model.load_state_dict(model_state_dict)
    test_model.eval()

    # 测试
    pred_list = []
    label_list = []

    print("开始测试...")
    with torch.no_grad():
        for (features1, features3, labels), graph_batch in zip(test_loader, test_graph_loader):
            features1 = torch.tensor(features1, dtype=torch.float).to(device)
            features3 = torch.tensor(features3, dtype=torch.float).to(device)
            labels = torch.tensor(labels, dtype=torch.float).to(device)
            graph_batch = graph_batch.to(device)

            outputs = test_model(features1, graph_batch, features3).to(device)
            outputs = torch.where(outputs > 0.5, torch.tensor(1., device=device), torch.tensor(0., device=device))

            pred_list.extend(outputs.squeeze().cpu().detach().numpy())
            label_list.extend(labels.squeeze().cpu().detach().numpy())

    # 将预测结果和标签转换为numpy数组
    pred_array = np.array(pred_list)
    label_array = np.array(label_list)

    # 直接使用Model_Evaluate函数计算所有指标
    sn, sp, acc, mcc = Model_Evaluate(label_array, pred_array)

    print("\n测试结果:")
    print(f"灵敏度 (SN): {sn:.4f}")
    print(f"特异度 (SP): {sp:.4f}")
    print(f"准确率 (ACC): {acc:.4f}")
    print(f"马修斯相关系数 (MCC): {mcc:.4f}")

    return acc


if __name__ == "__main__":
    test()