import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from Data_process import  device
from model import model
from feature_extract.CE import EK_test_tensor
from feature_extract.Word2vec import w2c_test_tensor
from feature_extract.token_encoding import K_num_test_tensor
from utils import Dataset2,cal_score

import warnings
warnings.filterwarnings('ignore')

test_seq_positive_path = 'data/Dataset_mouse/npy/test_seq_positive.npy'
test_label_positive_path = 'data/Dataset_mouse/npy/test_label_positive.npy'
test_seq_negative_path = 'data/Dataset_mouse/npy/test_seq_negative.npy'
test_label_negative_path = 'data/Dataset_mouse/npy/test_label_negative.npy'
criterion = nn.BCEWithLogitsLoss()

batch_size =128
# 序列
test_pos_sequences = np.load(test_seq_positive_path)
test_pos_sequences=test_pos_sequences.tolist()
test_neg_sequences = np.load(test_seq_negative_path)
test_neg_sequences=test_neg_sequences.tolist()
test_sequences = np.concatenate([test_pos_sequences,test_neg_sequences ], axis=0)  # 按行进行合并
# 标签
test_label_positive=np.load(test_label_positive_path)
test_label_negative=np.load(test_label_negative_path)
test_labels = np.concatenate([test_label_positive, test_label_negative], axis=0)
test_labels_tensor=torch.tensor(test_labels)
test_DC_tensor =w2c_test_tensor
test_DC2_tensor=EK_test_tensor
test_DC3_tensor=K_num_test_tensor

def test_model(model, test_loader, device):
    model.eval()
    pred_list = []
    label_list = []
    with torch.no_grad():
        for features1,features2,features3, labels in tqdm(test_loader):
            features1 = torch.tensor(features1, dtype=torch.float)
            features2 = torch.tensor(features2, dtype=torch.float)
            features3 = torch.tensor(features3, dtype=torch.float)
            labels = torch.tensor(labels, dtype=torch.float)
            features1 = features1.to(device)
            features2 = features2.to(device)
            features3 = features3.to(device)
            labels = labels.to(device)
            outputs = model(features1,features2,features3)
            outputs = torch.where(outputs > 0.5, torch.tensor(1., device=device), torch.tensor(0., device=device))

            pred_list.extend(outputs.squeeze().cpu().detach().numpy())
            label_list.extend(labels.squeeze().cpu().detach().numpy())

        score = cal_score(label_list, pred_list)
    return score

# 独立测试
DCtest_dataset = Dataset2(test_DC_tensor,test_DC2_tensor, test_DC3_tensor,test_labels_tensor)
DCtest_loader = torch.utils.data.DataLoader(DCtest_dataset, batch_size=batch_size, shuffle=True)

model=model(out_channels=16,kernel_size=3,stride=1,hidden_size=12).to(device)
model_state_dict=torch.load('model_save.pth')
model.load_state_dict(model_state_dict)
test_score = test_model(model, DCtest_loader, device)
print("------------------test--------------------:",test_score)
