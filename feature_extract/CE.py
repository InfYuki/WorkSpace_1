import pandas as pd
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

''' 训练数据 '''
train_seq_positive_path = 'data/Dataset_mouse/npy/train_seq_positive.npy'
train_label_positive_path = 'data/Dataset_mouse/npy/train_label_positive.npy'
train_seq_negative_path = 'data/Dataset_mouse/npy/train_seq_negative.npy'
train_label_negative_path = 'data/Dataset_mouse/npy/train_label_negative.npy'

seed = 42
torch.manual_seed(seed)

print(device)


train_pos_sequences = np.load(train_seq_positive_path)
#print(train_pos_sequences)
# 查看数据类型和形状
#print("Data type:", type(train_pos_sequences))
#print("Shape:", train_pos_sequences.shape)
train_pos_sequences=train_pos_sequences.tolist()
train_neg_sequences = np.load(train_seq_negative_path)
train_neg_sequences=train_neg_sequences.tolist()
train_sequences = np.concatenate([train_pos_sequences,train_neg_sequences ], axis=0)  # 按行进行合并

train_label_positive=np.load(train_label_positive_path)
train_label_negative=np.load(train_label_negative_path)

test_seq_positive_path = 'data/Dataset_mouse/npy/test_seq_positive.npy'
test_label_positive_path = 'data/Dataset_mouse/npy/test_label_positive.npy'
test_seq_negative_path = 'data/Dataset_mouse/npy/test_seq_negative.npy'
test_label_negative_path = 'data/Dataset_mouse/npy/test_label_negative.npy'

# 序列
test_pos_sequences = np.load(test_seq_positive_path)
test_pos_sequences=test_pos_sequences.tolist()
test_neg_sequences = np.load(test_seq_negative_path)
test_neg_sequences=test_neg_sequences.tolist()
test_sequences = np.concatenate([test_pos_sequences,test_neg_sequences ], axis=0)  # 按行进行合并

def EIIP(seq):
    std = {"A": 0.12601,
           "T": 0.13400,
           "C": 0.08060,
           "G": 0.13350}
    res = []
    for x in seq:
        res.append(std[x])
    return np.array(res)

def numerical_transform(seq):
    std = {"A": 0,
           "G": 1,
           "C": 2,
           "T": 3,
           }
    res = []
    for i, x in enumerate(seq):
        res.append(std[x])
    return np.array(res)

def get_features(seq):
    res1 = numerical_transform(seq)
    res2 = EIIP(seq)
    res = np.concatenate([res1,res2],axis=0)

    return np.array(res).flatten()

data_EK=[]
for seq in train_sequences:
    data_EK.append(get_features(seq))
data_EK=np.array(data_EK)

data_test_EK=[]
for seq in test_sequences:
    data_test_EK.append(get_features(seq))
data_test_EK=np.array(data_test_EK)

EK_tensor= torch.tensor(data_EK, dtype=torch.float)
EK_test_tensor= torch.tensor(data_test_EK, dtype=torch.float)



