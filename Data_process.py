import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random
def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

random_seed(42)
''' 训练数据 '''
train_seq_positive_path = 'data/Dataset_mouse/npy/train_seq_positive.npy'
train_label_positive_path = 'data/Dataset_mouse/npy/train_label_positive.npy'
train_seq_negative_path = 'data/Dataset_mouse/npy/train_seq_negative.npy'
train_label_negative_path = 'data/Dataset_mouse/npy/train_label_negative.npy'

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

DC_train_labels = np.concatenate([train_label_positive, train_label_negative], axis=0)
DC_labels_tensor=torch.tensor(DC_train_labels)

