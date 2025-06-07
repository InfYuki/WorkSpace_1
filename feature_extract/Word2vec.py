from gensim.models import Word2Vec
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_sen_1(seq):
    return list(seq)

''' 训练数据 '''
train_seq_positive_path = 'data/Dataset_mouse/npy/train_seq_positive.npy'
train_label_positive_path = 'data/Dataset_mouse/npy/train_label_positive.npy'
train_seq_negative_path = 'data/Dataset_mouse/npy/train_seq_negative.npy'
train_label_negative_path = 'data/Dataset_mouse/npy/train_label_negative.npy'

seed = 42
torch.manual_seed(seed)

print(device)

train_pos_sequences = np.load(train_seq_positive_path)
train_pos_sequences=train_pos_sequences.tolist()
train_neg_sequences = np.load(train_seq_negative_path)
train_neg_sequences=train_neg_sequences.tolist()
train_sequences = np.concatenate([train_pos_sequences,train_neg_sequences ], axis=0)  # 按行进行合并

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

model1 = Word2Vec([to_sen_1(seq) for seq in train_sequences],
                  vector_size=4,
                  min_count=1,
                  window=1,
                  workers=-1,
                  epochs=256)
model1.save("feature_extract/word2vec_model/4mC-word2vec.model")

from gensim.models import Word2Vec
model_w2c = Word2Vec.load("feature_extract/word2vec_model/4mC-word2vec.model")

def word2vec(seq):
    res = []
    seq_list = list(seq)

    for x in seq_list:
        tmp_w2v = model_w2c.wv[x]
        res.append(list(tmp_w2v))
    return np.array(res)
data_word2vec=[]
for i in train_sequences:
   i=word2vec(i)
   data_word2vec.append(i)
data_word2vec= np.array([sequence.flatten() for sequence in data_word2vec])

data_word2vec_test=[]
for i in test_sequences:
   i=word2vec(i)
   data_word2vec_test.append(i)
data_word2vec_test= np.array([sequence.flatten() for sequence in data_word2vec_test])

w2c_tensor= torch.tensor(data_word2vec, dtype=torch.float)
w2c_test_tensor= torch.tensor(data_word2vec_test, dtype=torch.float)

sc = StandardScaler()
sc.fit(w2c_tensor)
w2c_tensor = sc.transform(w2c_tensor) # 164
w2c_test_tensor = sc.transform(w2c_test_tensor)

w2c_tensor = w2c_tensor.reshape(-1, 41, 4)  # [样本数量，1968]
w2c_test_tensor= w2c_test_tensor.reshape(-1, 41, 4)
