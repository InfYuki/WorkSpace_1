import pandas as pd
import torch
import numpy as np
def dna_2mer_encoding(dna_sequence):
    encoding_rule = {'AT': 0, 'TC': 1, 'GA': 2, 'TG': 3,
                     'CA': 4, 'GT': 5, 'AC': 6, 'CT': 7,
                     'GC': 8, 'CG': 9, 'TT': 10, 'AA': 11,
                     'GG': 12, 'CC': 13, 'TA': 14, 'AG': 15}

    encoded_sequence = []
    for i in range(len(dna_sequence) - 1):
        two_mer = dna_sequence[i:i + 2]
        if two_mer in encoding_rule:
            encoded_sequence.append(encoding_rule[two_mer])

    return encoded_sequence

def get_features(seq):
    res = dna_2mer_encoding(seq)
    return np.array(res).flatten()

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
train_pos_sequences=train_pos_sequences.tolist()
train_neg_sequences = np.load(train_seq_negative_path)
train_neg_sequences=train_neg_sequences.tolist()
train_sequences = np.concatenate([train_pos_sequences,train_neg_sequences ], axis=0)

test_seq_positive_path = 'data/Dataset_mouse/npy/test_seq_positive.npy'
test_label_positive_path = 'data/Dataset_mouse/npy/test_label_positive.npy'
test_seq_negative_path = 'data/Dataset_mouse/npy/test_seq_negative.npy'
test_label_negative_path = 'data/Dataset_mouse/npy/test_label_negative.npy'


#
# 序列
test_pos_sequences = np.load(test_seq_positive_path)
test_pos_sequences=test_pos_sequences.tolist()
test_neg_sequences = np.load(test_seq_negative_path)
test_neg_sequences=test_neg_sequences.tolist()
test_sequences = np.concatenate([test_pos_sequences,test_neg_sequences ], axis=0)

kmer_num=[]
for seq in train_sequences:
    kmer_num.append(dna_2mer_encoding(seq))
kmer_num=np.array(kmer_num)
K_num_tensor=torch.tensor(kmer_num)


kmer_num_test=[]
for seq in test_sequences:
    kmer_num_test.append(dna_2mer_encoding(seq))
kmer_num_test=np.array(kmer_num_test)
K_num_test_tensor=torch.tensor(kmer_num_test)
