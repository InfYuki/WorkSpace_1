import pandas as pd
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
import sys
import math
from itertools import product  # 添加这一行导入product函数
from collections import Counter

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取项目根目录的路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))




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


def CKSNAP(seq, k=2):
    """
    k空间核酸对组成
    计算序列中间隔k个位置的核酸对出现频率
    """
    nucleotides = ['A', 'C', 'G', 'T']
    kmer_pairs = [''.join(pair) for pair in product(nucleotides, repeat=2)]

    feature_vector = []

    for gap in range(k + 1):  # 包括间隔0,1,...,k
        pair_count = {pair: 0 for pair in kmer_pairs}
        total_pairs = 0

        for i in range(len(seq) - gap - 1):
            pair = seq[i] + seq[i + gap + 1]
            if pair in pair_count:  # 确保只考虑有效的核苷酸对
                pair_count[pair] += 1
                total_pairs += 1

        # 计算频率并添加到特征向量
        if total_pairs > 0:
            for pair in kmer_pairs:
                feature_vector.append(pair_count[pair] / total_pairs)
        else:
            feature_vector.extend([0] * len(kmer_pairs))

    return np.array(feature_vector)


def DNC(seq):
    """
    二核苷酸组成
    计算序列中所有可能的二核苷酸组合的频率
    """
    nucleotides = ['A', 'C', 'G', 'T']
    dinucleotides = [''.join(pair) for pair in product(nucleotides, repeat=2)]

    # 初始化计数字典
    count_dict = {dinuc: 0 for dinuc in dinucleotides}

    # 计算二核苷酸频率
    for i in range(len(seq) - 1):
        dinuc = seq[i:i + 2]
        if dinuc in count_dict:
            count_dict[dinuc] += 1

    # 计算总二核苷酸数量
    total_count = len(seq) - 1

    # 转换为频率
    freq_vector = []
    for dinuc in dinucleotides:
        if total_count > 0:
            freq_vector.append(count_dict[dinuc] / total_count)
        else:
            freq_vector.append(0)

    return np.array(freq_vector)


def ENAC(seq, window_size=5):
    """
    增强核酸组成
    计算序列中每个核苷酸在滑动窗口内的频率
    """
    nucleotides = ['A', 'C', 'G', 'T']
    seq_length = len(seq)

    # 初始化特征向量
    feature_vector = []

    # 对每个核苷酸计算频率
    for nucleotide in nucleotides:
        # 初始化频率列表
        freq_list = []

        # 对序列中的每个位置计算窗口内的频率
        for i in range(seq_length):
            # 确定窗口范围
            start = max(0, i - window_size // 2)
            end = min(seq_length, i + window_size // 2 + 1)
            window = seq[start:end]

            # 计算窗口内特定核苷酸的频率
            count = window.count(nucleotide)
            freq = count / len(window) if window else 0
            freq_list.append(freq)

        # 将该核苷酸的频率列表添加到特征向量
        feature_vector.extend(freq_list)

    return np.array(feature_vector)

# 核苷酸物理化学性质
physicochemical_properties = {
    'A': [1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5],  # [A, C, G, T, 氢键供体, 氢键受体, 疏水性]
    'C': [0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.3],
    'G': [0.0, 0.0, 1.0, 0.0, 0.5, 1.0, 0.1],
    'T': [0.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.2],
}
# 核苷酸化学性质 (NCP)
def calculate_ncp(sequence):
    """计算核苷酸化学性质"""
    features = []
    for nucleotide in sequence:
        if nucleotide in physicochemical_properties:
            features.extend(physicochemical_properties[nucleotide])
        else:
            features.extend([0] * 7)  # 对于未知核苷酸，使用零向量
    return features

# 核苷酸组成 (NAC)
def calculate_nac(sequence):
    """计算核苷酸组成"""
    count = Counter(sequence)
    total = len(sequence)
    features = []
    for base in ['A', 'C', 'G', 'T']:
        features.append(count[base] / total)
    return features

# 3. 三核苷酸组成 (TNC)
def calculate_tnc(sequence):
    """计算三核苷酸组成"""
    features = []
    total = len(sequence) - 2
    for first in ['A', 'C', 'G', 'T']:
        for second in ['A', 'C', 'G', 'T']:
            for third in ['A', 'C', 'G', 'T']:
                count = 0
                for i in range(total):
                    if sequence[i:i+3] == first + second + third:
                        count += 1
                features.append(count / total)
    return features

def get_features(seq):
    res1 = numerical_transform(seq)
    res2 = EIIP(seq)
    res3 = calculate_nac(seq)
    res = np.concatenate([res1,res2],axis=0)

    return np.array(res).flatten()


def Bio_feature_out(dataset_name):
    ''' 训练数据 '''
    if dataset_name == 'Dataset_mouse':
        train_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_positive.npy')
        train_seq_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_negative.npy')
        train_label_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_label_positive.npy')
        train_label_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_label_negative.npy')

        test_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_seq_positive.npy')
        test_seq_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_seq_negative.npy')
        test_label_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_label_positive.npy')
        test_label_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_label_negative.npy')
    seed = 42
    torch.manual_seed(seed)

    print(device)

    train_pos_sequences = np.load(train_seq_positive_path)
    # print(train_pos_sequences)
    # 查看数据类型和形状
    # print("Data type:", type(train_pos_sequences))
    # print("Shape:", train_pos_sequences.shape)
    train_pos_sequences = train_pos_sequences.tolist()
    train_neg_sequences = np.load(train_seq_negative_path)
    train_neg_sequences = train_neg_sequences.tolist()
    train_sequences = np.concatenate([train_pos_sequences, train_neg_sequences], axis=0)  # 按行进行合并

    train_label_positive = np.load(train_label_positive_path)
    train_label_negative = np.load(train_label_negative_path)

    # 序列
    test_pos_sequences = np.load(test_seq_positive_path)
    test_pos_sequences = test_pos_sequences.tolist()
    test_neg_sequences = np.load(test_seq_negative_path)
    test_neg_sequences = test_neg_sequences.tolist()
    test_sequences = np.concatenate([test_pos_sequences, test_neg_sequences], axis=0)  # 按行进行合并

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

    print(EK_tensor)
    print(EK_tensor.size())

    return EK_tensor

#Bio_feature_out('Dataset_mouse')