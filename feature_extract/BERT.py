import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os
import sys
from sklearn.preprocessing import StandardScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 获取项目根目录的路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 加载数据
train_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_positive.npy')
train_seq_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_negative.npy')
train_label_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_label_positive.npy')
train_label_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_label_negative.npy')

test_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_seq_positive.npy')
test_seq_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_seq_negative.npy')
test_label_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_label_positive.npy')
test_label_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_label_negative.npy')
# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

print(device)

# 加载训练数据
train_pos_sequences = np.load(train_seq_positive_path)
train_pos_sequences = train_pos_sequences.tolist()
train_neg_sequences = np.load(train_seq_negative_path)
train_neg_sequences = train_neg_sequences.tolist()
train_sequences = np.concatenate([train_pos_sequences, train_neg_sequences], axis=0)

# 加载测试数据
test_pos_sequences = np.load(test_seq_positive_path)
test_pos_sequences = test_pos_sequences.tolist()
test_neg_sequences = np.load(test_seq_negative_path)
test_neg_sequences = test_neg_sequences.tolist()
test_sequences = np.concatenate([test_pos_sequences, test_neg_sequences], axis=0)


# 定义一个将DNA序列转为BERT可接受的字符串的函数
def dna_to_text(seq):
    # 每个碱基作为一个"词"
    return " ".join(list(seq))


# 使用transformers的DNABERT或直接用预训练的BERT
model_name = "zhihan1996/DNA_bert_6"  # 这是一个针对DNA序列的BERT模型
cache_dir = "bert_model"

# 创建缓存目录
os.makedirs(cache_dir, exist_ok=True)

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
bert_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
bert_model = bert_model.to(device)
bert_model.eval()


# 定义获取BERT编码的函数，直接输出固定形状的张量
def get_bert_embeddings(sequences, seq_length=41, output_dim=8):
    all_embeddings = []

    with torch.no_grad():
        for seq in tqdm(sequences):
            # 将DNA序列转为文本
            text = dna_to_text(seq)

            # 将文本转为BERT的输入格式
            inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                               max_length=seq_length + 2, truncation=True)  # +2 是为了[CLS]和[SEP]
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 通过BERT获取编码
            outputs = bert_model(**inputs)

            # 获取所有token的表示（不包括[CLS]和[SEP]）
            token_embeddings = outputs.last_hidden_state[:, 1:-1, :]  # 跳过[CLS]和[SEP]

            # 确保我们有41个token
            if token_embeddings.size(1) < seq_length:
                # 如果token数量不足，补零
                padding = torch.zeros(1, seq_length - token_embeddings.size(1),
                                      token_embeddings.size(2), device=device)
                token_embeddings = torch.cat([token_embeddings, padding], dim=1)
            elif token_embeddings.size(1) > seq_length:
                # 如果token数量过多，截断
                token_embeddings = token_embeddings[:, :seq_length, :]

            # 将每个token的768维表示映射到4维
            # 我们可以使用简单的平均池化或者线性投影
            # 这里使用平均池化，将768维分为4组，每组取平均
            group_size = token_embeddings.size(2) // output_dim
            reduced_embeddings = torch.zeros(1, seq_length, output_dim, device=device)

            for i in range(output_dim):
                start_idx = i * group_size
                end_idx = (i + 1) * group_size if i < output_dim - 1 else token_embeddings.size(2)
                reduced_embeddings[:, :, i] = token_embeddings[:, :, start_idx:end_idx].mean(dim=2)

            # 转换为numpy数组
            seq_embedding = reduced_embeddings.cpu().numpy()
            all_embeddings.append(seq_embedding[0])  # 取出batch维度

    return np.array(all_embeddings)


# 使用BERT获取训练集和测试集的编码
print("Encoding training sequences...")
train_embeddings = get_bert_embeddings(train_sequences)
print("Encoding test sequences...")
test_embeddings = get_bert_embeddings(test_sequences)

# 此时train_embeddings和test_embeddings的形状应该是 [样本数, 41, 4]
print("Train embeddings shape:", train_embeddings.shape)
print("Test embeddings shape:", test_embeddings.shape)

# 标准化
sc = StandardScaler()
train_flat = train_embeddings.reshape(train_embeddings.shape[0], -1)
test_flat = test_embeddings.reshape(test_embeddings.shape[0], -1)

sc.fit(train_flat)
train_flat = sc.transform(train_flat)
test_flat = sc.transform(test_flat)

# 重塑回[样本数, 41, 4]
bert_tensor = torch.tensor(train_flat.reshape(train_embeddings.shape), dtype=torch.float)
bert_test_tensor = torch.tensor(test_flat.reshape(test_embeddings.shape), dtype=torch.float)

print("BERT embeddings final shape:", bert_tensor.shape)
print("BERT test embeddings final shape:", bert_test_tensor.shape)