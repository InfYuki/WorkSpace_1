import numpy as np
import torch
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib后端为Agg，避免显示问题
matplotlib.use('Agg')
# 导入PyG相关库
import torch_geometric
from torch_geometric.data import Data, Dataset
import networkx as nx  # 仅用于可视化
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA
import pickle

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 碱基对应关系
BASE_PAIRS = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G'
}

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载BERT模型和分词器
model_name = "zhihan1996/DNA_bert_6"  # 这是一个针对DNA序列的BERT模型
cache_dir = "bert_model"

# 创建缓存目录
os.makedirs(cache_dir, exist_ok=True)

# 加载BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
bert_model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
bert_model = bert_model.to(device)
bert_model.eval()

# 全局PCA模型
global_pca = None
PCA_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "global_pca_model.pkl")


def generate_complementary_strand(sequence):
    """生成互补链"""
    complementary = ''
    for base in sequence:
        if base in BASE_PAIRS:
            complementary += BASE_PAIRS[base]
        else:
            complementary += 'N'  # 处理非标准碱基
    return complementary


def fit_global_pca(sequences, n_components=8):
    """在所有序列上拟合一个全局PCA模型

    参数:
        sequences: 序列列表
        n_components: PCA组件数量

    返回:
        全局PCA模型
    """
    global global_pca

    # 如果已经存在PCA模型文件，直接加载
    if os.path.exists(PCA_MODEL_PATH):
        print(f"加载已有的PCA模型: {PCA_MODEL_PATH}")
        with open(PCA_MODEL_PATH, 'rb') as f:
            global_pca = pickle.load(f)
        return global_pca

    print("拟合全局PCA模型...")
    all_embeddings = []

    for sequence in tqdm(sequences, desc="收集BERT嵌入"):
        with torch.no_grad():
            text = " ".join(list(sequence))
            inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                               max_length=len(sequence) + 2, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = bert_model(**inputs)
            token_embeddings = outputs.last_hidden_state[:, 1:-1, :].cpu().numpy()[0]

            for emb in token_embeddings:
                all_embeddings.append(emb)

    # 拟合PCA模型
    print(f"拟合PCA模型，降至{n_components}维...")
    global_pca = PCA(n_components=n_components)
    global_pca.fit(np.array(all_embeddings))

    # 保存PCA模型
    print(f"保存PCA模型到: {PCA_MODEL_PATH}")
    with open(PCA_MODEL_PATH, 'wb') as f:
        pickle.dump(global_pca, f)

    return global_pca


def build_dna_pyg_graph(sequence):
    """将DNA序列构建成PyG图，使用BERT嵌入作为节点特征，并用全局PCA降维到8维

    参数:
        sequence: 碱基序列，例如 'ACGGATTC'

    返回:
        data: PyG的Data对象
    """
    global global_pca

    # 确保全局PCA模型已经加载
    if global_pca is None:
        raise ValueError("全局PCA模型尚未拟合，请先调用fit_global_pca函数")

    # 生成互补链
    complementary = generate_complementary_strand(sequence)

    # 为了效率，先获取整个序列的BERT嵌入
    with torch.no_grad():
        # 将DNA序列转为文本
        text = " ".join(list(sequence))

        # 将文本转为BERT的输入格式
        inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                           max_length=len(sequence) + 2, truncation=True)  # +2 是为了[CLS]和[SEP]
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 通过BERT获取编码
        outputs = bert_model(**inputs)

        # 获取所有token的表示（不包括[CLS]和[SEP]）
        token_embeddings = outputs.last_hidden_state[:, 1:-1, :].cpu().numpy()[0]  # 跳过[CLS]和[SEP]，并转为numpy数组

    # 收集所有节点的原始BERT嵌入
    all_embeddings = []

    # 主链节点嵌入
    for i in range(len(sequence)):
        all_embeddings.append(token_embeddings[i])

    # 互补链节点嵌入（使用主链对应位置的嵌入）
    for i in range(len(complementary)):
        all_embeddings.append(token_embeddings[i])

    # 使用全局PCA进行降维
    all_embeddings_array = np.array(all_embeddings)
    reduced_embeddings = global_pca.transform(all_embeddings_array)

    # 转换为PyTorch张量
    node_features = [torch.tensor(emb, dtype=torch.float) for emb in reduced_embeddings]

    # 边索引
    edge_index = []

    # 主链内部连接
    for i in range(len(sequence) - 1):
        edge_index.append([i, i + 1])
        edge_index.append([i + 1, i])  # 双向边

    # 互补链内部连接
    offset = len(sequence)
    for i in range(len(complementary) - 1):
        edge_index.append([i + offset, i + 1 + offset])
        edge_index.append([i + 1 + offset, i + offset])  # 双向边

    # 碱基对连接
    for i in range(len(sequence)):
        edge_index.append([i, i + offset])
        edge_index.append([i + offset, i])  # 双向边

    # 边类型
    edge_type = []
    # 主链内部连接
    for _ in range(2 * (len(sequence) - 1)):
        edge_type.append(0)  # 0表示主链内部连接

    # 互补链内部连接
    for _ in range(2 * (len(complementary) - 1)):
        edge_type.append(1)  # 1表示互补链内部连接

    # 碱基对连接
    for _ in range(2 * len(sequence)):
        edge_type.append(2)  # 2表示碱基对连接

    # 转换为PyTorch张量
    x = torch.stack(node_features)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_type, dtype=torch.long).unsqueeze(1)  # 添加维度以匹配PyG格式

    # 创建PyG数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # 添加序列信息作为额外属性
    data.sequence = sequence
    data.complementary = complementary

    return data


# 从数据集加载序列并处理
def load_and_process_for_gnn():
    """加载数据并处理为GNN可用的格式"""
    global global_pca

    # 获取项目根目录的路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 训练数据
    train_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_positive.npy')
    train_seq_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_negative.npy')
    train_label_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_label_positive.npy')
    train_label_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_label_negative.npy')

    print(f"加载训练数据: {train_seq_positive_path}")

    # 加载序列和标签
    train_pos_sequences = np.load(train_seq_positive_path).tolist()
    train_neg_sequences = np.load(train_seq_negative_path).tolist()
    train_sequences = np.concatenate([train_pos_sequences, train_neg_sequences], axis=0)

    train_label_positive = np.load(train_label_positive_path)
    train_label_negative = np.load(train_label_negative_path)
    train_labels = np.concatenate([train_label_positive, train_label_negative], axis=0)

    # 拟合全局PCA模型
    fit_global_pca(train_sequences, n_components=8)

    # 创建训练数据集
    train_dataset = DNAGraphDataset(train_sequences, train_labels)

    # 测试数据
    test_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_seq_positive.npy')
    test_seq_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_seq_negative.npy')
    test_label_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_label_positive.npy')
    test_label_negative_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/test_label_negative.npy')

    print(f"加载测试数据: {test_seq_positive_path}")

    # 加载序列和标签
    test_pos_sequences = np.load(test_seq_positive_path).tolist()
    test_neg_sequences = np.load(test_seq_negative_path).tolist()
    test_sequences = np.concatenate([test_pos_sequences, test_neg_sequences], axis=0)

    test_label_positive = np.load(test_label_positive_path)
    test_label_negative = np.load(test_label_negative_path)
    test_labels = np.concatenate([test_label_positive, test_label_negative], axis=0)

    # 创建测试数据集
    test_dataset = DNAGraphDataset(test_sequences, test_labels)

    print(f"训练数据集大小: {len(train_dataset)}")
    print(f"测试数据集大小: {len(test_dataset)}")

    return train_dataset, test_dataset


# 创建自定义PyG数据集
class DNAGraphDataset(Dataset):
    def __init__(self, sequences, labels=None, transform=None, pre_transform=None):
        """
        创建DNA图数据集

        参数:
            sequences: 序列列表
            labels: 标签列表 (可选)
            transform: 数据转换函数 (可选)
            pre_transform: 预处理转换函数 (可选)
        """
        super(DNAGraphDataset, self).__init__(None, transform, pre_transform)
        self.sequences = sequences
        self.labels = labels
        self.data_list = []

        # 预处理所有序列
        print("构建DNA图数据集...")
        for i, seq in enumerate(tqdm(sequences)):
            # 构建PyG图
            data = build_dna_pyg_graph(seq)

            # 添加标签（如果有）
            if labels is not None:
                data.y = torch.tensor([labels[i]], dtype=torch.float)

            self.data_list.append(data)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


# 提供一个简单的接口来测试功能
def test_visualization():
    """测试可视化功能"""
    # 确保全局PCA模型已加载
    global global_pca
    if global_pca is None:
        # 获取项目根目录的路径
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_positive.npy')
        train_pos_sequences = np.load(train_seq_positive_path).tolist()
        # 只使用少量序列拟合PCA，加快测试速度
        fit_global_pca(train_pos_sequences[:100], n_components=8)

    # 示例序列
    sequence = "ACGGATTC"
    print(f"序列: {sequence}")
    print(f"互补链: {generate_complementary_strand(sequence)}")

    # 构建PyG图
    pyg_data = build_dna_pyg_graph(sequence)

    # 打印图的信息
    print(f"节点数: {pyg_data.num_nodes}")
    print(f"边数: {pyg_data.num_edges}")
    print(f"节点特征维度: {pyg_data.x.shape}")
    print(f"边索引维度: {pyg_data.edge_index.shape}")
    print(f"边属性维度: {pyg_data.edge_attr.shape}")

    # 可视化并保存图像
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dna_pyg_graph_example.png")
    visualize_dna_graph(pyg_data, save_path=save_path)

    return pyg_data


# 可视化数据集中的前几个序列
def visualize_dataset_examples(num_examples=3):
    """可视化数据集中的前几个序列"""
    # 确保全局PCA模型已加载
    global global_pca
    if global_pca is None:
        # 获取项目根目录的路径
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_positive.npy')
        train_pos_sequences = np.load(train_seq_positive_path).tolist()
        # 只使用少量序列拟合PCA，加快测试速度
        fit_global_pca(train_pos_sequences[:100], n_components=8)

    # 获取项目根目录的路径
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 训练数据
    train_seq_positive_path = os.path.join(root_dir, 'data/Dataset_mouse/npy/train_seq_positive.npy')

    print(f"加载训练数据: {train_seq_positive_path}")

    # 加载序列
    try:
        train_pos_sequences = np.load(train_seq_positive_path).tolist()

        # 创建输出目录
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization_examples")
        os.makedirs(output_dir, exist_ok=True)

        # 可视化前几个序列
        for i in range(min(num_examples, len(train_pos_sequences))):
            sequence = train_pos_sequences[i]
            print(f"\n示例 {i + 1}:")
            print(f"序列: {sequence}")
            print(f"互补链: {generate_complementary_strand(sequence)}")

            # 构建PyG图
            pyg_data = build_dna_pyg_graph(sequence)

            # 打印图的信息
            print(f"节点数: {pyg_data.num_nodes}")
            print(f"边数: {pyg_data.num_edges}")
            print(f"节点特征维度: {pyg_data.x.shape}")
            print(f"边索引维度: {pyg_data.edge_index.shape}")
            print(f"边属性维度: {pyg_data.edge_attr.shape}")

            # 可视化并保存图像
            save_path = os.path.join(output_dir, f"dna_pyg_graph_example_{i + 1}.png")
            visualize_dna_graph(pyg_data, save_path=save_path, title=f"DNA双链图结构 - 示例 {i + 1}")

        print(f"\n可视化结果已保存至: {output_dir}")
        return True
    except Exception as e:
        print(f"可视化示例时出错: {e}")
        return False


def visualize_dna_graph(pyg_data, figsize=(12, 6), save_path=None, title=None):
    """可视化DNA图结构（从PyG数据转换为NetworkX进行可视化）"""
    # 将PyG图转换为NetworkX图用于可视化
    G = nx.Graph()

    sequence = pyg_data.sequence
    complementary = pyg_data.complementary

    # 添加主链节点
    for i, base in enumerate(sequence):
        G.add_node(f"main_{i}", base=base, strand="main", position=i)

    # 添加互补链节点
    for i, base in enumerate(complementary):
        G.add_node(f"comp_{i}", base=base, strand="complementary", position=i)

    # 添加主链内部连接
    for i in range(len(sequence) - 1):
        G.add_edge(f"main_{i}", f"main_{i + 1}", type="backbone")

    # 添加互补链内部连接
    for i in range(len(complementary) - 1):
        G.add_edge(f"comp_{i}", f"comp_{i + 1}", type="backbone")

    # 添加碱基对连接
    for i in range(len(sequence)):
        G.add_edge(f"main_{i}", f"comp_{i}", type="hydrogen_bond")

    plt.figure(figsize=figsize)

    # 设置节点位置
    pos = {}
    for node in G.nodes():
        if "main_" in node:
            i = int(node.split("_")[1])
            pos[node] = (i, 1)
        else:  # complementary strand
            i = int(node.split("_")[1])
            pos[node] = (i, 0)

    # 获取节点碱基类型
    node_colors = []
    for node in G.nodes():
        base = G.nodes[node]['base']
        if base == 'A':
            node_colors.append('green')
        elif base == 'T':
            node_colors.append('red')
        elif base == 'G':
            node_colors.append('blue')
        elif base == 'C':
            node_colors.append('yellow')
        else:
            node_colors.append('gray')

    # 绘制节点
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300)

    # 绘制边
    backbone_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'backbone']
    hydrogen_edges = [(u, v) for u, v, d in G.edges(data=True) if d['type'] == 'hydrogen_bond']

    nx.draw_networkx_edges(G, pos, edgelist=backbone_edges, width=2)
    nx.draw_networkx_edges(G, pos, edgelist=hydrogen_edges, width=1, style='dashed')

    # 添加节点标签
    labels = {node: G.nodes[node]['base'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels)

    if title:
        plt.title(title)
    else:
        plt.title("DNA双链图结构")
    plt.axis('off')

    # 始终保存图像，而不是显示
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dna_graph.png")

    plt.savefig(save_path)
    plt.close()

    print(f"图像已保存至: {save_path}")


# 如果直接运行此脚本，则执行测试
if __name__ == "__main__":
    # 可视化数据集中的前三个序列
    print("可视化数据集中的前三个序列...")
    visualize_dataset_examples(3)

    print("\n测试完成!")

# 导出用于GNN的图数据集
BDGraph_train_dataset, BDGraph_test_dataset = None, None


def get_graph_datasets():
    """获取图数据集，懒加载方式"""
    global BDGraph_train_dataset, BDGraph_test_dataset
    if BDGraph_train_dataset is None or BDGraph_test_dataset is None:
        BDGraph_train_dataset, BDGraph_test_dataset = load_and_process_for_gnn()
    return BDGraph_train_dataset, BDGraph_test_dataset