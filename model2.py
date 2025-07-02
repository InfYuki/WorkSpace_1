import torch.nn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_geometric.nn import (
    GCNConv, GATConv, TransformerConv, GINConv, SAGEConv,
    GraphConv, global_mean_pool, global_max_pool, global_add_pool,
    JumpingKnowledge
)
import torch
#torch.backends.cudnn.enabled = False

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=41):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.w_omega = nn.Parameter(torch.Tensor(hidden_size * 2, hidden_size * 2))
        self.u_omega = nn.Parameter(torch.Tensor(hidden_size * 2, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def forward(self, lstm_output):
        u = torch.tanh(torch.matmul(lstm_output, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = lstm_output * att_score
        context = torch.sum(scored_x, dim=1)
        return context

''''''
# 添加GCN模块
class GCNModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCNModule, self).__init__()
        # 修改输入通道数为34，与BDGraph.py生成的节点特征维度匹配
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index, batch):
        # 第一层GCN
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # 第二层GCN
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)

        # 第三层GCN
        x = self.conv3(x, edge_index)

        # 全局池化，得到图级表示
        x = global_mean_pool(x, batch)

        return x


# 1. GraphSAGE模型 - 适合异构节点特征
class GraphSAGEModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGEModule, self).__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index)

        x = global_mean_pool(x, batch)
        return x


# 2. 图变换器模型 - 能捕获长距离依赖
class GraphTransformerModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphTransformerModule, self).__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.transformer1 = TransformerConv(hidden_channels, hidden_channels, heads=4, dropout=0.2)
        self.transformer2 = TransformerConv(hidden_channels * 4, hidden_channels, heads=1, dropout=0.2)
        self.lin2 = nn.Linear(hidden_channels, out_channels)

        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels * 4)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.ln1(x)

        x = self.transformer1(x, edge_index)
        x = F.relu(x)
        x = self.ln2(x)
        x = self.dropout(x)

        x = self.transformer2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.lin2(x)

        x = global_mean_pool(x, batch)
        return x


# 3. 图同构网络 - 理论上具有最强的表达能力
class GINModule(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GINModule, self).__init__()

        nn1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv1 = GINConv(nn1)

        nn2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.conv2 = GINConv(nn2)

        nn3 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.conv3 = GINConv(nn3)

        self.jump = JumpingKnowledge(mode='cat')
        self.lin = nn.Linear(hidden_channels + hidden_channels + out_channels, out_channels)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, batch):
        x1 = self.conv1(x, edge_index)
        x1 = F.relu(x1)
        x1 = self.dropout(x1)

        x2 = self.conv2(x1, edge_index)
        x2 = F.relu(x2)
        x2 = self.dropout(x2)

        x3 = self.conv3(x2, edge_index)

        # 使用跳跃连接组合不同层的特征
        x = self.jump([x1, x2, x3])

        # 全局池化
        x = global_mean_pool(x, batch)

        # 最终投影
        x = self.lin(x)

        return x

'''
###########################################
Module3
MLP  ResNet  DenseNet
'''

# 添加MLP模块用于处理x3
class MLPModule(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(MLPModule, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.size()

        # 使用reshape而不是view，避免内存布局问题
        x = x.reshape(-1, input_dim)  # [batch_size*seq_len, input_dim]

        # MLP处理
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)

        # 恢复原始形状，同样使用reshape
        x = x.reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, output_dim]

        return x


# 添加ResNet模块用于处理x3
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=kernel_size // 2, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 如果输入和输出通道数不同，需要使用1x1卷积进行调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ResNetModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=2):
        super(ResNetModule, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()

        # ResNet块
        self.layer1 = self._make_layer(32, 32, num_blocks, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks, stride=1)

        # 输出层
        self.conv_out = nn.Conv1d(64, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn_out = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(0.3)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, in_channels]
        # 转换为卷积所需的形状: [batch_size, in_channels, seq_len]
        x = x.permute(0, 2, 1)

        # 通过网络
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.relu(self.bn_out(self.conv_out(x)))

        # 转换回原始形状: [batch_size, seq_len, out_channels]
        x = x.permute(0, 2, 1)

        return x


# 添加DenseNet模块用于处理x3
class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size, dropout_rate=0.2):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv1 = nn.Conv1d(in_channels, bn_size * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(bn_size * growth_rate)
        self.conv2 = nn.Conv1d(bn_size * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.dropout(out)

        # 连接输入和输出特征图
        out = torch.cat([x, out], 1)

        return out


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, bn_size, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = DenseLayer(in_channels + i * growth_rate, growth_rate, bn_size, dropout_rate)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        return out


class DenseNetModule(nn.Module):
    def __init__(self, in_channels, out_channels, growth_rate=12, block_config=(6, 12),
                 bn_size=4, dropout_rate=0.2):
        super(DenseNetModule, self).__init__()

        # 初始卷积层
        self.conv1 = nn.Conv1d(in_channels, 2 * growth_rate, kernel_size=3,
                               padding=1, bias=False)

        # DenseBlocks
        num_features = 2 * growth_rate
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, growth_rate, bn_size, dropout_rate)
            self.blocks.append(block)
            num_features += num_layers * growth_rate

            # 添加过渡层，除了最后一个块
            if i != len(block_config) - 1:
                transition = Transition(num_features, num_features // 2)
                self.transitions.append(transition)
                num_features = num_features // 2

        # 最终BatchNorm
        self.bn_final = nn.BatchNorm1d(num_features)
        self.relu_final = nn.ReLU(inplace=True)

        # 输出层
        self.conv_out = nn.Conv1d(num_features, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 输入形状: [batch_size, seq_len, in_channels]
        # 转换为卷积所需的形状: [batch_size, in_channels, seq_len]
        x = x.permute(0, 2, 1)

        # 初始卷积
        x = self.conv1(x)

        # DenseBlocks和过渡层
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        # 最终处理
        x = self.bn_final(x)
        x = self.relu_final(x)
        x = self.conv_out(x)

        # 转换回原始形状: [batch_size, seq_len, out_channels]
        x = x.permute(0, 2, 1)

        return x

class model(nn.Module):
    def __init__(self,out_channels,kernel_size,stride,hidden_size):
        super(model, self).__init__()

        # 替换LSTM为Transformer
        self.input_projection = nn.Linear(8, 24)
        self.positional_encoding = PositionalEncoding(d_model=24)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=24,
            nhead=8,
            dim_feedforward=96,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer,
            num_layers=4
        )
        self.output_projection = nn.Linear(24, 24)  # 保持与原始输出相同维度

        #self.lstm_x1 = nn.LSTM(input_size=4, hidden_size=12,num_layers=1, batch_first=True, bidirectional=True)
        #self.lstm_x1_2 = nn.LSTM(input_size=24, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True)

        # 替换x2处理模块为GCN
        # 输入特征维度为34（BERT嵌入(32维) + 位置信息(1维) + 链类型(1维)，总共34维） 换成32
        # 输出维度为24，与原始模块相同
        self.gnn = GINModule(in_channels=8, hidden_channels=32, out_channels=24)

        # 添加一个投影层，将GCN输出转换为与原始x2相同的形状
        self.gnn_projection = nn.Linear(24, 41 * 24)  # 假设原始x2是[batch_size, 41, 24]

        self.embedding2 = nn.Embedding(num_embeddings=16, embedding_dim=24)
        self.convx3_1 = nn.Conv1d(in_channels=24, out_channels=12, kernel_size=3, stride=stride, bias=False)
        self.convx3_2 = nn.Conv1d(in_channels=12, out_channels=24, kernel_size=3, stride=stride, bias=False)
        self.lstm_x3_1 = nn.LSTM(input_size=24, hidden_size=12,num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_x3_2 = nn.LSTM(input_size=24, hidden_size=12, num_layers=1, batch_first=True, bidirectional=True)
        self.attention=Attention(hidden_size)


        # 修改全连接层的输入维度
        # 计算连接后的特征总维度
        # x1: [batch_size, 41, 24]
        # x2: [batch_size, 41, 24]
        # x3: [batch_size, 41, 24]
        # 连接后: [batch_size, 123, 24] -> 展平为 [batch_size, 123*24] = [batch_size, 2952]
        # [batch_size, 164, 24] -> 展平为 [batch_size, 164*24] = [batch_size, 3936]
        #input_dim = 41 * 24 * 3  # 2952
        input_dim = 3936
        self.fc1 = nn.Linear(input_dim, 160)
        self.fc2 = nn.Linear(160, 1)
        self.dropout = nn.Dropout(0.5)
        self.ln = nn.LayerNorm(24)
        self.bn5 = nn.BatchNorm1d(24)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

        '''
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.bn3 = nn.BatchNorm1d(out_channels+3)
        self.bn4 = nn.BatchNorm1d(12)
        self.conv1x1 = nn.Conv1d(in_channels=out_channels, out_channels=out_channels + 3, kernel_size=1, stride=stride)
        self.lstm2 = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=1, batch_first=True,
                             bidirectional=True)
        self.lstm = nn.LSTM(input_size=out_channels + 3, hidden_size=hidden_size, num_layers=1, batch_first=True,
                            bidirectional=True)  # input_size=912
        self.embedding = nn.Embedding(num_embeddings=8, embedding_dim=out_channels)
        self.conv_op = nn.Conv1d(in_channels=out_channels, out_channels=out_channels + 3, kernel_size=kernel_size,
                                 padding=1, stride=stride, bias=False)
        self.conv_op2 = nn.Conv1d(in_channels=out_channels + 3, out_channels=out_channels + 3, kernel_size=kernel_size,
                                  padding=1, stride=stride, bias=False)
        '''
        #self.embedding = nn.Embedding(num_embeddings=8, embedding_dim=out_channels)
        #self.mlp_x3 = MLPModule(input_dim=out_channels, hidden_dim=48, output_dim=24, dropout_rate=0.4)
        #self.bn1 = nn.BatchNorm1d(out_channels)
        #self.ln_x3 = nn.LayerNorm(24)

        # 用于x3的处理 - 使用ResNet替换MLP
        #self.embedding = nn.Embedding(num_embeddings=8, embedding_dim=out_channels)
        #self.resnet_x3 = ResNetModule(in_channels=out_channels, out_channels=24, num_blocks=3)
        #self.bn1 = nn.BatchNorm1d(out_channels)
        #self.ln_x3 = nn.LayerNorm(24)
        # 用于x3的处理 - 使用DenseNet替换ResNet
        self.embedding = nn.Embedding(num_embeddings=8, embedding_dim=out_channels)
        self.densenet_x3 = DenseNetModule(in_channels=out_channels, out_channels=24,
                                        growth_rate=12, block_config=(4, 8),
                                        bn_size=4, dropout_rate=0.2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.ln_x3 = nn.LayerNorm(24)


    def forward(self, x1,x2_data,x3):

        # 处理x1: [batch_size, 41, 4]
        # 将特征维度映射到transformer所需的维度
        x1 = self.input_projection(x1)  # [batch_size, 41, 24]
        # 添加位置编码
        x1 = self.positional_encoding(x1)
        # 通过transformer
        x1 = self.transformer_encoder(x1)
        # 最终投影
        x1 = self.output_projection(x1)  # [batch_size, 41, 24]
        x1 = self.dropout(x1)

        #x1, _ = self.lstm_x1(x1)
        #x1= self.dropout(x1)
        #x1, _ =self.lstm_x1_2(x1)

        # 处理x2: 使用GCN
        x2 = self.gnn(x2_data.x, x2_data.edge_index, x2_data.batch)  # [batch_size, 24]

        # 将GCN输出投影到原始x2的形状
        batch_size = x1.size(0)
        x2 = self.gnn_projection(x2)  # [batch_size, 41*24]
        x2 = x2.view(batch_size, 41, 24)  # [batch_size, 41, 24]

        '''
        x3 = self.embedding(x3.long())
        x3 = self.bn1(x3.permute(0,2,1))
        original = self.conv1x1(x3)

        x3 = self.relu(self.conv_op(x3))
        x3 = self.dropout(x3)
        x3 = self.dropout(self.bn3(original+x3))

        x3 = self.relu(self.conv_op2(x3))
        x3 = self.dropout(x3)
        x3, _ = self.lstm(x3.permute(0,2,1))
        x3 = self.bn2(x3.permute(0, 2, 1))
        x3, _ = self.lstm2(x3.permute(0, 2, 1))
        '''
        # 处理x3: 使用MLP
        #x3 = self.embedding(x3.long())
        #x3 = self.bn1(x3.permute(0, 2, 1))  # [batch_size, out_channels, 41]
        #x3 = x3.permute(0, 2, 1)  # [batch_size, 41, out_channels]

        # 通过MLP处理
        #x3 = self.mlp_x3(x3)  # [batch_size, 41, 24]
        #x3 = self.ln_x3(x3)
        #x3 = self.dropout(x3)

        # 处理x3: 使用ResNet
        #x3 = self.embedding(x3.long())  # [batch_size, 41, out_channels]
        #x3 = self.resnet_x3(x3)  # [batch_size, 41, 24]
        #x3 = self.ln_x3(x3)
        #x3 = self.dropout(x3)
        # 处理x3: 使用DenseNet
        x3 = self.embedding(x3.long())  # [batch_size, 41, out_channels]
        x3 = self.densenet_x3(x3)  # [batch_size, 41, 24]
        x3 = self.ln_x3(x3)
        x3 = self.dropout(x3)


        # 打印形状以便调试
        #print(f"x1 shape: {x1.shape}")
        #print(f"x2 shape: {x2.shape}")
        #print(f"x3 shape: {x3.shape}")


        #x = torch.cat([x1, x2, x3.permute(0,2,1)], axis=1)  # 传统特征和预训练特征 结合:([64, 30, 106]),([64, 1, 54])
        #x = x.reshape(x.size(0), -1)

        # 合并特征
        x = torch.cat([x1, x2, x3], axis=1)
        #print(f"Concatenated shape: {x.shape}")
        x = x.reshape(x.size(0), -1)
        #print(f"Flattened shape: {x.shape}")

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return self.sigmoid(x)
