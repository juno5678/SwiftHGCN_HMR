import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv

class AdaptiveAdjacencyMatrix(nn.Module):
    def __init__(self, num_nodes):
        super(AdaptiveAdjacencyMatrix, self).__init__()
        # 使用可學習的參數來調整初始鄰接矩陣
        self.adaptive_params = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self, initial_adj):
        # 計算最終的自適應鄰接矩陣
        adaptive_adj = initial_adj + self.adaptive_params
        # 確保鄰接矩陣是對稱的
        adaptive_adj = (adaptive_adj + adaptive_adj.t()) / 2
        # 確保鄰接矩陣的元素範圍在 [0, 1] 之間
        adaptive_adj = torch.sigmoid(adaptive_adj)
        return adaptive_adj

class GCN(nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.adaptive_A = AdaptiveAdjacencyMatrix(num_nodes)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        initial_adj = torch.eye(data.num_nodes, device=data.x.device)  # 使用單位矩陣作為初始鄰接矩陣
        A = self.adaptive_A(initial_adj)
        edge_index = A.nonzero(as_tuple=False).t()
        edge_weight = A[edge_index[0], edge_index[1]]

        x = F.relu(self.conv1(data.x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        return x

# 創建圖數據
def create_graph_data(num_nodes, num_features):
    x = torch.randn(num_nodes, num_features)  # 節點特徵
    edge_index = torch.randint(0, num_nodes, (2, 2 * num_nodes))  # 隨機生成邊
    return Data(x=x, edge_index=edge_index)

num_graphs = 10
num_nodes = 5
num_features = 3
batch_size = 2

# 創建數據集
dataset = [create_graph_data(num_nodes, num_features) for _ in range(num_graphs)]
dataloader = DataLoader(dataset, batch_size=batch_size)

# 創建模型
in_channels = num_features
hidden_channels = 4
out_channels = 2
model = GCN(num_nodes, in_channels, hidden_channels, out_channels)

# 訓練過程
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    for data in dataloader:
        model.train()
        optimizer.zero_grad()
        output = model(data)
        target = torch.randn_like(output)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
