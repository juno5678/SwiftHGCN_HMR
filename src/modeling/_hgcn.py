import torch
from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
from torch_geometric.data.batch import Batch
from src.modeling.data.Hypergraph_batch import HG_Batch
from src.modeling._gcnn import GraphLinear
from src.modeling.__hgcn_gcn_conv import HGConv_edge_to_node, HGConv_node_to_edge
from torch_geometric.nn import GCNConv
from torch_geometric.nn.dense.linear import Linear

class HGGCN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.hgc_n2e = HGConv_node_to_edge(in_channels, hidden_channels)
        self.hgc_e2n = HGConv_edge_to_node(hidden_channels, out_channels)
        self.lin1 = GraphLinear(hidden_channels, hidden_channels // 4)
        self.gcn = GCNConv(hidden_channels // 4, hidden_channels // 4)
        self.lin2 = GraphLinear(hidden_channels // 4, hidden_channels)

        self.norm1 = torch.nn.LayerNorm(hidden_channels // 4)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        # self.norm3 = torch.nn.LayerNorm(hidden_channels)
        # self.norm3 = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x, hyperedge_index, graph_index, hyperedge_attr=None):
        B, N, C = x.shape
        if hyperedge_attr == None:
            data = [Data(x=x[i], edge_index=hyperedge_index) for i in range(x.shape[0])]
        else:
            data = [Data(x=x[i], edge_index=hyperedge_index, edge_attr=hyperedge_attr[i]) for i in range(x.shape[0])]

        num_edges = int(hyperedge_index[1].max()) + 1
        # 將 hypergraph 重複堆疊 batch 次
        hg_batch_data = HG_Batch.from_data_list(data)
        node_feature = hg_batch_data.x
        output, total_num_edges, total_num_nodes, hyperedge_weight = self.hgc_n2e(hg_batch_data.x,
                                                                                  hg_batch_data.edge_index,
                                                                                  hyperedge_attr=hg_batch_data.edge_attr)
        res_x = output
        output = output.view(B, num_edges, -1).transpose(1, 2)
        output = self.lin1(output).transpose(1, 2)
        output = F.relu(self.norm1(output))
        g_data = [Data(x=output[i], edge_index=graph_index) for i in range(x.shape[0])]
        g_batch_data = Batch.from_data_list(g_data)
        output = self.gcn(g_batch_data.x, g_batch_data.edge_index)
        output = output.view(B, num_edges, -1).transpose(1, 2)
        output = self.lin2(output).transpose(1, 2)
        output = F.relu(self.norm2(output))
        output = output.contiguous().view(B * num_edges, -1) + res_x
        # output = output.contiguous().view(B * num_edges, -1)

        output = self.hgc_e2n(output, hg_batch_data.edge_index, hyperedge_attr=hg_batch_data.edge_attr,
                              num_edges=total_num_edges, hyperedge_weight=hyperedge_weight, num_nodes=total_num_nodes)
        # output += node_feature
        # output = self.norm3(output)
        # output = F.relu(self.norm3(output))

        output = output.view(B, N, -1)

        return output


class HGGCN_JV(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.hgc_n2e = HGConv_node_to_edge(in_channels, hidden_channels)
        self.hgc_e2n = HGConv_edge_to_node(hidden_channels, out_channels)
        self.lin1 = GraphLinear(hidden_channels, hidden_channels // 4)
        self.gcn = GCNConv(hidden_channels // 4, hidden_channels // 4)
        self.lin2 = GraphLinear(hidden_channels // 4, hidden_channels)

        self.norm1 = torch.nn.LayerNorm(hidden_channels//4)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)
        # self.lin1 = GraphLinear(in_channels, hidden_channels // 4)
        # self.gcn = GCNConv(hidden_channels // 4, hidden_channels // 4)
        # self.lin2 = GraphLinear(hidden_channels // 4, hidden_channels)

    def forward(self, x, joint_x,  hyperedge_index, graph_index, hyperedge_attr=None):
        B, N, C = x.shape
        if hyperedge_attr == None:
            data = [Data(x=x[i], edge_index=hyperedge_index) for i in range(x.shape[0])]
        else:
            data = [Data(x=x[i], edge_index=hyperedge_index, edge_attr=hyperedge_attr[i]) for i in range(x.shape[0])]

        num_edges = int(hyperedge_index[1].max()) + 1
        # 將 hypergraph 重複堆疊 batch 次
        hg_batch_data = HG_Batch.from_data_list(data)
        output, total_num_edges, total_num_nodes, hyperedge_weight = self.hgc_n2e(hg_batch_data.x,
                                                                                  hg_batch_data.edge_index,
                                                                                  hyperedge_attr=hg_batch_data.edge_attr)
        # res_x = output
        output = output.view(B, num_edges, -1).transpose(1, 2)
        output = self.lin1(output).transpose(1, 2)
        output = F.relu(self.norm1(output))
        g_data = [Data(x=output[i], edge_index=graph_index) for i in range(x.shape[0])]
        g_batch_data = Batch.from_data_list(g_data)
        output = self.gcn(g_batch_data.x, g_batch_data.edge_index)
        output = output.view(B, num_edges, -1).transpose(1, 2)
        output = self.lin2(output).transpose(1, 2)
        output = F.relu(self.norm2(output))
        # output = output.contiguous().view(B * num_edges, -1) + res_x
        output = output.contiguous().view(B * num_edges, -1)

        output = self.hgc_e2n(output, hg_batch_data.edge_index, hyperedge_attr=hg_batch_data.edge_attr,
                              num_edges=total_num_edges, hyperedge_weight=hyperedge_weight, num_nodes=total_num_nodes)

        output = output.view(B, N, -1)

        return output

class HGCN(torch.nn.Module):

    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(HypergraphConv(in_channels, hidden_channels))
        # for _ in range(num_layers - 2):
            # self.convs.append(HypergraphConv(hidden_channels, hidden_channels, use_attention=True, concat=False, heads=2, dropout=0.5))
            # self.convs.append(HypergraphConv(hidden_channels, hidden_channels))
        self.convs.append(HypergraphConv(hidden_channels, out_channels))

    def forward(self, x, hyperedge_index, hyperedge_attr=None):
        B, N, C =x.shape

        if hyperedge_attr == None:
            data = [Data(x=x[i], edge_index=hyperedge_index) for i in range(x.shape[0])]
        else:
            data = [Data(x=x[i], edge_index=hyperedge_index, edge_attr=hyperedge_attr[i]) for i in range(x.shape[0])]

        batch_data = HG_Batch.from_data_list(data)
        output = batch_data.x
        for conv in self.convs:
            output = conv(output, batch_data.edge_index, hyperedge_attr=batch_data.edge_attr)
        output = output.view(B, N, -1)

        return output

class HGGCNBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super(HGGCNBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, hidden_channels)
        self.conv1 = HGGCN(hidden_channels, hidden_channels, hidden_channels)
        self.conv2 = HGGCN(hidden_channels, hidden_channels, hidden_channels)
        # self.conv1 = HGGCN_JV(hidden_channels, hidden_channels, hidden_channels)
        # self.conv2 = HGGCN_JV(hidden_channels, hidden_channels, hidden_channels)

        # self.conv = HGCN(out_channels // 2, out_channels // 2, mesh_type)
        self.lin2 = GraphLinear(hidden_channels, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = torch.nn.LayerNorm(in_channels)
        self.norm1 = torch.nn.LayerNorm(hidden_channels)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x, hyperedge_index, graph_index, hyperedge_attr=None):
        trans_y = F.relu(self.pre_norm(x)).transpose(1,2)
        y = self.lin1(trans_y).transpose(1,2)
        y = F.relu(self.norm1(y))
        y = self.conv1(y, hyperedge_index, graph_index)
        y = self.conv2(y, hyperedge_index, graph_index)
        trans_y = F.relu(self.norm2(y)).transpose(1,2)
        y = self.lin2(trans_y).transpose(1,2)



        return y


class HyperGraphResBlock(torch.nn.Module):
    """
    Graph Residual Block similar to the Bottleneck Residual Block in ResNet
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
        super(HyperGraphResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin1 = GraphLinear(in_channels, hidden_channels//2)
        self.conv = HGCN(in_channels=hidden_channels//2, out_channels=hidden_channels, hidden_channels=hidden_channels//4)
        # self.conv = HGCN(out_channels // 2, out_channels // 2, mesh_type)
        self.lin2 = GraphLinear(hidden_channels, out_channels)
        self.skip_conv = GraphLinear(in_channels, out_channels)
        # print('Use BertLayerNorm in GraphResBlock')
        self.pre_norm = torch.nn.LayerNorm(in_channels)
        self.norm1 = torch.nn.LayerNorm(hidden_channels//2)
        self.norm2 = torch.nn.LayerNorm(hidden_channels)

    def forward(self, x, incident_matrix):
        trans_y = F.relu(self.pre_norm(x)).transpose(1,2)
        y = self.lin1(trans_y).transpose(1,2)

        y = F.relu(self.norm1(y))
        y = self.conv(y, incident_matrix)

        trans_y = F.relu(self.norm2(y)).transpose(1,2)
        y = self.lin2(trans_y).transpose(1,2)

        z = x+y

        return z

# class HGGCNResBlock(torch.nn.Module):
#     """
#     Graph Residual Block similar to the Bottleneck Residual Block in ResNet
#     """
#     def __init__(self, in_channels, out_channels, hidden_channels=64, num_layers=2):
#         super(HGGCNResBlock, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.lin1 = GraphLinear(in_channels, hidden_channels)
#         # self.conv = HGGConv(in_channels=hidden_channels, out_channels=hidden_channels)
#         # self.conv = HGCN(out_channels // 2, out_channels // 2, mesh_type)
#         self.lin2 = GraphLinear(hidden_channels, out_channels)
#         self.skip_conv = GraphLinear(in_channels, out_channels)
#         # print('Use BertLayerNorm in GraphResBlock')
#         self.pre_norm = torch.nn.LayerNorm(in_channels)
#         self.norm1 = torch.nn.LayerNorm(hidden_channels)
#         self.norm2 = torch.nn.LayerNorm(hidden_channels)
#
#     def forward(self, x, incident_matrix, adj_matrix):
#         trans_y = F.relu(self.pre_norm(x)).transpose(1,2)
#         y = self.lin1(trans_y).transpose(1,2)
#
#         y = F.relu(self.norm1(y))
#         y = self.conv(y, incident_matrix, adj_matrix)
#
#         trans_y = F.relu(self.norm2(y)).transpose(1,2)
#         y = self.lin2(trans_y).transpose(1,2)
#
#         # z = x+y
#
#         return y
