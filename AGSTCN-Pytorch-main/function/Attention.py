import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F




def attention(query, key, mask=None, dropout=None):
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)

        return p_attn

def clones(module, N):
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class GCN(nn.Module):
    def __init__(self, spatial_channels, d_model):
        """
        spatial_channels与d_model其实是一个，就是一个时间片上有几个特征
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(GCN, self).__init__()
        self.Theta1 = nn.Parameter(torch.FloatTensor(d_model,
                                                     spatial_channels))  # nn.Parameter作用是将一个不可训练的tensor转换为可训练的tensor
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        lfs = torch.einsum("ij,jklm->kilm", [A_hat, X.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        t2 = F.relu(torch.matmul(lfs, self.Theta1))
        return t2
class MultiHeadAttention_gcn(nn.Module):

        def __init__(self, h,d, dropout=0.1):
            super(MultiHeadAttention_gcn, self).__init__()
            assert d % h == 0
          #d  = 时间片数量*一个时间片上的特征数量
            self.d_k = d // h
            self.h = h
            self.linears = clones(nn.Linear(d, d), 2)
            self.dropout = nn.Dropout(p=dropout)

        def forward(self, query, key, mask=None):
            if mask is not None:
                mask = mask.unsqueeze(1)

            nbatches = query.size(0)

            query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                          for l, x in zip(self.linears, (query, key))]
            attn = attention(query, key, mask=mask, dropout=self.dropout)

            return attn


if __name__ == '__main__':



    #
    embed_dim = 4#每个时间片上的特征数
    # num_heads = 3#注意力头的数量
    query = torch.randn(32,207, 2)  # 假设有207个序列，每个序列有12个时间片，每个时间片上四个特征
    test = torch.rand(1, 207, 12, 4)
    # key = value = query  # q k v相同
    #
    # print(output.shape)
    final_out = query.unsqueeze(0)#把输出的张量后两维展平，并在第0维拼接一维用来充当batchsize（这个batch就定死了是1应该就行
    #
    listlear = []
    attn = MultiHeadAttention_gcn(2, 2)
    listlear.append(attn)#注意力引导所生成的邻接矩阵所用的方法，参数分别为，注意力头的数量；时间片数量*每个时间片上特征数；节点数量
    j = listlear[0]
    attn_tensor = j(final_out, final_out)
    attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
    gcn = GCN(4,4)
    out = gcn(test,attn_adj_list[0].squeeze(0))
    print("i")





