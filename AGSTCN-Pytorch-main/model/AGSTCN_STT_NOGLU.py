import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from function.Attention import MultiHeadAttention_gcn
from util.utils import load_metr_la_data, get_normalized_adj
#添加了注意力引导层，且结构为空间-时间-时间的stgcn

class TimeBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size=3):

        super(TimeBlock, self).__init__()

        self.conv3 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, X):

        # Convert into NCHW format for pytorch to perform convolutions.
        X = X.permute(0, 3, 1, 2)
        # temp = self.conv1(X) + torch.sigmoid(self.conv2(X))
        # out = F.relu(self.conv3(X))
        out = F.relu(self.conv3(X))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out
class GCN(nn.Module):
    def __init__(self, d_model,out_model,device):
        """
        d_model是一个时间片上有几个特征
        """
        super(GCN, self).__init__()
        # self.outputs = torch.zeros(batch_size, num_nodes, time_steps, d_model).to(device=device)
        self.Theta1 = nn.Parameter(torch.FloatTensor(d_model,
                                                     out_model)).to(device=device)  # nn.Parameter作用是将一个不可训练的tensor转换为可训练的tensor
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.Theta1.shape[1])
        self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X, A_hat):


        outputs = torch.zeros_like(X).to(X.device)
        # 遍历时间步
        if A_hat.dim() == 2:
            A_hat_expanded = A_hat.unsqueeze(0).repeat(X.shape[0], 1, 1)
        else:
            A_hat_expanded = A_hat
        if X.dim() == 4:
            for t in range(X.shape[2]):
                # 取出当前时间步的节点特征
                X_t = X[:, :, t, :]  # 形状是 (batch_size, num_nodes, num_features)
                # 对当前时间步的节点特征应用图卷积
                outputs[:, :, t, :] = torch.matmul(A_hat_expanded, X_t)
            t2 = F.relu(torch.matmul(outputs, self.Theta1))
            return t2
        else:
            out = torch.matmul(A_hat_expanded, X)
            out_put = F.relu(torch.matmul(out, self.Theta1))
            return out_put
            # t2 = F.relu(torch.einsum("ijkl,lp->ijkp", [out, self.Theta1]))

class AGSTCNBlock(nn.Module):


    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes,dev):

        super(AGSTCNBlock, self).__init__()

        self.gcn = GCN(in_channels, spatial_channels, device=dev)
        self.temporal1 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        self.temporal2 = TimeBlock(in_channels=out_channels,
                                   out_channels=spatial_channels)

        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X, A_hat):

        # t = self.temporal1(X)
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        t2 =  self.gcn(X,A_hat)
        # t3 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t3 = F.relu(torch.matmul(t2, self.Theta1))
        # t33 = F.relu(torch.matmul(lfs, self.Theta1))
        t3 = self.temporal1(t2)
        t4 = self.temporal2(t3)
        return self.batch_norm(t4)
        # return t3

class AGSTCNBlock2(nn.Module):


    def __init__(self,  spatial_channels, out_channels,
                 num_nodes,dev):

        super(AGSTCNBlock2, self).__init__()

        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
        #                                              spatial_channels))
        self.temporal1 = TimeBlock(in_channels=spatial_channels*4,
                                   out_channels=out_channels)
        self.temporal2 = TimeBlock(in_channels=out_channels,
                                   out_channels=spatial_channels)
        self.ag = GCN_at( d_model=spatial_channels, spatial_channels=spatial_channels,device=dev,ag_head=4)
        self.agheads = spatial_channels * 4
        self.batch_norm = nn.BatchNorm2d(num_nodes)
    #     self.reset_parameters()
    #
    # def reset_parameters(self):
    #     stdv = 1. / math.sqrt(self.Theta1.shape[1])
    #     self.Theta1.data.uniform_(-stdv, stdv)

    def forward(self, X):


        # t = self.temporal1(X)   #out 64
        tmp = torch.rand(X.shape[0], X.shape[1], X.shape[2], self.agheads)
        outputs = torch.zeros_like(tmp).to(X.device)
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        for index in range(X.shape[2]):
            # 取出当前时间步的节点特征
            X_t = X[:, :, index, :]
            outputs[:, :, index, :] = self.ag(X_t)

        # t3 = F.relu(torch.einsum("ijkl,lp->ijkp", [lfs, self.Theta1]))
        # t3 = F.relu(torch.matmul(t2, self.Theta1))
        # t33 = F.relu(torch.matmul(lfs, self.Theta1))
        t4 = self.temporal1(outputs)
        t5 = self.temporal2(t4)
        return self.batch_norm(t5)

class GCN_at(nn.Module):
    def __init__(self, d_model,spatial_channels,device,ag_head):
        """
        d_model是一个时间片上有几个特征
        """
        super(GCN_at, self).__init__()
        self.heads = ag_head
        self.layers = []
        self.ag = MultiHeadAttention_gcn(ag_head, d_model).to(device=device)
        for i in range(ag_head):
            self.layers.append(GCN(d_model=d_model,out_model=spatial_channels, device=device).to(device=device))


    def forward(self, X):
        """
        X: 输入数据 形状为 (batch_size, num_nodes, num_timesteps, num_features(一个时间步上的特征数量)).
        A_hat: 邻接矩阵
        return: 输出数据 形状为 (batch_size, num_nodes, num_timesteps, num_features(一个时间步上的特征数量)).
        """
        gcn_out_list = []
        attn_tensor = self.ag(X, X)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        for head in range(self.heads):
            adj = attn_adj_list[head]
            gcn_layer = self.layers[head]
            gcn_out = gcn_layer(X, adj)
            gcn_out_list.append(gcn_out+X)
        final_output = torch.cat(gcn_out_list, dim=2)

        return (final_output)
class AGSTCN(nn.Module):


    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output,dev):

        super(AGSTCN, self).__init__()
        self.block1 = AGSTCNBlock(in_channels=num_features, out_channels=16,
                                  spatial_channels=64, num_nodes=num_nodes, dev=dev)
        self.block2 = AGSTCNBlock2(out_channels=16,
                                   spatial_channels=64, num_nodes=num_nodes, dev=dev)
        # self.last_temporal = TimeBlock(in_channels=64, out_channels=64)
        self.fully = nn.Linear((num_timesteps_input - 4 * 2) * 64,
                               num_timesteps_output)

    def forward(self,  X,A_hat):

        out1 = self.block1(X, A_hat)
        out2 = self.block2(out1)
        # out3 = self.last_temporal(out2)
        out4 = self.fully(out2.reshape((out2.shape[0], out2.shape[1], -1)))
        return out4

if __name__ == '__main__':
    A, X, means, stds = load_metr_la_data()
    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)
    test = torch.rand(32, 207, 12, 2)

    net = AGSTCN(A_wave.shape[0],
                 test.shape[3],
                 12,
                 3, dev='cpu').to(device=torch.device('cpu'))

    out = net(test,A_wave)
    #print(out.shape)
    # print("value: {}".format(tensor[: ,1].detach().cpu().numpy()))
    # print("reference value: {}".format(out[: ,1].detach().cpu().numpy()*167.75+694))
