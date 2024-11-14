import numpy as np
import torch
import scipy.sparse as sp

from util.utils import get_normalized_adj


def normalize(mx):
    """行归一化稀疏矩阵"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_tensor(tensor):
    """归一化张量"""
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True)
    return (tensor - mean) / std


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将稀疏矩阵转换为torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def load_custom_data(path="./data", dataset='DBLP3.npz', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    print("正在加载{}数据集...".format(dataset))
    filedpath = path
    filename  = dataset
    file      = np.load(filedpath)

    features  = file['attmats'] #(n_node, n_time, att_dim)
    labels    = file['labels']  #(n_node, num_classes)
    graphs    = file['adjs']    #(n_time, n_node, n_node)

    # 归一化特征
    n_node, n_time, att_dim = features.shape
    tmp_g = torch.rand(n_time, n_node, n_node)
    graphs_outputs = torch.zeros_like(tmp_g)
    for t in range(n_time):
        features[:, t, :] = normalize_tensor(torch.FloatTensor(features[:, t, :])).numpy()
    n_node, n_steps, n_dim = np.shape(features)
    for i in range(n_steps):
        graphs[i,:,:] += np.eye(n_node, dtype=np.int32)
        graphs_outputs[i,:,:] = torch.from_numpy(get_normalized_adj(graphs[i, :, :]))

    # A_wave = torch.from_numpy(A_wave)
    # 归一化图结构
    # adj_list = []
    # for t in range(n_time):
    #     adj = sp.coo_matrix(graphs[t])
    #     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #     adj = normalize(adj + sp.eye(adj.shape[0]))
    #     adj_list.append(sparse_mx_to_torch_sparse_tensor(adj))

    # idx_train = range(int(n_node * 0.1))
    # idx_val = range(int(n_node * 0.1), int(n_node * 0.2))
    # idx_test = range(int(n_node * 0.2), n_node)
    idx_train = range(int(n_node * train_ratio))
    idx_val = range(int(n_node * train_ratio), int(n_node * (train_ratio + val_ratio)))
    idx_test = range(int(n_node * (train_ratio + val_ratio)), n_node)

    features = torch.FloatTensor(features)
    labels = torch.LongTensor(labels)
    # adj_list = [adj.to_dense() for adj in adj_list]# 转换为稠密张量以便处理
    # # graphs_outputs = torch.zeros_like(graphs)
    # # for i in range(len(adj_list)):
    # #     graphs_outputs[i, :, :] = adj_list[i]
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return graphs_outputs, features, labels, idx_train, idx_val, idx_test

# path = "./data/DBLP3.npz"
# dataset = "DBLP3"
# graphs_outputs, features, labels, idx_train, idx_val, idx_test = load_custom_data(path, dataset)
# labels2 = labels.argmax(dim=1)
# print(1)