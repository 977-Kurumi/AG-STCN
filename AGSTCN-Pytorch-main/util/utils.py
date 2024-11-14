import os
import zipfile
import numpy as np
import torch
from sklearn import preprocessing
import random

def load_metr_la_data():
    if (not os.path.isfile("./data/adj_mat.npy")
            or not os.path.isfile("../data/node_values.npy")):
        with zipfile.ZipFile("./data/METR-LA.zip", 'r') as zip_ref:
            zip_ref.extractall("data/")

    A = np.load("./data/adj_mat.npy")
    X = np.load("./data/node_values.npy").transpose((1, 2, 0))
    X = X.astype(np.float32)

    # Normalization using Z-score method
    means = np.mean(X, axis=(0, 2))
    X = X - means.reshape(1, -1, 1)
    stds = np.std(X, axis=(0, 2))
    X = X / stds.reshape(1, -1, 1)

    return A, X, means, stds


def get_normalized_adj(A):
    """
    Returns the degree normalized adjacency matrix.
    """
    A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
    D = np.array(np.sum(A, axis=1)).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = np.reciprocal(np.sqrt(D))
    A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    return A_wave


def generate_dataset(X, num_timesteps_input, num_timesteps_output):
    """12 3
    Takes node features for the graph and divides them into multiple samples
    along the time-axis by sliding a window of size (num_timesteps_input+
    num_timesteps_output) across it in steps of 1.
    :param X: Node features of shape (num_vertices, num_features,
    num_timesteps)
    :return:
        - Node features divided into multiple samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_input).
        - Node targets for the samples. Shape is
          (num_samples, num_vertices, num_features, num_timesteps_output).
    """
    # Generate the beginning index and the ending index of a sample, which
    # contains (num_points_for_training + num_points_for_predicting) points
    indices = [(i, i + (num_timesteps_input + num_timesteps_output)) for i
               in range(X.shape[2] - (
                num_timesteps_input + num_timesteps_output) + 1)]

    # Save samples
    features, target = [], []
    for i, j in indices:
        features.append(
            X[:, :, i: i + num_timesteps_input].transpose(
                (0, 2, 1)))
        target.append(X[:, 0, i + num_timesteps_input: j])

    return torch.from_numpy(np.array(features)), \
           torch.from_numpy(np.array(target))


def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    return normalized_laplacian


def meanstd_normalization_tensor(tensor):
    # tensor: n_node, n_steps, n_dim
    n_node, n_steps, n_dim = tensor.shape
    # tensor_norm = np.ones([n_node, n_steps, n_dim])
    tensor_reshape = preprocessing.scale(np.reshape(tensor, [n_node, n_steps * n_dim]), axis=1)
    tensor_norm = np.reshape(tensor_reshape, [n_node, n_steps, n_dim])

    # print norm_x
    return tensor_norm
def sample_neighbors(adj_matrix, nodes, num_hops=1):
    all_nodes = set(nodes.tolist())
    current_level_nodes = set(nodes.tolist())

    for _ in range(num_hops):
        next_level_nodes = set()
        for node in current_level_nodes:
            neighbors = torch.nonzero(adj_matrix[node]).squeeze().tolist()
            if isinstance(neighbors, int):
                neighbors = [neighbors]
            next_level_nodes.update(neighbors)
        all_nodes.update(next_level_nodes)
        current_level_nodes = next_level_nodes

    return list(all_nodes)


def construct_subgraph(feature_matrix, adj_matrix, sampled_nodes, time_steps):
    sampled_nodes_tensor = torch.tensor(sampled_nodes, dtype=torch.long)
    subgraph_features = feature_matrix[sampled_nodes_tensor]
    subgraph_adj = adj_matrix[:, sampled_nodes_tensor, :][:, :, sampled_nodes_tensor]

    return subgraph_features, subgraph_adj


def create_mini_batch(feature_matrix, adj_matrix, labels, batch_size, sampled_flag, num_hops=1):
    num_nodes = feature_matrix.shape[0]
    time_steps = feature_matrix.shape[1]

    unsampled_nodes = torch.nonzero(~sampled_flag).squeeze().tolist()
    if isinstance(unsampled_nodes, int):
        unsampled_nodes = [unsampled_nodes]

    if len(unsampled_nodes) == 0:
        return None, None, None, sampled_flag

    if len(unsampled_nodes) < batch_size:
        center_nodes = unsampled_nodes
    else:
        center_nodes = random.sample(unsampled_nodes, batch_size)

    for node in center_nodes:
        sampled_flag[node] = True

    sampled_nodes = sample_neighbors(adj_matrix[0], torch.tensor(center_nodes, dtype=torch.long), num_hops)

    subgraph_features, subgraph_adj = construct_subgraph(feature_matrix, adj_matrix, sampled_nodes, time_steps)
    subgraph_labels = labels[torch.tensor(sampled_nodes, dtype=torch.long)]

    return subgraph_features, subgraph_adj, subgraph_labels, sampled_flag

