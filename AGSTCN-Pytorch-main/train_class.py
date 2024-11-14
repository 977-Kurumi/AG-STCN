import os
import torch
import torch.nn as nn
from tqdm import tqdm
import csv
import pickle as pk
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from model.AGSTCN_TST_GLU_F import AGSTCN
from util.data import load_custom_data
from util.utils import create_mini_batch





# 特征矩阵、邻接矩阵和标签如下：
path = "./data/reddit.npz"
dataset = "reddit"
adj_matrix, feature_matrix, labels, idx_train, idx_val, idx_test = load_custom_data(path, dataset)
labels2 = labels.argmax(dim=1)

# 超参数设置
batch_size = 32
num_epochs = 1000
learning_rate = 1e-3

# 模型初始化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AGSTCN(feature_matrix.shape[0], feature_matrix.shape[2], feature_matrix.shape[1], 10, dev=device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 打开文件准备写入
# 模型训练
Losses = []
ACC = []
F1 = []
Precision = []
Recall = []
with open('metrics_class_re.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Loss', 'Accuracy', 'F1', 'Precision', 'Recall'])
    for epoch in range(num_epochs):
        all_preds = []
        all_labels = []
        losses = []
        sampled_flag = torch.zeros(feature_matrix.shape[0], dtype=torch.bool)  # 每个epoch重新初始化采样标记

        with tqdm(total=feature_matrix.shape[0], desc=f'Epoch {epoch + 1}/{num_epochs}', unit='node') as pbar:
            while not sampled_flag.all():
                subgraph_features, subgraph_adj, subgraph_labels, sampled_flag = create_mini_batch(
                    feature_matrix, adj_matrix, labels2, batch_size, sampled_flag, num_hops=1
                )

                if subgraph_features is None:  # 无法创建新的mini-batch时，跳出循环
                    break

                subgraph_features = subgraph_features.float().to(device)
                subgraph_adj = subgraph_adj.float().to(device)
                subgraph_labels = subgraph_labels.long().to(device)

                model.train()
                optimizer.zero_grad()
                output = model(torch.unsqueeze(subgraph_features, 0), subgraph_adj)
                loss = criterion(output, subgraph_labels)
                loss.backward()
                optimizer.step()
                preds = torch.argmax(output, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(subgraph_labels.cpu())
                losses.append(loss.detach().cpu().numpy())

                pbar.update(len(subgraph_labels))

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        precision = precision_score(all_labels, all_preds, average='macro')
        recall = recall_score(all_labels, all_preds, average='macro')
        Losses.append(sum(losses) / len(losses))
        ACC.append(acc)
        F1.append(f1)
        Precision.append(precision)
        Recall.append(recall)

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {sum(losses) / len(losses)}, ACC: {acc}, F1: {f1}, Precision: {precision}, Recall: {recall}')
        writer.writerow([epoch + 1, sum(losses) / len(losses), acc, f1, precision, recall])
        # 将指标写入文件
        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses2_re.pk", "wb") as fd:
            pk.dump((Losses,
                     ACC,
                     F1,
                     Precision,
                     Recall,), fd)
