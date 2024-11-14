import os, time
import argparse
import pickle as pk

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm

from baseline.TGCN import TGCN

from util.utils import generate_dataset, load_metr_la_data, get_normalized_adj

use_gpu = True
num_timesteps_input = 12
num_timesteps_output = 3

epochs = 1000
batch_size = 16
hidden_dim = 100
save_path = os.path.abspath(os.path.join(os.getcwd(), './result/weigths'))
save_path2 = os.path.abspath(os.path.join(os.getcwd(), './result/pig'))
parser = argparse.ArgumentParser(description='AGSTCN')
args = parser.parse_args()
args.device = torch.device('cuda')
if torch.cuda.is_available():
    args.device = torch.device('cuda')


# 更换了指标计算流程

def train_epoch(training_input, training_target, batch_size):
    """
    Trains one epoch with the given data.
    :param training_input: Training inputs of shape (num_samples, num_nodes,
    num_timesteps_train, num_features).
    :param training_target: Training targets of shape (num_samples, num_nodes,
    num_timesteps_predict).
    :param batch_size: Batch size to use during training.
    :return: Average loss for this epoch.
    """
    permutation = torch.randperm(training_input.shape[0])
    pbar = tqdm(total=training_input.shape[0], desc='Training', unit='batch')
    epoch_training_losses = []
    # for i in range(0, training_input.shape[0], batch_size):
    #     pbar.update(batch_size)
    #     net.train()
    #     optimizer.zero_grad()
    #     indices = permutation[i:i + batch_size]
    #     X_batch, y_batch = training_input[indices], training_target[indices]
    #     X_batch = X_batch.to(device=args.device)
    #     y_batch = y_batch.to(device=args.device)
    #     # tatol_loss = 0.0
    #     # for j in range(X_batch.shape[0]):
    #     #     X_data , Y_data = X_batch[j], y_batch[j]
    #     #     out = net( X_data.unsqueeze(0),A_wave)
    #     #     loss = loss_criterion(out, Y_data.unsqueeze(0))
    #     #  tatol_loss += loss
    #     #     # batch_loss.append(loss)
    #     # batch_loss = tatol_loss / batch_size
    #     out = net( X_batch,A_wave)
    #     loss = loss_criterion(out, y_batch)
    #     loss.backward()
    #     optimizer.step()
    #     epoch_training_losses.append(loss.detach().cpu().numpy())
    for i in range(0, training_input.shape[0], batch_size):
        pbar.update(batch_size)  # 更新进度条
        net.train()
        optimizer.zero_grad()  # 清除梯度以进行下一轮训练

        indices = permutation[i:i + batch_size]
        # indices2 = permutation[i:i + batch_size*207]
        X_batch, y_batch = training_input[indices], training_target[indices]
        X_batch = X_batch.to(device=args.device)
        y_batch = y_batch.to(device=args.device)

        out = net(X_batch)
        loss = loss_criterion(out, y_batch)
        loss.backward()  # 计算梯度
        optimizer.step()  # 使用计算得到的梯度更新参数
        epoch_training_losses.append(loss.detach().cpu().numpy())

        # 在 tqdm 进度条中显示当前 batch 的损失
        pbar.set_postfix({'loss': loss.item()})
    pbar.close()
    return sum(epoch_training_losses) / len(epoch_training_losses)



if __name__ == '__main__':
    torch.manual_seed(7)

    A, X, means, stds = load_metr_la_data()

    split_line1 = int(X.shape[2] * 0.6)
    split_line2 = int(X.shape[2] * 0.8)
    #
    # split_line1 = 100
    # split_line2 = 200

    train_original_data = X[:, :, :split_line1]
    val_original_data = train_original_data[:, :, 7563:7563 + 6855]
    # val_original_data = X[:, :, split_line1:split_line2]
    test_original_data = X[:, :, split_line2:]

    training_input, training_target = generate_dataset(train_original_data,
                                                       num_timesteps_input=num_timesteps_input,
                                                       num_timesteps_output=num_timesteps_output)
    # training_input = training_input.reshape(training_input.shape[0] * training_input.shape[1], training_input.shape[2],
    #                                         training_input.shape[3])
    # training_target = training_target.reshape(training_target.shape[0] * training_target.shape[1],
    #                                           training_target.shape[2])

    val_input, val_target = generate_dataset(val_original_data,
                                             num_timesteps_input=num_timesteps_input,
                                             num_timesteps_output=num_timesteps_output)
    # val_input = val_input.reshape(val_input.shape[0] * val_input.shape[1], val_input.shape[2], val_input.shape[3])
    #val_target = val_target.reshape(val_target.shape[0] * val_target.shape[1], val_target.shape[2])

    test_input, test_target = generate_dataset(test_original_data,
                                               num_timesteps_input=num_timesteps_input,
                                               num_timesteps_output=num_timesteps_output)
    #test_input = test_input.reshape(test_input.shape[0] * test_input.shape[1], test_input.shape[2], test_input.shape[3])
   # test_target = test_target.reshape(test_target.shape[0] * test_target.shape[1], test_target.shape[2])

    A_wave = get_normalized_adj(A)
    A_wave = torch.from_numpy(A_wave)

    A_wave = A_wave

    net = TGCN(A_wave, hidden_dim,num_timesteps_output).to(device=args.device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = nn.MSELoss()

    training_losses = []
    validation_losses = []
    validation_maes = []
    validation_rmse = []
    validation_mape = []
    validation_mse = []
    validation_r2 = []
    best_result = {"iter": 0, "val_MAE": 10.}
    for epoch in range(epochs):
        start_time_t = time.time()
        print("Epoch %d Training starts at %s" % (epoch, time.asctime(time.localtime(time.time()))))
        loss = train_epoch(training_input, training_target, batch_size=batch_size)
        print('[learning] epoch %i >> %3.2f%%' % (epoch, 100),
              'completed in %.2f (sec) <<' % (time.time() - start_time_t))
        print('Training:\tEpoch : %d\tTime : %.4fs\t Loss: %.5f' % (epoch, time.time() - start_time_t, loss))
        training_losses.append(loss)
        if epoch + 1 == epochs:
            torch.save(net.state_dict(), os.path.join(save_path, 'final_epoch.pth'))

        # Run validation
        with torch.no_grad():
            net.eval()
            val_input = val_input.to(device=args.device)
            val_target = val_target.to(device=args.device)

            epoch_validating_losses = []
            start_time_v = time.time()
            print("Validating Epoch %d starts at %s" % (epoch, time.asctime(time.localtime(time.time()))))
            pbar = tqdm(total=val_input.shape[0], desc='Validating', unit='sample')
            permutation = torch.randperm(val_input.shape[0])

            total_mae, total_rmse, total_mape, total_mse, total_r2 = 0.0, 0.0, 0.0, 0.0, 0.0

            for j in range(0, val_input.shape[0], batch_size):
                indices = permutation[j:j + batch_size]
                pbar.update(len(indices))  # Update by the number of samples in the current batch
                X_data, Y_data = val_input[indices], val_target[indices]
                # Y_data = Y_data.flatten(0,1)
                out = net(X_data)
                loss = loss_criterion(out, Y_data)
                epoch_validating_losses.append(loss.detach().cpu().item())

                out_unnormalized = out.detach().cpu().numpy() * stds[0] + means[0]
                target_unnormalized = Y_data.detach().cpu().numpy() * stds[0] + means[0]

                mae = np.mean(np.abs(out_unnormalized - target_unnormalized))
                rmse = np.sqrt(np.mean((out_unnormalized - target_unnormalized) ** 2))
                mask = target_unnormalized != 0
                mape = np.mean(
                    np.abs((out_unnormalized[mask] - target_unnormalized[mask]) / target_unnormalized[mask])) * 100
                mse = np.mean((out_unnormalized - target_unnormalized) ** 2)

                y_mean = np.mean(target_unnormalized)
                ssr = np.sum((target_unnormalized - out_unnormalized) ** 2)
                sst = np.sum((target_unnormalized - y_mean) ** 2)
                r2 = 1 - (ssr / sst) if sst != 0 else 0

                total_mae += mae * len(indices)
                total_rmse += rmse * len(indices)
                total_mape += mape * len(indices)
                total_mse += mse * len(indices)
                total_r2 += r2 * len(indices)

            pbar.close()

            total_samples = val_input.shape[0]
            val_loss = sum(epoch_validating_losses) / len(epoch_validating_losses)

            epoch_mae = total_mae / total_samples
            epoch_rmse = total_rmse / total_samples
            epoch_mape = total_mape / total_samples
            epoch_mse = total_mse / total_samples
            epoch_r2 = total_r2 / total_samples

            validation_losses.append(val_loss)
            # val_loss = sum(validation_losses) / len(validation_losses)

            validation_maes.append(epoch_mae)
            validation_rmse.append(epoch_rmse)
            validation_mape.append(epoch_mape)
            validation_mse.append(epoch_mse)
            validation_r2.append(epoch_r2)

            print('[Validating] epoch %i >> %3.2f%%' % (epoch, 100),
                  'completed in %.2f (sec) <<' % (time.time() - start_time_v))
            print('Validating:\tEpoch : %d\tTime : %.4fs\t Loss: %.5f' % (epoch, time.time() - start_time_v, val_loss))

            val_input = val_input.to(device="cpu")
            val_target = val_target.to(device="cpu")

        #
        print("Training loss: {}".format(training_losses[-1]))
        print("Validation loss: {}".format(validation_losses[-1]))
        print("Validation MAE: {}".format(validation_maes[-1]))
        print("Validation MSE: {}".format(validation_mse[-1]))
        print("Validation RMSE: {}".format(validation_rmse[-1]))
        print("Validation MAPE: {}".format(validation_mape[-1]))
        print("Validation R^2: {}".format(validation_r2[-1]))

        if epoch_mae <= best_result["val_MAE"]:
            torch.save(net.state_dict(), os.path.join(save_path, 'Epoch%03d_Loss%.5f_MAE%.5f_R2%.5f.pth' % (
            (epoch + 1, training_losses[-1], epoch_mae, epoch_r2))))
            print('NEW BEST:\tEpoch %d\t Loss: %.5f\t MAE: %.5f\t R^2: %.5f' % (
            epoch, training_losses[-1], epoch_mae, epoch_r2))
            print('save successfully!')
            best_result["val_MAE"] = epoch_mae
            best_result["iter"] = epoch

        plt.plot(training_losses, label="training loss")
        plt.plot(validation_losses, label="validation loss")
        plt.legend()
        plt.savefig(os.path.join(save_path2, "loss_history"))

        plt.cla()
        plt.close("all")

        checkpoint_path = "checkpoints/"
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        with open("checkpoints/losses1.pk", "wb") as fd:
            pk.dump((training_losses,
                     validation_losses,
                     validation_maes,
                     validation_mse,
                     validation_rmse,
                     validation_mape,
                     validation_r2), fd)
    print('FINAL BEST RESULT: \tEpoch : %d\tBest Val (MAE : %.4f)'
          % (best_result['iter'], best_result['val_MAE']))
