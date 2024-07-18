import os
# import random
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
# import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.optim as optim
# from scipy.ndimage import median_filter
# import matplotlib.pyplot as plt
# from torch.utils.data.dataset import ConcatDataset
from tqdm import tqdm

from CMM.Evaluate import train, valid, eval_acc, eval_auc
from utils import make_sure_path_exists, show_f1score, show_loss

# train_df3, test_df3 = train_test_split(df3, test_size=0.5, shuffle=True, random_state=42)


def specaug(data, targets):
    batch_size, _, H, W = data.size()
    for i in range(0, batch_size):
        time_masking = T.TimeMasking(time_mask_param=100)
        frequency_masking = T.FrequencyMasking(freq_mask_param=10)
        data[i] = time_masking(frequency_masking(data[i]))
    return data, targets


# 定义自定义数据集类
class AudioDataset(Dataset):
    def __init__(self, dataframe1, data_folder1, sr=22050, dimension=220500):
        self.dataframe1 = dataframe1
        # self.dataframe2 = dataframe2
        self.data_folder1 = data_folder1
        # self.data_folder2 = data_folder2
        self.sr = sr
        self.dim = dimension

    def __getitem__(self, index):
        # print(self.data_folder1)
        # print(str(self.dataframe1.iloc[index]['itemid']))
        # filename1 = os.path.join(self.data_folder1[0], str(self.dataframe1.iloc[index]['itemid']) + ".wav")
        filename1 = os.path.join(self.data_folder1, str(self.dataframe1.iloc[index]['itemid']) + ".wav")
        # print(filename1)
        wb_wav, _ = librosa.load(filename1, sr=self.sr)
        if len(wb_wav) >= self.dim:
            wb_wav = wb_wav[0: self.dim]
        else:
            wb_wav = np.pad(wb_wav, (0, self.dim - len(wb_wav)), "constant")

        # if (self.data_folder2 != ""):
        #     r2 = random.randint(0, len(self.dataframe2) - 1)
        #     filename2 = os.path.join(self.data_folder2, str(self.dataframe2.iloc[r2]['itemid']) + ".wav")
        #     wb_s, _ = librosa.load(filename2, sr=self.sr)
        #     if len(wb_s) >= self.dim:
        #         wb_s = wb_s[0: self.dim]
        #     else:
        #         wb_s = np.pad(wb_s, (0, self.dim - len(wb_s)), "constant")
        #     wb_wav = wav_synthesize(wb_wav, wb_s)

        # wb_wav = add_noise(wb_wav)
        # wb_wav = pitch_shift_spectrogram(wb_wav)
        # wb_wav = time_shift_spectrogram(wb_wav)

        frame_length = 1024  # 46 ms frames
        hop_length = 315  # 14 ms hop size
        n_fft = frame_length
        n_mels = 80
        fmin = 50
        fmax = 11000

        # Calculate log mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=wb_wav,
            sr=self.sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            window='hann'
        )

        # Convert to log scale
        wb_wav = librosa.power_to_db(abs(mel_spec), ref=np.max)
        wb_wav_tensor = torch.tensor(wb_wav, dtype=torch.float32)
        wb_wav_tensor = wb_wav_tensor.transpose(0, 1)
        wb_wav_tensor = torch.unsqueeze(wb_wav_tensor, dim=0)

        label = self.dataframe1.iloc[index]['hasbird']
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return wb_wav_tensor, label_tensor

    def __len__(self):
        return len(self.dataframe1)

# 创建训练集和测试集的自定义数据集对象



# 使用 DataLoader加载数据
import torch.nn.functional as F
import torch.nn as nn


class BirdSoundCNN(nn.Module):
    def __init__(self):
        super(BirdSoundCNN, self).__init__()

        # Batch Normalization
        self.batch_norm0 = nn.BatchNorm2d(1)

        # Convolutional layers with MaxPooling
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 1))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 1))
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(3, 1))
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=16 * 7 * 8, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)  # Assuming binary classification

        # self.fc1 = nn.Linear(in_features=112, out_features=1)

        # Dropout layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # print(x.shape)
        x = self.batch_norm0(x)
        # print(x.shape)
        # Convolutional layers with MaxPooling and LeakyReLU activation
        x = F.leaky_relu(self.conv1(x))
        # print(x.shape)
        x = self.maxpool1(x)
        # print(x.shape)
        x = F.leaky_relu(self.conv2(x))
        # print(x.shape)
        x = self.maxpool2(x)
        # print(x.shape)
        x = F.leaky_relu(self.conv3(x))
        # print(x.shape)
        x = self.maxpool3(x)
        # print(x.shape)
        x = F.leaky_relu(self.conv4(x))
        # print(x.shape)
        x = self.maxpool4(x)
        # print(x.shape)

        # Flatten
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        # print(x.shape)
        # Fully connected layers with LeakyReLU activation and Dropout
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))

        # x = torch.sigmoid(self.fc1(x))

        # print(x.shape)

        return x  # Assuming binary classification


import logging

# import matplotlib.pyplot as plt
# from sklearn.metrics import roc_auc_score

# %%
def run(iRepeat,savepath):

    # 读取CSV文件
    csv_file_path1 = 'sound/ff1010bird.csv'
    df1 = pd.read_csv(csv_file_path1)
    csv_file_path2 = 'sound/warblrb10k.csv'
    df2 = pd.read_csv(csv_file_path2)
    # csv_file_path3 = './my_index/BirdVoxDCASE20k_csvpublic.csv'
    # df3 = pd.read_csv(csv_file_path3)
    df1P = df1[df1['hasbird'] == 1]
    df1N = df1[df1['hasbird'] == 0].sample(n=2000)
    df2P = df2[df2['hasbird'] == 1].sample(n=2000)
    df2N = df2[df2['hasbird'] == 0]
    df1_2k = pd.merge(df1P, df1N, how='outer')
    df2_2k = pd.merge(df2P, df2N, how='outer')
    # 划分数据集
    training_df1 = pd.read_csv('dataset/train_f.csv')
    test_df1 = pd.read_csv('dataset/test_f.csv')
    valid_df1 = pd.read_csv('dataset/valid_f.csv')

    train_df2 = pd.read_csv('dataset/train_w.csv')
    test_df2 = pd.read_csv('dataset/train_w.csv')

    training_dataset_ff = AudioDataset(training_df1, data_folder1='sound/ff1010bird')
    valid_dataset_ff = AudioDataset(valid_df1, data_folder1='sound/ff1010bird')
    test_dataset_ff = AudioDataset(test_df1, data_folder1='sound/ff1010bird')
    test_dataset_warb = AudioDataset(test_df2, data_folder1='sound/warblrb10k')
    print(training_dataset_ff.__len__())
    batch_size = 64

    training_loader_ff = DataLoader(training_dataset_ff, batch_size=batch_size, shuffle=True)
    valid_loader_ff = DataLoader(valid_dataset_ff, batch_size=batch_size, shuffle=True)
    test_loader_ff = DataLoader(test_dataset_ff, batch_size=batch_size, shuffle=True)

    test_loader_warb = DataLoader(test_dataset_warb, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    model = BirdSoundCNN()

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss function and optimizer
    # criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for binary classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = None

    # %%
    loss_history = []

    # Training loop
    checkpoint_path = 'checkpoints/checkpoint_time100_freq10_re.pth'  # Specify the path where you want to save the checkpoint
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")

    else:
        start_epoch = 0
        print("No checkpoint found. Starting training from the beginning.")

    # %% Training loop
    save_folder_final = os.path.join(savepath, 'repeat_' + str(iRepeat))
    make_sure_path_exists(save_folder_final)
    log_file_path = save_folder_final + '/log_spec_plus_t300_freq30.txt'
    logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    training_f1_list, valid_f1_list = [], []
    training_loss_list, valid_loss_list = [], []

    best_valid_f1 = 0

    epochs = 200
    for epoch in range(start_epoch, epochs):
        # print(epoch)

        training_loss,train_acc, training_f1, model = train(model, training_loader_ff, optimizer, scheduler, device, epoch,
                                                  epochs)
        valid_loss,valid_acc, valid_f1 = valid(model, valid_loader_ff, device, epoch, epochs)

        # valid_loss_my, valid_my = valid(model, valid_loader_ff, device, epoch, epochs)

        training_f1_list.append(training_f1)
        valid_f1_list.append(valid_f1)

        training_loss_list.append(training_loss)
        valid_loss_list.append(valid_loss)
        print(f"train_f1{training_f1},valid_f1:{valid_f1},train_acc:{train_acc},valid_acc:{valid_acc}")
        print(valid_f1)
        if valid_f1 > best_valid_f1:
            print(f"Validation F1 Improved - {best_valid_f1} ---> {valid_f1}")
            logging.info(f"Validation F1 Improved - {best_valid_f1} ---> {valid_f1}")
            torch.save(model.state_dict(), os.path.join(save_folder_final, f'model_{iRepeat}.pth'))
            # torch.save(model, os.path.join(save_folder_final, f'model_{iRepeat}.bin'))
            print(f"Saved model checkpoint at ./model_{iRepeat}.pth")
            best_valid_f1 = valid_f1
        logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {training_loss}')
        logging.info(f'Epoch {epoch + 1}/{epochs}, Loss: {valid_loss}')

    # %% load test
    model = BirdSoundCNN()
    model.to(device)
    model.load_state_dict(torch.load(os.path.join(save_folder_final, f'model_{iRepeat}.pth')))
    model.eval()

    # %%
    # acc, gt_label, pred_label = eval_acc(model, device, test_loader_ff, 'test')
    auc, _, _ = eval_auc(model, device, test_loader_ff, 'test')
    # save the result
    # np.savetxt(save_folder_final + '/gt_label.csv', torch.cat(gt_label).cpu().numpy(), delimiter=',')
    # np.savetxt(save_folder_final + '/pred_label.csv', torch.cat(pred_label).cpu().numpy(), delimiter=',')

    # save the result
    # acc_warb, gt_label_warb, pred_label_warb = eval_acc(model, device, test_loader_warb, 'test')
    auc_warb, _, _ = eval_auc(model, device, test_loader_warb, 'test')
    # np.savetxt(save_folder_final + '/gt_label_warb.csv', torch.cat(gt_label_warb).cpu().numpy(), delimiter=',')
    # np.savetxt(save_folder_final + '/pred_label_warb.csv', torch.cat(pred_label_warb).cpu().numpy(), delimiter=',')

    np.savetxt(save_folder_final + '/training_loss.csv', np.array(training_loss_list), delimiter=',')
    np.savetxt(save_folder_final + '/valid_loss.csv', np.array(valid_loss_list), delimiter=',')
    logging.info(f'Test1 AUC: {auc}')
    logging.info(f'Test2 AUC: {auc_warb}')
    ############################
    # visualization
    show_loss(np.array(training_loss_list), np.array(valid_loss_list), save_folder_final)
    show_f1score(np.array(training_f1_list), np.array(valid_f1_list), save_folder_final)





