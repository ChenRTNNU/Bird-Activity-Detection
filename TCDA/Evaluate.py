
import torch
import torch.nn as nn
# import torchaudio.transforms as T
# import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_auc_score

# from torchvision.transforms import v2

def create_augmented_batch(data, targets, t, f):
    batch_size, _, _, _ = data.size()

    # 复制一份原始数据
    augmented_data = data.clone()
    augmented_targets = targets.clone()

    for i in range(batch_size):
        label = targets[i].item()

        # 从原始数据中选择一个相应标签的随机数据
        if label == 1:
            # print(positive_indices)
            # print(negative_indices)
            # print(all_indices)
            index = torch.randint(0, batch_size, (1,)).item()
            chosen_data = data[index]
            # print(targets[index])
        else:
            negative_indices = (targets == 0).nonzero().squeeze(1)
            chosen_index = torch.randint(len(negative_indices), (1,)).item()
            chosen_data = data[negative_indices[chosen_index]]
            index = negative_indices[chosen_index]

        # 随机选择时间轴上的一段长度为t的区域
        time_axis_length = chosen_data.size(1)

        freq_axis_length = chosen_data.size(2)

        # print(augmented_data.shape) 56080
        # print(chosen_data.shape)
        # print(time_axis_length)
        # print(freq_axis_length)
        # print(augmented_targets.shape)
        # print(augmented_targets)
        start_time = torch.randint(0, time_axis_length - t + 1, (1,)).item()
        start_freq = torch.randint(0, freq_axis_length - f + 1, (1,)).item()
        # 对选择的区域进行算数平均
        augmented_data[i, :, start_time:start_time + t, :] = (data[i, :, start_time:start_time + t, :] + chosen_data[:,
                                                                                                         start_time:start_time + t,
                                                                                                         :]) / 2.0

        augmented_data[i, :, :, start_freq:start_freq + f] = (data[i, :, :, start_freq:start_freq + f] + chosen_data[:,
                                                                                                         :,
                                                                                                         start_freq:start_freq + f]) / 2.0
        # print(targets[index],targets[i])
        # augmented_targets[i]=(t*freq_axis_length+f*time_axis_length-t*f)/(time_axis_length*freq_axis_length)*0.5*targets[index]+targets[i]-(t*freq_axis_length+f*time_axis_length-t*f)/(time_axis_length*freq_axis_length)*0.5*targets[i]
    # print(targets)
    # print(augmented_targets)
    return augmented_data, augmented_targets
#%%
def loss_fn(outputs, labels):
    return nn.BCELoss()(outputs, labels.unsqueeze(1).float())  # Binary Cross-Entropy Loss for binary classification
 
    
#%%
def train(model, data_loader, optimizer, scheduler, device, epoch, epochs):
    model.train()

    pred = []
    label = []
    output = []
    running_loss = 0
    loop = tqdm(data_loader, position=0)
    for i, (mels, labels) in enumerate(loop):
        mels = mels.float()  # change to float
        # mels, labels = create_augmented_batch(mels, labels, 250, 25)

        mels = mels.to(device)
        
        # labels = labels.type(torch.LongTensor)
        labels = labels.to(device)
        outputs = model(mels)

        # print(mels.shape)
        # print(labels.shape)
                
        # _, preds = torch.max(outputs, 1)

        # loss = loss_fn(outputs, labels)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        
        # pred.extend(preds.view(-1).cpu().detach().numpy())
        output.extend(outputs.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())       
        
        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        loop.set_postfix(loss=loss.item())
    label_numpy=np.array(label)
    output_numpy=np.array(output)
    pred_numpy=np.where(output_numpy >= 0.5, 1, 0)
    train_acc = accuracy_score(label_numpy, pred_numpy)
    train_f1 =f1_score(label_numpy, pred_numpy)
    return running_loss / len(data_loader), train_acc,train_f1,model


def valid(model, data_loader, device, epoch, epochs):
    model.eval()
    
    running_loss = 0
    pred = []
    label = []
    output = []
    loop = tqdm(data_loader, position=0)
    for mels, labels in loop:
        mels = mels.float()  # change to float
        mels = mels.to(device)

        # labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        outputs = model(mels)
        #_, preds = torch.max(outputs, 1)

        loss = loss_fn(outputs, labels)

        running_loss += loss.item()

        output.extend(outputs.view(-1).cpu().detach().numpy())
        label.extend(labels.view(-1).cpu().detach().numpy())

        loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
        loop.set_postfix(loss=loss.item())
    label_numpy=np.array(label)
    output_numpy=np.array(output)
    pred_numpy=np.where(output_numpy >= 0.5, 1, 0)
    valid_acc = accuracy_score(label_numpy, pred_numpy)
    valid_f1 =f1_score(label_numpy, pred_numpy)
    return running_loss / len(data_loader), valid_acc,valid_f1


def eval_acc(model, device, dataloader, debug_name=None):
    model = model.to(device).eval()
    count = correct = 0
    gt_label = []
    pred_label = []
    for X, gt in dataloader:
        X = X.float()  # change to float
        logits = model(X.to(device))
        preds = torch.argmax(logits, dim=1)
        correct += sum(preds.to('cpu') == gt)
        count += len(gt)
        gt_label.append(gt)
        pred_label.append(preds)
    acc = correct / count
    if debug_name:
        print(f'{debug_name} acc = {acc:.4f}')
    return acc, gt_label, pred_label
    
def eval_auc(model, device, dataloader, debug_name=None):
    model = model.to(device).eval()
    gt_labels = []
    pred_probs = []
    for X, gt in dataloader:
        X = X.float().to(device)  # change to float and move to device
        logits = model(X)
        # probs = torch.softmax(logits, dim=1)
        gt_labels.append(gt.numpy())
        pred_probs.append(logits.detach().cpu().numpy())  # assuming binary classification and getting probability of positive class

    gt_labels = np.concatenate(gt_labels)
    pred_probs = np.concatenate(pred_probs)

    auc = roc_auc_score(gt_labels, pred_probs)
    if debug_name:
        print(f'{debug_name} AUC = {auc:.4f}')
    return auc, gt_labels, pred_probs