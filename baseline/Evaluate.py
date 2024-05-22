
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