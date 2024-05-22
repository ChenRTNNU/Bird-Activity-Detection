# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:40:37 2024

@author: Administrator
"""

import os
import errno
import numpy as np
import matplotlib.pyplot as plt


#%%
def path_to_list(tmp_path):
    # tmp_path = training_path
    tmp_list = []
    class_list = os.listdir(tmp_path)
    for class_name in class_list:
        class_path = os.path.join(tmp_path, class_name)
        audio_list = os.listdir(class_path)
        
        for audio_name in audio_list:
            tmp_list.append(os.path.join(tmp_path, class_name, audio_name))
    return tmp_list

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def loader_to_label(test_loader):
    test_label = []
    for X, gt in test_loader:
        X = X.float()  # change to float
        test_label.append(gt) 
    return test_label   

def save_result(save_folder_final, gt_label, pred_label, training_loss_list, valid_loss_list, 
                training_feat, validation_feat, test_feat,
                training_label, validation_label, test_label):
    np.savetxt(save_folder_final + '/gt_label.csv', np.hstack(gt_label), delimiter=',')
    np.savetxt(save_folder_final + '/pred_label.csv', np.hstack(pred_label), delimiter=',')        
    np.savetxt(save_folder_final + '/training_loss.csv', np.array(training_loss_list), delimiter=',')
    np.savetxt(save_folder_final + '/valid_loss.csv', np.array(valid_loss_list), delimiter=',')
    
    np.save(save_folder_final + '/training_feat', np.array(training_feat))
    np.save(save_folder_final + '/validation_feat', np.array(validation_feat))
    np.save(save_folder_final + '/test_feat', np.array(test_feat))
    np.save(save_folder_final + '/training_label', np.array(training_label))
    np.save(save_folder_final + '/validation_label', np.array(validation_label))
    np.save(save_folder_final + '/test_label', np.array(test_label))
    
def show_loss(train_loss, valid_loss, save_folder_final):     
    # plot the loss
    plt.figure()
    plt.plot(train_loss)
    plt.plot(valid_loss)
    plt.title('Mode loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(save_folder_final + '/Mode_loss.png')
    plt.show()        
    
def show_f1score(train_f1score, valid_f1score, save_folder_final):   
    # plot the accuracy
    plt.plot(train_f1score)
    plt.plot(valid_f1score)
    plt.title('Mode F1-score')
    plt.ylabel('F1-score')
    plt.xlabel('Epoch')
    plt.legend(['train', 'valid'], loc='upper left')
    plt.savefig(save_folder_final + '/Mode_f1score.png')                
    plt.show()    
    













    

