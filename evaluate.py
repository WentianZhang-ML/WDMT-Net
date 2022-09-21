import torch
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import evaluation.utils as utils
import os
import matplotlib.pyplot as plt
from skimage import io
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.metrics import top_k_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score



def pixel_wise_anomaly_detection(score, label):
    auc = roc_auc_score(label, score, average='micro')
    fpr, tpr, thres = roc_curve(label, score)    
    sum_ = [tpr[i]+1-fpr[i] for i in range(len(thres))]
    idx = np.argmax(np.array(sum_))
    TT = thres[idx]
    prob_f1 = [1 if s >= TT else 0 for s in score]
    acc = balanced_accuracy_score(label, prob_f1)
    f1 = f1_score(label, prob_f1, average='micro')
    return auc, acc, f1, TT


def eval_step(dataset, net, alpha, epoch = 0):
    score = []
    label = []
    loss_ = []
    test_loss = utils.AverageMeter()
    net.eval()
    data_batch = tqdm(dataset)
    data_batch.set_description("Evaluate")
    criterion = nn.MSELoss()
    with torch.no_grad():
        pdist_l1 = nn.PairwiseDistance(p=1).cuda()
        for iter_, (input_, img_id, img_hog, img_roi_label) in enumerate(data_batch):
            input_ = input_.cuda()
            feature, recon_image = net(input_, alpha)
            scores = pdist_l1(input_,recon_image[:,0,:,:].unsqueeze(1))
            loss = criterion(input_,recon_image[:,0,:,:].unsqueeze(1))
            test_loss.update(loss.item())

            for i in range(len(scores)):
                score_ = list(scores[i].view(-1).cpu().data.numpy())
                label_ = list(img_roi_label[i].view(-1).cpu().data.numpy())
                score.extend(score_)
                label.extend(label_)
        auc, acc, f1, thres = pixel_wise_anomaly_detection(score, label)
        loss_ave = test_loss.avg
    return auc, acc, f1, thres