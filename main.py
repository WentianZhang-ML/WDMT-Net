# encoding:utf-8
import argparse
import os
import sys
import time
import random
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

import evaluate as eval_
import utils as utils
import models as net

sys.path.append("../..")  # root dir
os.chdir(sys.path[0])  # current work dir

seed = 5
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='IDRiD', help='[IDRiD/ADAM]')
parser.add_argument("--datadir", type=str, default='./labels/', help='dataset json files')
parser.add_argument('--hog', dest='hog', action='store_true', help='use hog prediction')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--batch_size", type=float, default=32)
parser.add_argument("--epochs", type=float, default=200)
parser.add_argument("--eval_epoch", type=float, default=1)
parser.add_argument("--savedir", type=str, default='./models_logs/')
parser.add_argument("--deta", type=float, default=0)

def train(args):
    _time = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    logger_ = utils.logg(args, _time)

    if not args.hog:
        model = net.UNet_cut()
    else:
        model = net.UNet_cut(n_classes=2)
    train_step(model, args=args, logger=logger_, time_=_time, cut=True)



def train_step(net, args, logger, time_, cut = False):
    start_epoch = -1
    criterion = nn.MSELoss().cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-5, amsgrad=True)

    dataset = os.path_join(args.datadir,args.dataset)

    train_data, test_data = utils.get_dataset(dataset, args.batch_size)

    len_batch = train_data.__len__()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len_batch*args.epochs)

    net.cuda()

    save_dir = args.savedir + args.dataset + '/' + args.model + '/saved_models/Time_' + str(time_) + 'lr_' + str(args.lr) + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    [best_auc, best_precision, best_recall, best_acc, best_f1, best_thres] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    logger.info('**************************** start training target model! ******************************\n')
    logger.info(
        '---------|---------------------- VALID ---------------------|-- Training --|---------- Current Best ----------|\n')
    logger.info(
        '  epoch  |   AUC   PRECISION   RECALL   ACC   F-1   Thres   |     loss     |    AUC     ACC     F-1    Thres  |\n')
    logger.info(
        '--------------------------------------------------------------------------------------------------------------|\n')
    alpha_ = 1
    for epoch in range(start_epoch+1, args.epochs):
        training_loss = utils.AverageMeter()
        if args.hog:
            training_loss1 = utils.AverageMeter()
            training_loss2 = utils.AverageMeter()
        data_batch = tqdm(train_data)
        for iter_, (input_, img_id, img_hog, img_label) in enumerate(data_batch):
            if cut:
                alpha_cur = alpha_ - args.deta / float(len_batch)
                if alpha_cur > 0:
                    alpha_cur = alpha_cur
                else:
                    alpha_cur = 0
            else:
                alpha_cur = 1

            input_ = input_.cuda()
            if args.hog:
                img_hog = img_hog.cuda()

            net.train()
            feature, recon_image = net(input_,alpha_cur)
            if args.hog:
                loss1 = criterion(input_, recon_image[:, 0, :, :].unsqueeze(1))
                loss2 = criterion(img_hog, recon_image[:, 1, :, :].unsqueeze(1))
                loss = loss1+loss2
            else:
                loss = criterion(input_, recon_image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.update(loss.item())
            if args.hog:
                training_loss1.update(loss1.item())
                training_loss2.update(loss2.item())

            alpha_ = alpha_cur
            scheduler.step()

        if (epoch + 1) % args.eval_epoch == 0:
            [auc, acc, f1, thre]=eval_.eval_step(test_data, net, alpha=alpha_cur)

            thres = thre
            precision = 1
            recall = 1

            is_best = auc >= best_auc
            if is_best:
                best_auc = auc
                best_acc = acc
                best_f1 = f1
                best_thres = thres

                save_path = save_dir + 'Epoch_' + str(epoch + 1) + 'AUC_' + str(
                    round(auc * 100, 4)) + '_ACC_' + str(round(acc * 100, 4)) + 'F1_' + str(
                    round(f1 * 100, 4))+'_alpha_'+str(alpha_cur)+'.pth'

                checkpoint = {
                    "net": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch
                    }
                torch.save(checkpoint, save_path)
            logger.info(
                '  %3d  |  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f  %5.3f  |  %5.6f  |  %5.3f  %5.3f  %5.3f  %5.3f  |'
                % (
                    epoch + 1,
                    auc * 100, precision * 100, recall * 100, acc * 100, f1 * 100, thres,
                    training_loss.avg,
                    float(best_auc * 100), float(best_acc * 100), float(best_f1 * 100), best_thres))


if __name__ == "__main__":
    args = parser.parse_args()
    train(args)
