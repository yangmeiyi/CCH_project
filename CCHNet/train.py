import argparse
import datetime
import numpy as np
import time
import torch
import os
import torch.nn.functional as F
import shutil
import pandas as pd
import random
from sklearn import metrics
import copy
from torchvision import transforms
from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from scipy.stats import wasserstein_distance
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma, ApexScaler
from dataloader import CCHSeT_CSV
from utils import Logger, AverageMeter, accuracy, lipschitz
from CCHNet.model import cmt_s
from discriminator import CTDiscriminator
from pytorch_pretrained_vit import ViT


def get_args_parser():
    parser = argparse.ArgumentParser('CMT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=200, type=int)

    # Model parameters
    parser.add_argument('--model', default='cmt_s', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--image_size', default=224, type=int, help='images input size')
    parser.add_argument('--padding_size', default=0, type=int, help='images crop size')
    parser.add_argument('--crop_size', default=224, type=int, help='images crop size')
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument('--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float, default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.9,
                        help='weight decay (default: 0.9)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Augmentation parameters
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--upload', action='store_true', default=False,
                        help='upload the pretrained param')
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # Dataset parameters
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--gpus', default='0,1,2,3,4,5,6',
                        help='device id to use for training / testing')
    parser.add_argument('--seed', default=3407, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--test_freq', type=int, default=20)
    parser.add_argument('--test_epoch', type=int, default=260)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--update_temperature', action='store_true')
    parser.add_argument('--warmup-drop-path', action='store_true')
    parser.add_argument('--warmup-drop-path-epochs', type=int, default=20)
    return parser

parser = argparse.ArgumentParser('CMT training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
use_cuda = torch.cuda.is_available()
criterion = LabelSmoothingCrossEntropy(smoothing=0.2)
mse_loss = torch.nn.MSELoss(reduce=True)
# criterion = torch.nn.CrossEntropyLoss()

class dis_calculate():
    def __init__(self, discriminator, real, fake, lipconstraint, lambda2, Mtag=0):
        self._dis = discriminator
        self._real = real
        self._fake = fake
        self._lip = lipconstraint
        self._lambda2 = lambda2
        self._Mtag = Mtag

    def dis_cal(self):
        for param in self._dis.parameters():
            param.requires_grad = True
        self._dis.zero_grad()
        dreal, _, creal = self._dis(self._real)
        drealm = torch.mean(dreal, dim=0)
        drealm.backward(torch.cuda.FloatTensor([-1.]))
        dfake, _, cfake = self._dis(self._fake)
        dfake = torch.mean(dfake, dim=0)
        dfake.backward(torch.cuda.FloatTensor([1]))
        loss_penalty = self._lip.cal_gradient(self._real, self._fake)
        d1, d_1, c_1 = self._dis(self._real)
        d2, d_2, c_2 = self._dis(self._real)
        ctpenalty = self._lambda2 * ((d1 - d2) ** 2).mean(dim=1)
        ctpenalty += self._lambda2 * 0.1 * ((d_1 - d_2) ** 2).mean(dim=1)
        ctpenalty = torch.max(torch.zeros(ctpenalty.size()).cuda()
                              if torch.cuda.is_available() else torch.zeros(ctpenalty.size())
                              , ctpenalty - self._Mtag).mean()
        ctpenalty.backward()
        discost = dfake - dreal + loss_penalty + ctpenalty
        return discost.mean()


def train(model, discriminator, optimizer, optimizer_dis, data_loader):
    total_samples = len(data_loader.dataset)
    print("total_samples", total_samples)
    model.train()
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    people_id = []
    neg_pred_list = []
    labels = []

    for idx, data in enumerate(data_loader):
        inputs = data["img"].float()
        targets = data["label"].float()

        if use_cuda:
            inputs, targets = inputs.cuda(non_blocking=True), targets.long().cuda(non_blocking=True)
            # inputs, targets = Variable(inputs.cuda(), requires_grad=True), targets.long().cuda(non_blocking=True)
            targets = targets.squeeze()

        inputs_224 = copy.deepcopy(inputs)
        inputs_112 = inputs_224.resize_(inputs.size(0), 3, 112, 112)

        feature, outputs, _ = model(inputs)
        random_discriminator = random.random()
        if random_discriminator < 0.01:
            with torch.no_grad():
                outputs_112 = feature.clone()
                outputs_112.resize_(inputs.size(0), 3, 112, 112)
            for i in range(2):
                discost = dis_calculate(discriminator, inputs_112, outputs_112,
                                        lipschitz.GradientPenalty(discriminator), 2.0)
                cost = discost.dis_cal()
                optimizer_dis.step()

        loss = criterion(outputs, targets)
        neg_pred = outputs[:, 1]
        people_id.extend(data['name'])
        neg_pred_list.extend(neg_pred.detach().cpu().numpy().tolist())
        labels.extend(targets.detach().cpu().numpy().tolist())
        acc = accuracy(outputs.data, targets.data, topk=(1,))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print('[' + '{:5}'.format(idx * len(data["img"])) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * idx / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()) + ' Acc:' + '{:6.2f}'.format(acc[0].item()))

    df = pd.DataFrame({'people_id': people_id, 'neg_preds': neg_pred_list, 'labels': labels})

    acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
    single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
    statistic_sensitivity, statistic_specificity = Auc(df)
    print("training SE(pos): {}".format(statistic_sensitivity) + "  training SP(neg): {}".format(statistic_specificity))


def evaluate(model, data_loader):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0
    people_id = []
    neg_pred_list = []
    labels = []

    with torch.no_grad():

        for idx, data in enumerate(data_loader):
            inputs = data["img"].float()
            targets = data["label"].float()

            if use_cuda:
                inputs, targets = inputs.cuda(non_blocking=True), targets.long().cuda(non_blocking=True)
                # inputs, targets = Variable(inputs.cuda(), requires_grad=True), targets.long().cuda(non_blocking=True)
                targets = targets.squeeze()

            outputs = F.log_softmax(model(inputs)[1], dim=1)  #多了一个[0]
            predicts = F.softmax(model(inputs)[1], dim=1) #多了一个[0]
            loss = F.nll_loss(outputs, targets, reduction='sum')
            _, pred = torch.max(outputs, dim=1)
            neg_pred = predicts[:, 1]

            total_loss += loss.item()
            correct_samples += pred.eq(targets).sum()


            people_id.extend(data['name'])
            neg_pred_list.extend(neg_pred.detach().cpu().numpy().tolist())
            labels.extend(targets.detach().cpu().numpy().tolist())

        df = pd.DataFrame({'people_id': people_id, 'neg_preds': neg_pred_list, 'labels': labels})

        acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
        single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
        statistic_sensitivity, statistic_specificity = Auc(df)
        single_sensitivity = 1 - single_fpr
        single_specificity = single_tpr
        optimal_single_sensitivity = 1 - single_point[0]
        optimal_single_specificity = single_point[1]
        statistics_sensitivity = 1 - statistic_fpr
        statistics_specificity = statistic_tpr
        optimal_statistics_sensitivity = statistic_point[0]
        optimal_statistics_specificity = statistic_point[1]

        # return neg_pred_list, labels

        # get_RoC(df)


    return acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
           single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
           statistic_sensitivity, statistic_specificity


def Auc(df):
    def threshold(ytrue, ypred):
        fpr, tpr, thresholds = metrics.roc_curve(ytrue, ypred)
        y = tpr - fpr
        youden_index = np.argmax(y)
        optimal_threshold = thresholds[youden_index]
        point = [fpr[youden_index], tpr[youden_index]]
        roc_auc = metrics.auc(fpr, tpr)
        return optimal_threshold, point, fpr, tpr, roc_auc

    single_threshold, single_point, single_fpr, single_tpr, single = threshold(df['labels'], df['neg_preds'])
    print(single_threshold)
    df['single'] = (df['neg_preds'] >= 0.5).astype(int)  # single_threshold 改为0.5
    acc_single = (df['labels'] == df['single']).mean()
    df_signal_sensitivity = df.loc[df["labels"] == 1]
    signal_sensitivity = (df_signal_sensitivity['labels'] == df_signal_sensitivity['single']).mean()
    df_signal_specificity = df.loc[df["labels"] == 0]
    signal_specificity = (df_signal_specificity['labels'] == df_signal_specificity['single']).mean()
    print("training signal image SE(pos): {}".format(signal_sensitivity) + "  training signal image SP(neg): {}".format(signal_specificity))


    df = df.groupby('people_id')[['labels', 'neg_preds']].mean()
    statistic_threshold, statistic_point, statistic_fpr, statistic_tpr, statis = threshold(df['labels'], df['neg_preds'])
    np.save('cmt_neg_preds.npy', df['neg_preds'])
    # np.save('cmt_fpr_1.npy', statistic_fpr)
    # np.save('cmt_tpr_1.npy', statistic_tpr)
    # print(statistic_fpr)
    # print(statistic_tpr)
    # print(statistic_threshold)
    df['outputs'] = (df['neg_preds'] >= 0.5).astype(int)  # statistic_threshold 改为0.5
    acc_statistic = (df['labels'] == df['outputs']).mean()
    df_sensitivity = df.loc[df["labels"] == 1]
    statistic_sensitivity = (df_sensitivity['labels'] == df_sensitivity['outputs']).mean()
    df_specificity = df.loc[df["labels"] == 0]
    statistic_specificity = (df_specificity['labels'] == df_specificity['outputs']).mean()

    return acc_single, acc_statistic, single, statis, single_threshold, statistic_threshold, \
           single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
           statistic_sensitivity, statistic_specificity


def get_RoC(df):
    label = df['labels']
    scores = df['neg_preds']
    fpr, tpr, _ = metrics.roc_curve(label, scores)
    np.save('fpr_6.npy', fpr)
    np.save('tpr_6.npy', tpr)

def save_acc_checkpoint(state, is_best, checkpoint= "./save_model/2_9/", filename='cchnet_acc_1.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'cchnet_acc_best_1.pth.tar'))

def save_auc_checkpoint(state, is_best, checkpoint= "./save_model/2_9/", filename='cchnet_auc_1.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'cchnet_auc_best_1.pth.tar'))



print('==> Preparing dataset')
PATH = "/home/ubuntu/Code/Code/CCH/datasets/"
Liver_loader_train = CCHSeT_CSV(PATH, 'train', args)
Liver_loader_test = CCHSeT_CSV(PATH, 'test', args)
Liver_loader_val = CCHSeT_CSV(PATH, 'val', args)
train_loader = torch.utils.data.DataLoader(Liver_loader_train, batch_size=args.batch_size, shuffle=True,
                                           drop_last=False)
test_loader = torch.utils.data.DataLoader(Liver_loader_test, batch_size=args.batch_size, shuffle=False,
                                          drop_last=False)
val_loader = torch.utils.data.DataLoader(Liver_loader_val, batch_size=args.batch_size, shuffle=False,
                                          drop_last=False)


if __name__ == '__main__':
    model = cmt_s()
    discriminator = CTDiscriminator(0.2, 0.5, 0.5)

    if use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        discriminator = torch.nn.DataParallel(discriminator).cuda()

    if args.upload:
        checkpoint = torch.load(
            r"/home/ubuntu/Code/Code/CCH/code/save_model/2_9/cmt_acc_best_3.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=4e-5)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[100, 150], gamma=0.1)
    optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0, 0.9))
    lr_scheduler_dis = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer_dis, milestones=[100, 150], gamma=0.1)



    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    best_person_auc = 0
    best_person_acc = 0

    for epoch in range(args.start_epoch, args.epochs): #args.epochs
        print('Epoch:', epoch)
        start_time = time.time()
        train(model, discriminator, optimizer, optimizer_dis, train_loader)

        print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')
        acc_single, acc_statistic, single, auc_statistic, single_threshold, statistic_threshold, \
        single_fpr, single_tpr, single_point, statistic_fpr, statistic_tpr, statistic_point, \
        statistic_sensitivity, statistic_specificity = evaluate(model, test_loader)

        val_acc_single, val_acc_statistic, val_single, val_auc_statistic, val_single_threshold, val_statistic_threshold, \
        val_single_fpr, val_single_tpr, val_single_point, val_statistic_fpr, val_statistic_tpr, val_statistic_point, \
        val_statistic_sensitivity, val_statistic_specificity = evaluate(model, val_loader)

        print("SE(pos): {}".format(statistic_sensitivity) + "  SP(neg): {}".format(statistic_specificity))
        print("signle acc: {}".format(acc_single) + " signle auc: {}".format(single) + \
              " person auc: {}".format(auc_statistic) + " person acc: {}\n".format(acc_statistic))

        print("val_SE(pos): {}".format(val_statistic_sensitivity) + "  val_SP(neg): {}".format(val_statistic_specificity))
        print("val_signle acc: {}".format(val_acc_single) + " val_signle auc: {}".format(val_single) + \
              " val_person auc: {}".format(val_auc_statistic) + " val_person acc: {}\n".format(val_acc_statistic))

        best_person_auc = max(best_person_auc, auc_statistic)
        best_person_acc = max(best_person_acc, acc_statistic)
        auc_is_best = auc_statistic >= best_person_auc
        acc_is_best = acc_statistic >= best_person_acc

        # save model
        save_acc_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'auc': best_person_auc,
            'acc': best_person_acc,
            'optimizer': optimizer.state_dict(),
        }, acc_is_best)

        save_auc_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'auc': best_person_auc,
            'acc': best_person_acc,
            'optimizer': optimizer.state_dict(),
        }, auc_is_best)

        print(" best person auc: {}".format(best_person_auc) + " best person acc: {}\n".format(best_person_acc))
        lr_scheduler.step()