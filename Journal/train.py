from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from models.backbone import ResNet_18
import dataset.raf as dataset
from losses import SupConLoss
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=60, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=1, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.00025, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=2, help='manual seed')
#Device options
parser.add_argument('--gpu', default='1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
parser.add_argument('--n-labeled', type=int, default=4000,
                        help='Number of labeled data')
parser.add_argument('--n-labeled-fear', type=int, default=250,
                        help='Number of labeled fear images')
parser.add_argument('--train-iteration', type=int, default=900,
                        help='Number of iteration per epoch')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--mu', type=int, default=1, help='Scale of unlabeled data')
#Data
parser.add_argument('--train-root', type=str, default='/home/gpu/FER/datasets/RAFdataset/train',
                        help="root path to train data directory")
parser.add_argument('--test-root', type=str, default='/home/gpu/FER/datasets/RAFdataset/test',
                        help="root path to test data directory")
parser.add_argument('--label-train', default='/home/gpu/FER/datasets/RAFdataset/RAF_train_label2.txt', type=str, help='')
parser.add_argument('--label-test', default='/home/gpu/FER/datasets/RAFdataset/RAF_test_label2.txt', type=str, help='')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing RAF-DB')
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomApply([
            transforms.RandomCrop(224, padding=8)
        ], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_labeled_set, train_unlabeled_set, test_set = dataset.get_raf(args.train_root, args.label_train, args.test_root, args.label_test, args.n_labeled, args.n_labeled_fear, transform_train=transform_train, transform_val=transform_val)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size*args.mu, shuffle=True, num_workers=args.num_workers, drop_last=False)
    test_loader = data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=args.num_workers)

    # Model
    print("==> creating ResNet-18")

    def create_model(ema=False):
        model = ResNet_18(num_classes=7)
        model = torch.nn.DataParallel(model).cuda()

        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    criterion = nn.CrossEntropyLoss(reduction='none')
    criterion_simclr = SupConLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)

    # Loggers
    resume_logs = bool(args.resume) and os.path.isfile(args.resume)
    logger = Logger(
        os.path.join(args.out, f'Eacm_log_{args.n_labeled}.txt'),
        title='RAF',
        resume=resume_logs
    )
    logger.set_names([
        'Train Loss',
        'Train Loss X',
        'Train Loss U',
        'Train Loss S_fea',
        'Test Loss',
        'Test Acc.'
    ])

    logger_threshold = Logger(
        os.path.join(args.out, 'log_threshold.txt'),
        title='RAF',
        resume=resume_logs
    )

    logger_threshold.set_names([
        'Epoch', '0', '1', '2', '3', '4', '5', '6'
    ])

    test_accs = []
    threshold = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]

    # Resume from checkpoint
    start_epoch = 1
    resume_path = args.resume

    if os.path.isfile(resume_path):
        print("==> Resuming from:", resume_path)

        checkpoint = torch.load(
            resume_path,
            map_location='cuda',
            weights_only=False
        )

        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        # Restore adaptive thresholds
        if 'threshold' in checkpoint:
            threshold = checkpoint['threshold']
        else:
            # Recover thresholds from the last fully completed epoch
            target_epoch = checkpoint['epoch'] - 1
            threshold_log_path = os.path.join(
                args.out, 'log_threshold.txt'
            )

            with open(threshold_log_path, 'r') as f:
                lines = f.readlines()[1:]  # Skip header

            matching_rows = [
                line.strip().split()
                for line in lines
                if line.strip()
                and int(float(line.strip().split()[0])) == target_epoch
            ]

            if not matching_rows:
                raise RuntimeError(
                    f"Could not find thresholds for epoch {target_epoch}"
                )

            threshold = [
                float(value)
                for value in matching_rows[-1][1:8]
            ]

        print("Restored thresholds:", threshold)

        print("==> Resumed from epoch:", start_epoch)
        print("==> Best accuracy:", best_acc)

    # Train and val
    for epoch in range(start_epoch, args.epochs + 1):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch, args.epochs, state['lr']))

        train_loss, train_loss_x, train_loss_u, train_loss_sim_fea = train(logger_threshold, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, criterion_simclr, threshold, epoch, use_cuda)

        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats')

        # append logger file
        logger.append([train_loss, train_loss_x, train_loss_u, train_loss_sim_fea, test_loss, test_acc])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
                'threshold': [
                    float(t.item()) if torch.is_tensor(t) else float(t)
                    for t in threshold
                ],
            }, is_best)
        test_accs.append(test_acc)
    logger_threshold.close()
    logger.close()

    print('Best acc:')
    print(best_acc)


def train(logger_threshold, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion_ce, criterion_simclr, threshold, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_sim_fea = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    total_unlabeled = 0
    total_accepted = 0
    confidence_sum = 0.0
    confidence_max = 0.0

    class_names = [
        "Surprise", "Fear", "Disgust", "Happiness",
        "Sadness", "Anger", "Neutral"
    ]

    labeled_count = [0] * 7
    labeled_correct = [0] * 7
    labeled_confidence_sum = [0.0] * 7
    correct_confidence_sum = [0.0] * 7

    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x = next(labeled_train_iter)
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x = next(labeled_train_iter)

        try:
            (inputs_u, inputs_u2, inputs_strong), _ = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_strong), _ = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size_u = inputs_u.size(0)


        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            inputs_strong = inputs_strong.cuda()


        # compute guessed labels of unlabel samples
        outputs_u, feature_u = model(inputs_u)
        outputs_u2, feature_u2 = model(inputs_u2)
        p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
        max_probs, max_idx = torch.max(p, dim=1)
        max_idx = max_idx.detach()


        output_x, _ = model(inputs_x)
        Lx = criterion_ce(output_x, targets_x).mean()

        with torch.no_grad():
            probs_x = torch.softmax(output_x, dim=1)
            max_probs_x, predicted_x = torch.max(probs_x, dim=1)

            for i in range(7):
                class_mask = (targets_x == i)
                correct_mask = class_mask & (predicted_x == targets_x)

                labeled_count[i] += int(class_mask.sum().item())
                labeled_correct[i] += int(correct_mask.sum().item())

                labeled_confidence_sum[i] += (
                    max_probs_x[class_mask].sum().item()
                )

                correct_confidence_sum[i] += (
                    max_probs_x[correct_mask].sum().item()
                )

        # update threshold
        threshold = adaptive_threshold_generate(output_x, targets_x, threshold, epoch)
        logger_threshold.append([epoch, threshold[0], threshold[1], threshold[2], threshold[3], threshold[4], threshold[5], threshold[6]])

        mask = mask_generate(max_probs, max_idx, batch_size_u, threshold)
        total_unlabeled += batch_size_u
        total_accepted += int(mask.sum().item())
        confidence_sum += max_probs.sum().item()
        confidence_max = max(confidence_max, max_probs.max().item())
        if batch_idx < 10:
            print(
                f"Pseudo-labels accepted: {int(mask.sum().item())}/{batch_size_u} | "
                f"Max confidence: {max_probs.max().item():.4f} | "
                f"Mean confidence: {max_probs.mean().item():.4f} | "
                f"Thresholds: {[round(float(t.item()) if torch.is_tensor(t) else float(t), 4) for t in threshold]}"
            )

        mask_idx = np.where(mask.cpu() == 0)[0]
        features_prob = torch.cat([feature_u[mask_idx, :].unsqueeze(1), feature_u2[mask_idx, :].unsqueeze(1)], dim=1)
        if features_prob.shape[0] ==0:
            Ls = torch.from_numpy(np.array(0))
        else:
            Ls = criterion_simclr(features_prob)

        output_strong, _ = model(inputs_strong)
        Lu = criterion_ce(output_strong, max_idx) * mask
        Lu = Lu.mean()

        loss = Lx + Lu + Ls * 0.001

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_u.size(0))
        losses_sim_fea.update(Ls.item(), inputs_u.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        if batch_idx < 3:
            print(
                f"\n[DEBUG] Batch {batch_idx + 1}"
                f"\nLogits mean: {output_x.detach().mean().item():.6f}"
                f"\nLogits std: {output_x.detach().std().item():.6f}"
                f"\nClassifier weight grad: {model.module.classifier.weight.grad.norm().item():.6f}"
                f"\nClassifier bias grad: {model.module.classifier.bias.grad.norm().item():.6f}"
            )
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_s_fea: {loss_sim_fea:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    total=bar.elapsed_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_sim_fea=losses_sim_fea.avg,
                    )
        bar.next()
    bar.finish()
    print("\n========== PSEUDO-LABEL SUMMARY ==========")
    print("\n========== PSEUDO-LABEL SUMMARY ==========")
    print(f"Total unlabeled samples: {total_unlabeled}")
    print(f"Accepted pseudo-labels: {total_accepted}")
    print(f"Acceptance rate: {100 * total_accepted / total_unlabeled:.2f}%")
    print(f"Average confidence: {confidence_sum / total_unlabeled:.4f}")
    print(f"Peak confidence: {confidence_max:.4f}")
    print("==========================================\n")

    print("\n========== LABELLED CLASS DIAGNOSTICS ==========")

    for i in range(7):
        count = labeled_count[i]
        correct = labeled_correct[i]

        if count > 0:
            accuracy = 100 * correct / count
            avg_conf = labeled_confidence_sum[i] / count
            correct_conf = (
                correct_confidence_sum[i] / correct
                if correct > 0 else 0.0
            )

            print(
                f"{class_names[i]:10s} | "
                f"Count: {count:5d} | "
                f"Accuracy: {accuracy:6.2f}% | "
                f"Avg confidence: {avg_conf:.4f} | "
                f"Correct confidence: {correct_conf:.4f}"
            )

    print("=================================================\n")


    return (losses.avg, losses_x.avg, losses_u.avg, losses_sim_fea.avg)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets).mean()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Total: {total:} | Loss: {loss:.4f} | Accuracy: {top1: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        total=bar.elapsed_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        )
            bar.next()
        bar.finish()
    return (losses.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint=args.out, filename=f'Eacm_checkpoint_{args.n_labeled}.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'Eacm_model_best_{args.n_labeled}.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param * one_minus_alpha)
                # customized weight decay
                param.mul_(1 - self.wd)

def mask_generate(max_probs, max_idx, batch, threshold):
    mask_ori = torch.zeros(batch, device=max_probs.device)
    for i in range(7):
        idx = np.where(max_idx.cpu() == i)[0]
        m = max_probs[idx].ge(threshold[i]).float()
        for k in range(len(idx)):
            mask_ori[idx[k]]+=m[k]
    return mask_ori.cuda()

def adaptive_threshold_generate(outputs, targets, threshold, epoch):
    probs = torch.softmax(outputs, dim=1)
    max_probs, max_idx = torch.max(probs, dim=1)
    eq_idx = np.where(targets.eq(max_idx).cpu() == 1)[0]

    probs_new = max_probs[eq_idx]
    targets_new = targets[eq_idx]
    for i in range(7):
        idx = np.where(targets_new.cpu() == i)[0]
        if idx.shape[0] != 0:
            prob_mean = probs_new[idx].mean().cpu()
            threshold[i] = (prob_mean + threshold[i])/2 if prob_mean >= 0.8 else threshold[i]
    return threshold

if __name__ == '__main__':
    main()
