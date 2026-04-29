import os
import argparse
import datetime
import time
import csv
import pandas as pd
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from models import gan
from models.models import classifier32, classifier32ABN, classifierMLP, classifier1D32, classifierTPBranch
from datasets.osr_dataloader import MNIST_OSR, CIFAR10_OSR, CIFAR100_OSR, SVHN_OSR, Tiny_ImageNet_OSR, UAVCAN_OSR, UAVCANBit_OSR
from utils import Logger, save_networks, load_networks
from core import train, train_cs, test

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='mnist', help="mnist | svhn | cifar10 | cifar100 | tiny_imagenet | can.csv | can_bit.csv")
parser.add_argument('--dataroot', type=str, default='./data')
parser.add_argument('--outf', type=str, default='./log')
parser.add_argument('--out-num', type=int, default=50, help='For CIFAR100')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--gan_lr', type=float, default=0.0002, help="learning rate for gan")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1.0, help="temp")
parser.add_argument('--num-centers', type=int, default=1)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifierMLP')

# misc
parser.add_argument('--nz', type=int, default=100)
parser.add_argument('--ns', type=int, default=1)
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='../log')
parser.add_argument('--loss', type=str, default='ARPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def append_epoch_log(txt_path, summary):
    with open(txt_path, 'a') as f:
        f.write(summary + '\n')


def append_epoch_csv(csv_path, row):
    df = pd.DataFrame([row])
    header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode='a', header=header, index=False)


def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'mnist' in options['dataset']:
        Data = MNIST_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'cifar100' in options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader = Data.train_loader, Data.test_loader
        out_Data = CIFAR100_OSR(known=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        outloader = out_Data.test_loader
    elif options['dataset'] == 'can.csv':
        if options['model'] == 'classifierTPBranch':
            Data = UAVCANBit_OSR(
                known=options['known'],
                dataroot=options['dataroot'],
                use_gpu=use_gpu,
                num_workers=8,
                batch_size=options['batch_size'],
            )
        else:
            Data = UAVCAN_OSR(
                known=options['known'],
                dataroot=options['dataroot'],
                use_gpu=use_gpu,
                num_workers=8,
                batch_size=options['batch_size'],
            )
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif options['dataset'] == 'can_bit.csv':
        Data = UAVCANBit_OSR(
            known=options['known'],
            dataroot=options['dataroot'],
            use_gpu=use_gpu,
            num_workers=8,
            batch_size=options['batch_size'],
        )
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    else:
        Data = Tiny_ImageNet_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    
    options['num_classes'] = Data.num_classes

    # Model
    if options['dataset'] in ['can.csv', 'can_bit.csv']:
        if options['cs']:
            raise ValueError("dataset '{}' does not support --cs currently.".format(options['dataset']))
        if options['model'] == 'classifierMLP':
            dummy_data, _ = next(iter(trainloader))
            input_dim = dummy_data.shape[1]
            net = classifierMLP(num_classes=options['num_classes'], input_dim=input_dim)
        elif options['model'] == 'classifier1D32':
            net = classifier1D32(num_classes=options['num_classes'])
        elif options['model'] == 'classifierTPBranch':
            net = classifierTPBranch(num_classes=options['num_classes'])
        else:
            raise ValueError("Unsupported model '{}' for {}. Use 'classifierMLP', 'classifier1D32' or 'classifierTPBranch'.".format(options['model'], options['dataset']))
    else:
        if options['cs']:
            options['model'] = 'classifier32ABN'
            net = classifier32ABN(num_classes=options['num_classes'])
        else:
            options['model'] = 'classifier32'
            net = classifier32(num_classes=options['num_classes'])
    print("Creating model: {}".format(options['model']))
    feat_dim = 128

    if options['cs']:
        print("Creating GAN")
        nz, ns = options['nz'], 1
        if 'tiny_imagenet' in options['dataset']:
            netG = gan.Generator(1, nz, 64, 3)
            netD = gan.Discriminator(1, 3, 64)
        else:
            netG = gan.Generator32(1, nz, 64, 3)
            netD = gan.Discriminator32(1, 3, 64)
        fixed_noise = torch.FloatTensor(64, nz, 1, 1).normal_(0, 1)
        criterionD = nn.BCELoss()

    # Loss
    options.update(
        {
            'feat_dim': feat_dim,
            'use_gpu':  use_gpu
        }
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()
        if options['cs']:
            netG = nn.DataParallel(netG, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            netD = nn.DataParallel(netD, device_ids=[i for i in range(len(options['gpu'].split(',')))]).cuda()
            fixed_noise.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    ensure_dir(model_path)

    if options['dataset'] == 'cifar100':
        model_path += '_50'
        file_name = '{}_{}_{}_{}_{}'.format(options['model'], options['loss'], 50, options['item'], options['cs'])
    else:
        file_name = '{}_{}_{}_{}'.format(options['model'], options['loss'], options['item'], options['cs'])

    log_root = os.path.join(options['outf'], 'epoch_logs', options['dataset'])
    ensure_dir(log_root)
    epoch_txt_path = os.path.join(log_root, file_name + '.txt')
    epoch_csv_path = os.path.join(log_root, file_name + '.csv')

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]
    
    if options['dataset'] == 'tiny_imagenet':
        optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    else:
        optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)
    if options['cs']:
        optimizerD = torch.optim.Adam(netD.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))
        optimizerG = torch.optim.Adam(netG.parameters(), lr=options['gan_lr'], betas=(0.5, 0.999))

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,60,90,120])

    start_time = time.time()
    best_results = None
    best_epoch = -1
    best_oscr = float('-inf')
    last_results = None

    append_epoch_log(epoch_txt_path, 'start dataset={} model={} loss={} known={} unknown={} seed={}'.format(
        options['dataset'], options['model'], options['loss'], options['known'], options['unknown'], options['seed']))

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        if options['cs']:
            train_loss = train_cs(net, netD, netG, criterion, criterionD,
                optimizer, optimizerD, optimizerG,
                trainloader, epoch=epoch, **options)
        else:
            train_loss = train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            last_results = dict(results)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

            current_lr = optimizer.param_groups[0]['lr']
            epoch_row = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'ACC': results['ACC'],
                'AUROC': results['AUROC'],
                'OSCR': results['OSCR'],
                'TNR': results['TNR'],
                'AUIN': results['AUIN'],
                'AUOUT': results['AUOUT'],
                'lr': current_lr,
                'known': str(options['known']),
                'unknown': str(options['unknown']),
                'seed': options['seed'],
                'is_best_oscr': int(results['OSCR'] > best_oscr),
            }
            append_epoch_csv(epoch_csv_path, epoch_row)

            summary = (
                'epoch={epoch} train_loss={train_loss:.6f} ACC={ACC:.3f} AUROC={AUROC:.3f} '
                'OSCR={OSCR:.3f} TNR={TNR:.3f} AUIN={AUIN:.3f} AUOUT={AUOUT:.3f} '
                'lr={lr:.6f} known={known} unknown={unknown}'
            ).format(
                epoch=epoch + 1,
                train_loss=train_loss,
                ACC=results['ACC'],
                AUROC=results['AUROC'],
                OSCR=results['OSCR'],
                TNR=results['TNR'],
                AUIN=results['AUIN'],
                AUOUT=results['AUOUT'],
                lr=current_lr,
                known=options['known'],
                unknown=options['unknown'],
            )
            append_epoch_log(epoch_txt_path, summary)

            save_networks(net, model_path, file_name, criterion=criterion)

            if results['OSCR'] > best_oscr:
                best_oscr = results['OSCR']
                best_epoch = epoch + 1
                best_results = dict(results)
                best_name = file_name + '_best_oscr'
                save_networks(net, model_path, best_name, criterion=criterion)
                append_epoch_log(epoch_txt_path, 'best_oscr_update epoch={} OSCR={:.3f}'.format(best_epoch, best_oscr))

        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    if best_results is not None:
        append_epoch_log(epoch_txt_path, 'finished best_epoch={} best_OSCR={:.3f}'.format(best_epoch, best_oscr))
    if last_results is not None:
        append_epoch_log(epoch_txt_path, 'finished last_epoch={} last_OSCR={:.3f}'.format(options['max_epoch'], last_results['OSCR']))

    final_results = dict(last_results if last_results is not None else {})
    if best_results is not None:
        final_results['best_epoch'] = best_epoch
        final_results['best_ACC'] = best_results['ACC']
        final_results['best_AUROC'] = best_results['AUROC']
        final_results['best_OSCR'] = best_results['OSCR']
        final_results['best_TNR'] = best_results['TNR']
        final_results['best_AUIN'] = best_results['AUIN']
        final_results['best_AUOUT'] = best_results['AUOUT']
    if last_results is not None:
        final_results['last_epoch'] = options['max_epoch']
        final_results['last_ACC'] = last_results['ACC']
        final_results['last_AUROC'] = last_results['AUROC']
        final_results['last_OSCR'] = last_results['OSCR']
        final_results['last_TNR'] = last_results['TNR']
        final_results['last_AUIN'] = last_results['AUIN']
        final_results['last_AUOUT'] = last_results['AUOUT']

    return final_results

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)

    dataset_dir = options['dataset']
    if options['dataset'] == 'can_bit.csv':
        dataset_dir = 'can.csv'

    options['dataroot'] = os.path.join(options['dataroot'], dataset_dir)
    img_size = 32
    results = dict()
    
    from split import splits_2020 as splits
    
    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
        if options['dataset'] == 'cifar100':
            unknown = splits[options['dataset']+'-'+str(options['out_num'])][len(splits[options['dataset']])-i-1]
        elif options['dataset'] == 'tiny_imagenet':
            img_size = 64
            options['lr'] = 0.001
            unknown = list(set(list(range(0, 200))) - set(known))
        elif options['dataset'] == 'can.csv':
            unknown = list(set([0, 1, 2, 3]) - set(known))
        else:
            unknown = list(set(list(range(0, 10))) - set(known))

        options.update(
            {
                'item':     i,
                'known':    known,
                'unknown':  unknown,
                'img_size': img_size
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if options['dataset'] == 'cifar100':
            file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
        else:
            file_name = options['dataset'] + '.csv'

        res = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))