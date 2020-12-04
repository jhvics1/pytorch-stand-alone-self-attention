"""Run test from the best model checkpoint"""
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from config import get_args
from model import ResNet26, ResNet38, ResNet50
from preprocess import load_data


def main(args, logger):
    train_loader, test_loader = load_data(args)
    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    elif args.dataset == 'IMAGENET':
        num_classes = 1000

    print('img_size: {}, num_classes: {}, stem: {}'.format(args.img_size, num_classes, args.stem))
    if args.model_name == 'ResNet26':
        print('Model Name: {0}'.format(args.model_name))
        model = ResNet26(num_classes=num_classes, stem=args.stem)
    elif args.model_name == 'ResNet38':
        print('Model Name: {0}'.format(args.model_name))
        model = ResNet38(num_classes=num_classes, stem=args.stem)
    elif args.model_name == 'ResNet50':
        print('Model Name: {0}'.format(args.model_name))
        model = ResNet50(num_classes=num_classes, stem=args.stem)

    if args.pretrained_model:
        filename = 'best_model_' + str(args.dataset) + '_' + str(args.model_name) + '_' + str(args.stem) + '_ckpt.tar'
        print('filename :: ', filename)
        file_path = os.path.join('./checkpoint', filename)
        checkpoint = torch.load(file_path)

        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        model_parameters = checkpoint['parameters']
        print('Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, best_acc))
        logger.info('Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, best_acc))
    else:
        start_epoch = 1
        best_acc = 0.0

    if args.cuda:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.cuda()

    eval(model, test_loader, args)


def eval(model, test_loader, args):
    print('evaluation ...')
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Test acc: {0:.2f}'.format(acc))
    return acc


if __name__ == '__main__':
    args, logger = get_args()
    main(args, logger)
