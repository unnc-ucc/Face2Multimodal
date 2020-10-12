from __future__ import print_function
import fire
import os
import time
import torch
from torchvision import datasets, transforms
from models import DenseNet
from PIL import Image
import os.path
import numpy as np
import sys
import cv2
#import pickle


from torchvision.datasets.vision import VisionDataset


class faceDataset(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(faceDataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train  # training set or test set
        self.data = []
        self.targets = []

        '''data and labels'''
        imgspaths = []
        txt_file = '../dataset/edatxt/train.txt'
        f = open(txt_file)
        for line in f.readlines():
            filename = line.split(',')[0]
            hr = line.split(',')[1]
            if hr[-1] == '\n':
                hr = hr[:-1]
                imgspaths.append(filename)
                self.targets.append(int(hr))

        print(len(imgspaths))
        self.data = imgspaths

        

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        imgspath, target = self.data[index], self.targets[index]
        img = np.array(Image.open('../dataset/train/'+imgspath))
        img = cv2.resize(img,(224,224))
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
    
    
class facetestDataset(VisionDataset):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):

        super(facetestDataset, self).__init__(root, transform=transform,
                                      target_transform=target_transform)
        self.train = train  # training set or test set
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        '''data and labels'''
        imgspaths = []
        txt_file = '../dataset/edatxt/test.txt'
        f = open(txt_file)
        for line in f.readlines():
            filename = line.split(',')[0]
            hr = line.split(',')[1]
            if hr[-1] == '\n':
                hr = hr[:-1]
                imgspaths.append(filename)
                self.targets.append(int(hr))

        self.data = imgspaths
        
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        imgspath, target = self.data[index], self.targets[index]
        img = np.array(Image.open('../dataset/train/'+imgspath))
        img = cv2.resize(img,(224,224))
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")




class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on train mode
    model.train()

    end = time.time()
    for batch_idx, (input, target) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)

        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val, error.avg),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def test_epoch(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()

    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    'Error %.4f (%.4f)' % (error.val, error.avg),
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def train(model, train_set, valid_set, test_set, save, n_epochs=5,
          batch_size=64, lr=0.1, wd=0.0001, momentum=0.9, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    # Data loaders##########################################################
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()), num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()), num_workers=0)
    if valid_set is None:
        valid_loader = None
    else:
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False,
                                                   pin_memory=(torch.cuda.is_available()), num_workers=0)
    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],
                                                     gamma=0.1)

    # Start log
    with open(os.path.join(save, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error,test_error\n')

    # Train model
    best_error = 1
    for epoch in range(n_epochs):
        scheduler.step()
        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=n_epochs,
        )
        _, valid_loss, valid_error = test_epoch(
            model=model_wrapper,
            loader=valid_loader if valid_loader else test_loader,
            is_test=(not valid_loader)
        )

        # Determine if model is the best
        if valid_loader:
            if valid_error < best_error:
                best_error = valid_error
                print('New best error: %.4f' % best_error)
                torch.save(model.state_dict(), os.path.join(save, 'model.dat'))
                torch.save(model.state_dict(), os.path.join(save, 'mod.pkl'))
        else:
            torch.save(model.state_dict(), os.path.join(save, 'mod.pkl'))

        # Log results
        with open(os.path.join(save, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.6f,%0.5f,%0.5f,\n' % (
                (epoch + 1),
                train_loss,
                train_error,
                valid_loss,
                valid_error,
            ))

    # Final test of model on test set
    #model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()
    test_results = test_epoch(
        model=model,
        loader=test_loader,
        is_test=True
    )
    _, _, test_error = test_results
    with open(os.path.join(save, 'results.csv'), 'a') as f:
        f.write(',,,,,%0.5f\n' % (test_error))
    print('Final test error: %.4f' % test_error)

#############################valid_size
def demo(data='./', save='./output202', depth=100, growth_rate=12, efficient=True, valid_size=None,
         n_epochs=90, batch_size=256, seed=None):
    """
    A demo to show off training of efficient DenseNets.
    Trains and evaluates a DenseNet-BC on CIFAR-10.

    Args:
        data (str) - path to directory where data should be loaded from/downloaded
            (default $DATA_DIR)
        save (str) - path to save the model to (default /tmp)

        depth (int) - depth of the network (number of convolution layers) (default 40)
        growth_rate (int) - number of features added per DenseNet layer (default 12)
        efficient (bool) - use the memory efficient implementation? (default True)

        valid_size (int) - size of validation set
        n_epochs (int) - number of epochs for training (default 300)
        batch_size (int) - size of minibatch (default 256)
        seed (int) - manually set the random seed (default None)
    """

    # Get densenet configuration
    if (depth - 4) % 3:
        raise Exception('Invalid depth')
    block_config = [(depth - 4) // 6 for _ in range(4)]
    
    # Data transforms
    mean = [0.5071, 0.4867, 0.4408]
    stdv = [0.2675, 0.2565, 0.2761]
    train_transforms = transforms.Compose([
        #transforms.RandomCrop(224, padding=4),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Datasets
    train_set = faceDataset(data, train=True, transform=train_transforms, download=False)
    test_set = facetestDataset(data, train=False, transform=test_transforms, download=False)
    
    if valid_size:
        valid_set = faceDataset(data, train=False, transform=test_transforms, download=False)
        indices = torch.randperm(len(train_set))
        train_indices = indices[:len(indices) - valid_size]
        valid_indices = indices[len(indices) - valid_size:]
        train_set = torch.utils.data.Subset(train_set, train_indices)
        valid_set = torch.utils.data.Subset(valid_set, valid_indices)
    else:
        valid_set = None
    
    # Models
    model = DenseNet(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=200,
        small_inputs=False,
        efficient=efficient,
        #drop_rate=0.5
    )
    print(model)

    # Make save directory
    if not os.path.exists(save):
        os.makedirs(save)
    if not os.path.isdir(save):
        raise Exception('%s is not a dir' % save)
    #model.load_state_dict(torch.load(os.path.join(save, 'model.dat')))
    # Train the model
    train(model=model, train_set=train_set, valid_set=valid_set, test_set=test_set, save=save,
          n_epochs=n_epochs, batch_size=batch_size, seed=seed)
    print('Done!')


"""
A demo to show off training of efficient DenseNets.
Trains and evaluates a DenseNet-BC on CIFAR-10.

Try out the efficient DenseNet implementation:
python demo.py --efficient True --data <path_to_data_dir> --save <path_to_save_dir>

Try out the naive DenseNet implementation:
python demo.py --efficient False --data <path_to_data_dir> --save <path_to_save_dir>

Other args:
    --depth (int) - depth of the network (number of convolution layers) (default 40)
    --growth_rate (int) - number of features added per DenseNet layer (default 12)
    --n_epochs (int) - number of epochs for training (default 300)
    --batch_size (int) - size of minibatch (default 256)
    --seed (int) - manually set the random seed (default None)
"""
if __name__ == '__main__':
    fire.Fire(demo)


