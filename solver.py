import torch
from collections import OrderedDict
from torch.nn import utils, functional as F
from torch.optim import Adam
from torch.autograd import Variable
from torch.backends import cudnn
from networks.poolnet import build_model, weights_init
import scipy.misc as sm
import numpy as np
import os
import torchvision.utils as vutils
import cv2
import math
import time
import csv

class Solver(object):
    def __init__(self, train_loader, test_loader, config):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.iter_size = config.iter_size
        self.show_every = config.show_every
        self.lr_decay_epoch = [15,]
        self.build_model()
        if config.mode == 'test':
            print('Loading pre-trained model from %s...' % self.config.model)
            if self.config.cuda:
                self.net.load_state_dict(torch.load(self.config.model))
            else:
                self.net.load_state_dict(torch.load(self.config.model, map_location='cpu'))
            self.net.eval()

    # print the network information and parameter numbers
    def print_network(self, model, name):
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    # build the network
    def build_model(self):
        self.net = build_model(self.config.arch)
        if self.config.cuda:
            self.net = self.net.cuda()
        # self.net.train()
        self.net.eval()  # use_global_stats = True
        self.net.apply(weights_init)
        if self.config.load == '':
            self.net.base.load_pretrained_model(torch.load(self.config.pretrained_model))
        else:
            self.net.load_state_dict(torch.load(self.config.load))

        self.lr = self.config.lr
        self.wd = self.config.wd

        self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)
        self.print_network(self.net, 'PoolNet Structure')

    def get_sub_imgs_from_bboxes(self, img, img_name, annotation_path):
        csv_file = open(annotation_path)
        csv_reader = csv.reader(csv_file, delimiter=',')
        imgs = []
        positions = []
        # TODO: Print bboxes to check if are well read
        img_np = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() + np.array((104.00699, 116.66877, 122.67892))).astype(np.uint8).copy()
        h, w, c = img_np.shape

        for row in csv_reader:
            if row[0] == img_name:
                tol = 0.02
                x1 = max(int(float(row[1]) * (1.0 - tol)), 0)
                y1 = max(int(float(row[2]) * (1.0 - tol)), 0)
                x2 = min(int(float(row[3]) * (1.0 + tol)), w)
                y2 = min(int(float(row[4]) * (1.0 + tol)), h)
                imgs.append(img[:, :, y1:y2, x1:x2])
                positions.append([x1, y1, x2, y2])

                # Print boundinb box
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return imgs, positions

    def test(self):
        mode_name = 'sal_fuse'
        time_s = time.time()
        img_num = len(self.test_loader)
        for i, data_batch in enumerate(self.test_loader):
            images, name, im_size = data_batch['image'], data_batch['name'][0], np.asarray(data_batch['size'])
            # Get annotation file
            annotation_file = self.config.test_root.replace('images/', 'annotations/annotations_test.csv')

            subimages, positions = self.get_sub_imgs_from_bboxes(images, name, annotation_file)

            images_np = (images.squeeze(0).permute(1, 2, 0).cpu().numpy() + np.array(
                (104.00699, 116.66877, 122.67892))).astype(np.uint8).copy()

            for idx, img in enumerate(subimages):

                with torch.no_grad():
                    img = Variable(img)
                    if self.config.cuda:
                        img = img.cuda()
                    # Workaround
                    print(img.shape)
                    img = torch.nn.functional.interpolate(img, size=512)
                    print(img.shape)
                    img = img.permute(0, 1, 3, 2)
                    img = torch.nn.functional.interpolate(img, size=512)
                    img = img.permute(0, 1, 3, 2)

                    preds = self.net(img)
                    pred = np.squeeze(torch.sigmoid(preds).cpu().data.numpy())
                    multi_fuse = 255 * pred
                    # cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '.png'), multi_fuse)
                    masks = torch.nn.functional.interpolate(preds.squeeze(0).sigmoid().permute(1, 2, 0), 3)
                    # results = np.concatenate(((masks * 255).cpu().numpy(), img.squeeze(0).permute(1, 2, 0).cpu().numpy()), axis=0)
                    # cv2.imwrite(os.path.join(self.config.test_fold, name[:-4] + '_' + mode_name + '_' + str(idx) + '_img.png'),
                    #             results)

                    masks_np = ((masks * 255).cpu().numpy().astype(np.uint8)).copy()
                    cv2.normalize(masks_np, masks_np, 0, 255, cv2.NORM_MINMAX)
                    ret, masks_np = cv2.threshold(masks_np, 2, 255, cv2.THRESH_BINARY)

                    # images_np[positions[idx][1]:positions[idx][3], positions[idx][0]:positions[idx][2], 2]\
                    #     = cv2.addWeighted(images_np[positions[idx][1]:positions[idx][3], positions[idx][0]:positions[idx][2], 2], 0.4,
                    #                 masks_np[:, : , 1], 0.7, 0.0)

                    #
                    cv2.imshow("img",
                               (img.squeeze(0).permute(1, 2, 0).cpu().numpy() + np.array((104.00699, 116.66877, 122.67892))).astype(np.uint8).copy())
                    cv2.imshow("mask",
                               masks_np)
                    cv2.imshow("imgmask", images_np[positions[idx][1]:positions[idx][3], positions[idx][0]:positions[idx][2], :])
                    cv2.waitKey(0)

            cv2.imwrite(os.path.join(self.config.test_fold, name + 'full_img.png'),
                        images_np)
            input()

        time_e = time.time()
        print('Speed: %f FPS' % (img_num/(time_e-time_s)))
        print('Test Done!')

    # training phase
    def train(self):
        iter_num = len(self.train_loader.dataset) // self.config.batch_size
        aveGrad = 0
        for epoch in range(self.config.epoch):
            r_sal_loss= 0
            self.net.zero_grad()
            for i, data_batch in enumerate(self.train_loader):
                sal_image, sal_label = data_batch['sal_image'], data_batch['sal_label']
                if (sal_image.size(2) != sal_label.size(2)) or (sal_image.size(3) != sal_label.size(3)):
                    print('IMAGE ERROR, PASSING```')
                    continue
                sal_image, sal_label= Variable(sal_image), Variable(sal_label)
                if self.config.cuda:
                    # cudnn.benchmark = True
                    sal_image, sal_label = sal_image.cuda(), sal_label.cuda()

                sal_pred = self.net(sal_image)
                sal_loss_fuse = F.binary_cross_entropy_with_logits(sal_pred, sal_label, reduction='sum')
                sal_loss = sal_loss_fuse / (self.iter_size * self.config.batch_size)
                r_sal_loss += sal_loss.data

                sal_loss.backward()

                aveGrad += 1

                # accumulate gradients as done in DSS
                if aveGrad % self.iter_size == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    aveGrad = 0

                if i % (self.show_every // self.config.batch_size) == 0:
                    if i == 0:
                        x_showEvery = 1
                    print('epoch: [%2d/%2d], iter: [%5d/%5d]  ||  Sal : %10.4f' % (
                        epoch, self.config.epoch, i, iter_num, r_sal_loss/x_showEvery))
                    print('Learning rate: ' + str(self.lr))
                    r_sal_loss= 0

            if (epoch + 1) % self.config.epoch_save == 0:
                torch.save(self.net.state_dict(), '%s/models/epoch_%d.pth' % (self.config.save_folder, epoch + 1))

            if epoch in self.lr_decay_epoch:
                self.lr = self.lr * 0.1
                self.optimizer = Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.lr, weight_decay=self.wd)

        torch.save(self.net.state_dict(), '%s/models/final.pth' % self.config.save_folder)

def bce2d(input, target, reduction=None):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)

