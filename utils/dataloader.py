import os

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class UnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path):
        super(UnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   read image from the file
        #-------------------------------#
        jpg         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".png"))
        # png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".npy"))
        # jpg         = np.load(os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + ".npy"))
        png         = np.load(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".npy"))
        #-------------------------------#s
        #   data augmentation
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)
        if self.train:
            jpg = np.expand_dims(jpg, axis=0)
        else:
            jpg = np.expand_dims(jpg, axis=0)[:,:,:,0]
        # jpg         = np.transpose(preprocess_input(np.array(jpg, np.float64)), [2,0,1])
        # png         = np.array(png)
        png         = [np.array(png[i])for i in range(len(png))]
        return jpg, png

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        image   = cvtColor(image)
        # image = Image.fromarray(image)
        # label   = Image.fromarray(np.array(label))
        labels = []
        for i in range(label.shape[2]):
            labels.append(Image.fromarray(label[:,:,i]))
        #------------------------------#
        #  acquire the width and height of the image
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            iw, ih  = image.size
            # scale   = min(w/iw, h/ih)
            # nw      = int(iw*scale)
            # nh      = int(ih*scale)
            #
            # image       = image.resize((nw,nh), Image.BICUBIC)
            # new_image   = Image.new('RGB', [w, h], (128,128,128))
            # new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            #
            # label       = label.resize((nw,nh), Image.NEAREST)
            # new_label   = Image.new('L', [w, h], (0))
            # new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            new_image = image
            new_label = np.stack([label[:,:,i] for i in range(len(labels))],axis=0)
            return new_image, new_label

        #------------------------------------------#
        #   scale the image on width and height
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        # label = label.resize((nw,nh), Image.NEAREST)
        labels = [label.resize((nw,nh), Image.NEAREST) for label in labels]

        #------------------------------------------#
        #   flip
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            # label = label.transpose(Image.FLIP_LEFT_RIGHT)
            labels = [label.resize((nw, nh), Image.NEAREST) for label in labels]
        
        #------------------------------------------#
        #   Add gray bars to the excess parts of the image
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        # new_image = Image.new('RGB', (w,h), (128,128,128))
        # new_label = Image.new('L', (w,h), (0))
        # new_image.paste(image, (dx, dy))
        # new_label.paste(label, (dx, dy))
        new_image = Image.new('F', (w,h), (0))
        new_image.paste(image, (dx, dy))
        new_labels = []
        for label in labels:
            new_label = Image.new('F', (w, h), (0))
            new_label.paste(label, (dx, dy))
            new_labels.append(new_label)
        image = new_image
        # label = new_label
        label = new_labels

        image_data      = image
        # image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   transformation on HSV
        #---------------------------------#
        # r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        # dtype           = image_data.dtype
        # x       = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * r[0]) % 180).astype(dtype)
        # lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        #
        # image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label

def unet_dataset_collate(batch):
    images      = []
    pngs        = []
    # seg_labels  = []
    # for img, png, labels in batch:
    #     images.append(img)
    #     pngs.append(png)
    #     seg_labels.append(labels)
    for img, png in batch:
        images.append(np.array(img))
        pngs.append(np.stack(png,axis=0))
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # pngs        = torch.from_numpy(np.array(pngs)).long()
    pngs        = torch.from_numpy(np.stack(pngs,axis=0)).float()
    # seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    # return images, pngs, seg_labels
    return images, pngs
