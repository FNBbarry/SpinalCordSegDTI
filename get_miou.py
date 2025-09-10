import os

import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

from unet import Unet
from utils.utils_metrics import compute_mIoU, show_results

'''
When conducting index evaluation, the following points should be noted:
1. The generated image in this file is a grayscale image. Due to the relatively small pixel values, it will not display properly in JPG format. Therefore, an image appearing nearly black is considered normal.
2. This file computes the mIoU for the validation set. Currently, the library treats the test set as the validation set and does not maintain a separate test set.
3. Only models trained using VOC format data can utilize this file for mIoU calculation.
'''
if __name__ == "__main__":
    # pixels_numbers = [19,18,16,10,18,19,16,12]# uncomment this below line when apply pixel_number
    thresholds = [0.398, 0.352,0.287, 0.263, 0.395, 0.342, 0.316, 0.251]
    #---------------------------------------------------------------------------#
    #   miou_mode:0 for the whole procedure of miou, including the pred and calculate the miou
    #   miou_mode:1 for the pred
    #   miou_mode:2 for calculate the miou。
    #---------------------------------------------------------------------------#
    miou_mode       = 0
    #------------------------------#
    #   class number +1
    #------------------------------#
    num_classes     = 8#5
    # name_classes    = ["background","aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    # name_classes    = ["_background_","cat","dog"]
    name_classes    = ["0","1","2","3","4","5","6","7","8"]
    #-------------------------------------------------------#
    #   directory path for VOC dataset
    #-------------------------------------------------------#
    VOCdevkit_path  = r'E:\data\W_L_CSM_segmentation\CSM_DTI_10classes_multiply_channel_8classes_png_input'

    image_ids       = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),'r').read().splitlines()
    gt_dir          = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/")
    miou_out_path   = "miou_out"
    pred_dir        = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        unet = Unet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".png")
            # image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".npy")
            image       = Image.open(image_path)
            # image       = np.load(image_path)
            image       = unet.get_miou_png(image)
            # image.save(os.path.join(pred_dir, image_id + ".png"))
            np.save(os.path.join(pred_dir, image_id + ".npy"),image.detach().cpu().numpy())
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        # hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes)  
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes, thresholds)  
        # hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes, name_classes, pixels_numbers)  
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)