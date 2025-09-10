#-------------------------------------------------------#
#   train-val split for VOC dataset, generate txt file
#-------------------------------------------------------#
import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

#-------------------------------------------------------#
#   trainval_percent : percent for trian and val split
#-------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   directory path for VOC dataset
#-------------------------------------------------------#
VOCdevkit_path      = r'E:\data\W_L_CSM_segmentation\CSM_DTI_9clsses_confidence_map_max_distance_point'#'VOCdevkit'

if __name__ == "__main__":
    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/SegmentationClass')
    saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Segmentation')
    
    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".npy"):
            total_seg.append(seg)

    num     = len(total_seg)  
    list    = range(num)  
    tv      = int(num*trainval_percent)  
    tr      = int(tv*train_percent)  
    trainval= random.sample(list,tv)  
    train   = random.sample(trainval,tr)  
    
    print("train and val size",tv)
    print("traub suze",tr)
    # create txt file
    ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
    ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
    ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
    fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
    
    for i in list:  
        name = total_seg[i][:-4]+'\n'  
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()
    print("Generate txt in ImageSets done.")

    # print("Check datasets format, this may take a while.")
    # classes_nums        = np.zeros([256], np.int)
    # for i in tqdm(list):
    #     name            = total_seg[i]
    #     png_file_name   = os.path.join(segfilepath, name)
    #     if not os.path.exists(png_file_name):
    #         raise ValueError("there is no label image for %s, please chekc the data type"%(png_file_name))
    #
    #     png             = np.array(Image.open(png_file_name), np.uint8)
    #     if len(np.shape(png)) > 2:
    #         print("label image(%s) shape:%s, not binary or RGB please check the dataset"%(name, str(np.shape(png))))
    #
    #     classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)
    #
    # print("the value and count for the pixels")
    # print('-' * 37)
    # print("| %15s | %15s |"%("Key", "Value"))
    # print('-' * 37)
    # for i in range(256):
    #     if classes_nums[i] > 0:
    #         print("| %15s | %15s |"%(str(i), str(classes_nums[i])))
    #         print('-' * 37)
    #
    # if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
    #     print("only 0 and 255 in the label image please check the data")
    # elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
    #     print("only 0 in the label image please check the data")
