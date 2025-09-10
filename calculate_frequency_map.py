import os
import numpy as np
import cv2

VOCdevkit_path = r'E:\data\W_L_CSM_segmentation\CSM_DTI_10classes_multiply_channel'#'VOCdevkit'
with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),"r") as f:
    train_lines = f.readlines()

# w = [31,11,7,13,7,11,7,13,7]
w =   [15, 5,3, 6,3, 5,3, 6,3]
# h = [17,7,11,9,5,7,11,9,3]
h =   [8, 3, 5,4,2,3, 5,4,1]
thresholds = [0.399, 0.334,0.389, 0.339, 0.314, 0.376, 0.339, 0.321, 0.302]
class_frequency_map = np.zeros((9,len(train_lines),128,128))
for i,file in enumerate(train_lines):
    name = file.split()[0]
    file_path = os.path.join(r"E:\data\W_L_CSM_segmentation\20220902\miou_out\detection-results", name + ".npy")
    single_label_file = np.load(file_path)
    preds = np.zeros((single_label_file.shape[1],single_label_file.shape[2],single_label_file.shape[0]),dtype=np.float32)
    for j,threshold in enumerate(thresholds):
        threshold_x,threshold_y = np.where(single_label_file[j,:,:]>threshold)
        if threshold_x.shape[0]==0:
            continue
        preds[threshold_x,threshold_y,j]=1
        # calculate moments of binary image
        M = cv2.moments(preds[:,:,j])

        # calculate x,y coordinate of center
        x_point = int(M["m10"] / M["m00"])
        y_point = int(M["m01"] / M["m00"])
        class_frequency_map[j,i,64-h[j]:64+h[j]+1,64-w[j]:64+w[j]+1] = single_label_file[j,y_point-h[j]:y_point+h[j]+1,x_point-w[j]:x_point+w[j]+1]
        
class_frequency_map_mean = class_frequency_map.mean(axis=1)
np.save(os.path.join(r"E:\data\W_L_CSM_segmentation\20220902", 'class_frequency_map_pred.npy'), class_frequency_map_mean)

