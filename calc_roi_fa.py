import os
import numpy as np
import pandas as pd
import cv2

VOCdevkit_path = r'E:\data\W_L_CSM_segmentation\CSM_DTI_10classes_multiply_channel_8classes_png_input'#'VOCdevkit'
with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/test.txt"),"r") as f:
    train_lines = f.readlines()

thresholds = [0.398, 0.352,0.287, 0.263, 0.395, 0.342, 0.316, 0.251]
predicted_fa_metric = np.zeros((len(train_lines),len(thresholds)))
real_fa_metric = np.zeros((len(train_lines),len(thresholds)))
center_predvalue = np.zeros((len(train_lines),len(thresholds)))
all_names = []
for i,file in enumerate(train_lines):
    name = file.split()[0]
    predict_file_path = os.path.join(r"E:\data\W_L_CSM_segmentation\20221110\miou_out\test\detection-results", name + ".npy")
    real_file_path = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass/", name + ".npy")
    dti_file_path = os.path.join(r"E:\data\W_L_CSM_segmentation\CSM_DTI_10classes_multiply_channel_8classes\VOC2007\JPEGImages", name + ".npy")
    predicted_file = np.load(predict_file_path)
    dti_file = np.load(dti_file_path)
    real_file = np.load(real_file_path)

    preds = np.zeros((predicted_file.shape[1],predicted_file.shape[2],predicted_file.shape[0]),dtype=np.int)
    for j,threshold in enumerate(thresholds):
        threshold_x,threshold_y = np.where(predicted_file[j,:,:]>threshold)
        if threshold_x.shape[0]==0:
            continue
        preds[threshold_x,threshold_y,j]=1

        M = cv2.moments(preds[:,:,j].astype(np.float32))
        # calculate x,y coordinate of center
        center_points_0 = int(M["m10"] / M["m00"])
        center_points_1 = int(M["m01"] / M["m00"])
        center_predvalue[i,j] = predicted_file[j,int(center_points_1),int(center_points_0)]
    
    predicted_fa = preds*np.expand_dims(dti_file,axis=2)
    predicted_fa_metric[i,:] = predicted_fa.sum(axis=0).sum(axis=0)/preds.sum(axis=0).sum(axis=0)

    real_fa = real_file*np.expand_dims(dti_file,axis=2)
    real_fa_metric[i,:] = real_fa.sum(axis=0).sum(axis=0)/real_file.sum(axis=0).sum(axis=0)
    all_names.append(name)

pd.DataFrame(predicted_fa_metric,index=all_names).to_csv("predicted_fa.csv")
pd.DataFrame(real_fa_metric,index=all_names).to_csv("real_fa.csv")
pd.DataFrame(center_predvalue,index=all_names).to_csv("center_predvalue.csv")