import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #  dice
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# W, H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   fast create confution matrix
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

# def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes,thresholds):
# uncomment this line when you applied pixel_number.
# def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes,pixels_numbers):
    print('Num classes', num_classes)
    # hist = np.zeros((num_classes, num_classes))
    hist = np.zeros((num_classes+1, num_classes+1))

    #------------------------------------------------#
    #   get the paths of ground truth and prediction
    #------------------------------------------------#
    # gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]
    # pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]
    gt_imgs     = [join(gt_dir, x + ".npy") for x in png_name_list]
    pred_imgs   = [join(pred_dir, x + ".npy") for x in png_name_list]

    IOU_Recall_Precision = np.zeros((len(gt_imgs),3,num_classes+1))
    series_pred_values = np.zeros((len(gt_imgs),5,num_classes))
    center_points = np.zeros((len(gt_imgs),num_classes,6))
    pred_max_IOU_center_predvalue = np.zeros((len(gt_imgs),6,num_classes))
    # flag_threshold = np.zeros((len(gt_imgs),10,num_classes))
    # for flags_threshold_index in range(10):
    for ind in range(len(gt_imgs)):
        # pred = np.array(Image.open(pred_imgs[ind]))
        pred = np.load(pred_imgs[ind])
        preds = np.zeros((pred.shape[1],pred.shape[2],pred.shape[0]),dtype=np.int)

        for i,threshold in enumerate(thresholds):
            threshold_x,threshold_y = np.where(pred[i,:,:]>threshold)

            center_points[ind,i,4] = np.where(pred[i,:,:]==pred[i,:,:].max())[0]
            center_points[ind,i,5] = np.where(pred[i,:,:]==pred[i,:,:].max())[1]
            if threshold_x.shape[0]==0:
                pred_max_IOU_center_predvalue[ind,5,i] = pred[i,int(center_points[ind,i,5]),int(center_points[ind,i,4])]
                # flag_threshold[ind,flags_threshold_index,i] = pred[i,int(center_points[ind,i,5]),int(center_points[ind,i,4])]>(flags_threshold_index+1)/100
                # flag_threshold[ind,flags_threshold_index,i] = 0
                continue
            preds[threshold_x,threshold_y,i]=i+1

            M = cv2.moments(preds[:,:,i].astype(np.float32))
            # calculate x,y coordinate of center
            center_points[ind,i,0] = int(M["m10"] / M["m00"])
            center_points[ind,i,1] = int(M["m01"] / M["m00"])
            pred_max_IOU_center_predvalue[ind,5,i] = pred[i,int(center_points[ind,i,1]),int(center_points[ind,i,0])]
            # flag_threshold[ind,flags_threshold_index,i] = pred[i,int(center_points[ind,i,1]),int(center_points[ind,i,0])]>(flags_threshold_index+1)/10

        for j in range(8):
            tmp_pred_mask = pred[j,:,:]*(preds[:,:,j]/(j+1))
            tmp_series = tmp_pred_mask[tmp_pred_mask!=0]
            if tmp_series.shape[0]==0:
                continue
            series_pred_values[ind,2,j] = tmp_series.min()
            series_pred_values[ind,3,j] = tmp_series.max()

        # uncomment this line when you applied pixel_number.
        # for i,pixel_number in enumerate(pixels_numbers):
        #     tmp_pred = np.sort(pred[i,:,:].flatten())[::-1]
        #     for pixel_number_index in range(pixel_number):
        #         threshold_index = np.argwhere(pred[i,:,:]==tmp_pred[pixel_number_index])
        #         preds[threshold_index[0,0],threshold_index[0,1],i]=i+1
        # label = np.array(Image.open(gt_imgs[ind]))
        # label = np.load(gt_imgs[ind])[:,:,1:]#for nine classes
        label = np.load(gt_imgs[ind])#for ten classes
        labels = np.zeros_like(label,dtype=np.int)
        for i in range(label.shape[2]):
            threshold_x_1, threshold_y_1 = np.where(label[:, :, i] ==1)
            labels[threshold_x_1, threshold_y_1, i] = i+1

            M = cv2.moments(labels[:,:,i].astype(np.float32))
            # calculate x,y coordinate of center
            center_points[ind,i,2] = int(M["m10"] / M["m00"])
            center_points[ind,i,3] = int(M["m01"] / M["m00"])
            pred_max_IOU_center_predvalue[ind,0,i] = center_points[ind,i,0]-center_points[ind,i,2]
            pred_max_IOU_center_predvalue[ind,1,i] = center_points[ind,i,1]-center_points[ind,i,3]
            pred_max_IOU_center_predvalue[ind,2,i] = center_points[ind,i,4]-center_points[ind,i,2]
            pred_max_IOU_center_predvalue[ind,3,i] = center_points[ind,i,5]-center_points[ind,i,3]
        
        for j in range(8):
            tmp_label_mask = pred[j,:,:]*label[:,:,j]
            tmp_series = tmp_label_mask[tmp_label_mask!=0]
            series_pred_values[ind,0,j] = tmp_series.min()
            series_pred_values[ind,1,j] = tmp_series.max()
            series_pred_values[ind,4,j] = pred[j,:,:].max()

        # if the size of prediction and label are not the same, then skip this image
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   create the confusion matrix and accumulate
        #------------------------------------------------#
        # hist += fast_hist(label.flatten(), pred.flatten(),num_classes)
        # hist += fast_hist(labels.flatten(), preds.flatten(),num_classes)
        hist += fast_hist(labels.flatten(), preds.flatten(),num_classes+1)

        hist_simple = fast_hist(labels.flatten(), preds.flatten(),num_classes+1)

        pred_max_IOU_center_predvalue[ind,4,:] = 100 * per_class_iu(hist_simple).reshape(1,num_classes+1)[0][1:]

        IOU_Recall_Precision[ind,0,:] = 100 * per_class_iu(hist_simple).reshape(1,num_classes+1)
        IOU_Recall_Precision[ind,1,:] = 100 * per_class_PA_Recall(hist_simple).reshape(1,num_classes+1)
        IOU_Recall_Precision[ind,2,:] = 100 * per_class_Precision(hist_simple).reshape(1,num_classes+1)
        # calculate the mean IOU all classes for each 10 images
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    # new_iou = np.expand_dims(pred_max_IOU_center_predvalue[:,4,:],axis=1)*flag_threshold
    # flag_new_iou = new_iou.sum(axis=0)/flag_threshold.sum(axis=0)
    #------------------------------------------------#
    #   calculate the IOU for each class for each validation images
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    #------------------------------------------------#
    #   print the IoU of each class
    #------------------------------------------------#
    # for ind_class in range(num_classes):
    for ind_class in range(num_classes+1):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
            + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   calculate the mean IoU accross all classes in validation images, ignoring NaN values
    #-----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '; mPA: ' + str(round(np.nanmean(PA_Recall) * 100, 2)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)))  
    return np.array(hist, np.int), IoUs, PA_Recall, Precision

def adjust_axes(r, t, fig, axes):
    bb                  = t.get_window_extent(renderer=r)
    text_width_inches   = bb.width / fig.dpi
    current_fig_width   = fig.get_figwidth()
    new_fig_width       = current_fig_width + text_width_inches
    propotion           = new_fig_width / current_fig_width
    x_lim               = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    fig     = plt.gcf() 
    axes    = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val) 
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()

def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    # draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
    #     os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            