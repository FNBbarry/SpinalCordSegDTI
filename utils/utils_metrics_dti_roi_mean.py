import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import pandas as pd


def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   dice
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

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
# def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes,thresholds):
# uncomment this part when you apply pixel_number
def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes,pixels_numbers):
    print('Num classes', num_classes)

    hist = np.zeros((num_classes+1, num_classes+1))

    #------------------------------------------------#
    #   read the path list for label and prediction
    #------------------------------------------------#
    # gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]
    # pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]
    gt_imgs     = [join(gt_dir, x + ".npy") for x in png_name_list]
    pred_imgs   = [join(pred_dir, x + ".npy") for x in png_name_list]

    #------------------------------------------------#
    #   read each image-label pair
    #------------------------------------------------#
    IOU_Recall_Precision = np.zeros((len(gt_imgs),3,num_classes+1))
    all_names = []
    all_subject_roi_metric_mean = []
    for ind in range(len(gt_imgs)):
        # read DTI metrics
        import nibabel as nib
        data_dir = r"E:\data\W_L_CSM_segmentation\DTI"
        infos = pred_imgs[ind].split("\\")[-1].split("_")
        original_img = nib.load(os.path.join(data_dir,infos[0],infos[1],'FA.hdr'))
        data = original_img.get_data()
        data = data.squeeze()
        img = np.rot90(data[:, :, int(infos[2].split(".")[0]) - 1], 1)
        rois_mean_metric = np.zeros((1,num_classes))
        # pred = np.array(Image.open(pred_imgs[ind]))
        pred = np.load(pred_imgs[ind])
        preds = np.zeros((pred.shape[1],pred.shape[2],pred.shape[0]),dtype=np.int)

        # for i,threshold in enumerate(thresholds):
        #     threshold_x,threshold_y = np.where(pred[i,:,:]>threshold)
        #     preds[threshold_x,threshold_y,i]=i+1
        # uncomment this part when you apply pixel_number
        threshold_0_index = np.argwhere(pred[0,:,:]>0.515)
        preds[threshold_0_index[:,0],threshold_0_index[:,1],0]=1
        tmp_a = np.zeros((pred.shape[1],pred.shape[2]))
        tmp_a[threshold_0_index[:,0],threshold_0_index[:,1]]=1
        for i,pixel_number in enumerate(pixels_numbers):
            tmp_pred = np.sort(pred[i+1,:,:].flatten())[::-1]
            for pixel_number_index in range(pixel_number):
                threshold_index = np.argwhere(pred[i+1,:,:]==tmp_pred[pixel_number_index])
                tmp_b = np.zeros((pred.shape[1], pred.shape[2]))
                tmp_b[threshold_index[0,0],threshold_index[0,1]] = 1
                tmp = tmp_a+tmp_b
                if len(np.argwhere(tmp==2))==0:
                    threshold_final_index = threshold_index
                else:
                    threshold_final_index = np.argwhere(tmp==2)
                preds[threshold_final_index[0,0],threshold_final_index[0,1],i+1]=i+1+1

        # calculate the mean DTI value for each class 
        for slice_number in range(num_classes):
            rois_mean_metric[0, slice_number] = np.sum(preds[:, :, slice_number]/(slice_number+1) * img) / np.sum(preds[:, :, slice_number] == slice_number+1)
        all_subject_roi_metric_mean.append(rois_mean_metric)
        all_names.append(pred_imgs[ind].split("\\")[-1].split(".")[0])
        # label = np.array(Image.open(gt_imgs[ind]))
        # label = np.load(gt_imgs[ind])
        # label = np.load(gt_imgs[ind])[:,:,1:]
        label = np.load(gt_imgs[ind])# for 10 classes
        labels = np.zeros_like(label,dtype=np.int)
        for i in range(label.shape[2]):
            threshold_x_1, threshold_y_1 = np.where(label[:, :, i] ==1)
            labels[threshold_x_1, threshold_y_1, i] = i+1

        # if the size of label image and prediction image are not equal, ignore this image.
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        #------------------------------------------------#
        #   calculate the confusion matrix
        #------------------------------------------------#
        # hist += fast_hist(label.flatten(), pred.flatten(),num_classes)
        # hist += fast_hist(labels.flatten(), preds.flatten(),num_classes)
        hist += fast_hist(labels.flatten(), preds.flatten(),num_classes+1)

        hist_simple = fast_hist(labels.flatten(), preds.flatten(),num_classes+1)
        IOU_Recall_Precision[ind,0,:] = 100 * per_class_iu(hist_simple).reshape(1,num_classes+1)
        IOU_Recall_Precision[ind,1,:] = 100 * per_class_PA_Recall(hist_simple).reshape(1,num_classes+1)
        IOU_Recall_Precision[ind,2,:] = 100 * per_class_Precision(hist_simple).reshape(1,num_classes+1)
        # print the mean value of IOU for each class in each 10 images
        if ind > 0 and ind % 10 == 0:  
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                    ind, 
                    len(gt_imgs),
                    100 * np.nanmean(per_class_iu(hist)),
                    100 * np.nanmean(per_class_PA_Recall(hist)),
                    100 * per_Accuracy(hist)
                )
            )
    roi_names_list = ['whole_cord',
                      'left_dorsal_column', 'left_lateral_column', 'left_ventral_column', 'left_grey_matter',
                      'right_dorsal_column', 'right_lateral_column', 'right_ventral_column', 'right_grey_matter']
    # save the mean value of DTI for each class of segmentation mask 
    pd.DataFrame(np.concatenate(all_subject_roi_metric_mean, axis=0), index=all_names,
                 columns=roi_names_list).to_csv("all_subject_dti_metric_predicted_roi_mean.csv")
    #------------------------------------------------#
    #   calculate the mIoU for each image on the val set
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    #------------------------------------------------#
    #   print mIoU on each class
    #------------------------------------------------#
    # for ind_class in range(num_classes):
    for ind_class in range(num_classes+1):
        print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
            + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' + str(round(Precision[ind_class] * 100, 2)))

    #-----------------------------------------------------------------#
    #   calculate the mean vale of IoU on all class and skip the nan
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
            