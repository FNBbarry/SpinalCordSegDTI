import os
import os.path as osp
import nibabel as nib
from roifile import ImagejRoi
import numpy as np
import pandas as pd

from labelme import utils

path = r'E:\data\W_L_CSM_segmentation\DTI'
subject_dir = os.listdir(path)

file_folders = ['B0', 'E0', 'E1', 'FA']
roi_names_list = ['whole_cord',
                  'left_dorsal_column', 'left_lateral_column', 'left_ventral_column', 'left_grey_matter',
                  'right_dorsal_column', 'right_lateral_column', 'right_ventral_column', 'right_grey_matter']
all_names = []
all_subject_roi_metric_mean = []
file_folders_index = 3
for subject in subject_dir:
    path_subject = os.path.join(path, subject)
    xlsx_file_name = os.path.join(path_subject, "roi_name_list.xlsx")
    for file_folder in os.listdir(path_subject):
        individual_folders = os.path.join(path_subject, file_folder)
        if os.path.isdir(individual_folders):
            roi_file_label = pd.read_excel(xlsx_file_name, index_col=0, sheet_name=file_folder)
            if file_folder == '1' or file_folder == '2' or file_folder == '3':
                roi_list = {}
                individual = 'ROI_analysis'
                for file in os.listdir(os.path.join(individual_folders, individual)):
                    if os.path.join(individual_folders, individual, file).split('.')[-1] == 'zip':
                        slice_number = os.path.join(individual_folders, individual, file).split('.')[0].split('-')[-1]
                        roi_list[slice_number] = ImagejRoi.fromfile(os.path.join(individual_folders, individual, file))
                        # roi_list.coordinates()
                for individual in os.listdir(individual_folders):
                    if individual == file_folders[file_folders_index] + '.hdr':
                        original_img = nib.load(os.path.join(individual_folders, individual))
                        data = original_img.get_data()
                        data = data.squeeze()
                        for key in roi_list.keys():
                            # data[:, :, int(key)] = np.fliplr(data[:, :, int(key)])
                            img = np.rot90(data[:, :, int(key) - 1], 1)
                            # img_binary = np.rot90(data[:, :, int(key)-1],1)
                            # img = img_binary[:,:,np.newaxis]
                            # img = np.repeat(img,3,axis=2)
                            # each slice has 9 rois, the roi with label 1 includes the whole cord, and the others are sub-regions
                            multiply_channels_classes = np.zeros((data.shape[0], data.shape[1], len(roi_names_list)))
                            # multiply_channels_classes[:,:,0] = np.ones((data.shape[0], data.shape[1]))
                            rois_mean_metric = np.zeros((1, len(roi_names_list)))
                            out_dir = subject + '_' + file_folder + '_' + key

                            for i in range(len(roi_list[key])):
                                points_data = []
                                # if roi_file_label["class_label_"+key][i] == 1:
                                #     continue
                                roi_points_data = {}
                                roi_points_data['points'] = [coord for coord in roi_list[key][i].coordinates()]
                                roi_points_data['label'] = roi_names_list[roi_file_label["class_label_" + key][i] - 1]
                                roi_points_data['shape_type'] = 'polygon'
                                points_data.append(roi_points_data)
                                label_name_to_value = {'background': 0}
                                for shape in points_data:  # data['shapes'] has name for each class and its coordnates
                                    label_name = shape['label']
                                    if label_name in label_name_to_value:
                                        label_value = label_name_to_value[label_name]
                                    else:
                                        label_value = len(label_name_to_value)
                                        label_name_to_value[label_name] = label_value

                                # label_values must be dense
                                label_values, label_names = [], []
                                for ln, lv in sorted(label_name_to_value.items(), key=lambda x: x[1]):
                                    label_values.append(lv)
                                    label_names.append(ln)
                                assert label_values == list(range(len(label_values)))

                                lbl = utils.shapes_to_label(img.shape, points_data, label_name_to_value)

                                points = np.stack(np.where(lbl == 1), axis=1)
                                for ind_x, ind_y in points:
                                    multiply_channels_classes[ind_x, ind_y, roi_file_label["class_label_" + key][i] - 1] = 1
                                rois_mean_metric[0, roi_file_label["class_label_" + key][i] - 1] = \
                                    np.sum(multiply_channels_classes[:, :, roi_file_label["class_label_" + key][i] - 1]*img)/np.sum(multiply_channels_classes[:, :, roi_file_label["class_label_" + key][i] - 1] == 1)

                            all_subject_roi_metric_mean.append(rois_mean_metric)
                            all_names.append(out_dir)
pd.DataFrame(np.concatenate(all_subject_roi_metric_mean, axis=0), index=all_names,columns=roi_names_list).to_csv(
    "all_subject_dti_metric_roi_mean.csv")
print('finished')
