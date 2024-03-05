import cv2
import os
import torch
import scipy.io
import numpy as np
from utils import depadding, normal_to_rgb
from utils import load_imgs_mask, process_normal


def run(model, path_obj, nb_img, folder_save, obj_name, calibrated):
    imgs, mask, padding, zoom_coord, original_shape = load_imgs_mask(path=path_obj,
                                                                     nb_img=nb_img,
                                                                     calibrated=calibrated)
        
    normal = process_normal(model=model,
                            imgs=imgs,
                            mask=mask)

    normal_resize = depadding(normal,
                              padding=padding)

    normal_resize = torch.from_numpy(normal_resize)
    normal_resize = torch.nn.functional.normalize(normal_resize, 2, -1).numpy()
        
    pad_x_min = np.zeros((zoom_coord[0], normal_resize.shape[1], 3))
    pad_x_max = np.zeros((zoom_coord[1], normal_resize.shape[1], 3))
    normal_resize = np.concatenate((pad_x_min,
                                    normal_resize,
                                    pad_x_max), axis=0)
            
    pad_y_min = np.zeros((normal_resize.shape[0], zoom_coord[2], 3))
    pad_y_max = np.zeros((normal_resize.shape[0], zoom_coord[3], 3))
            
    normal_resize = np.concatenate((pad_y_min,
                                    normal_resize,
                                    pad_y_max), axis=1)
      
    normal_resize_rgb = normal_to_rgb(normal_resize)

    if not os.path.exists(folder_save):
            os.makedirs(folder_save)
     
    cv2.imwrite(os.path.join(folder_save, "{}.png".format(obj_name)),
                normal_resize_rgb[:,:,::-1])
        
    scipy.io.savemat(os.path.join(folder_save, "{}.mat".format(obj_name)),
                     {'Normal_est': normal_resize})