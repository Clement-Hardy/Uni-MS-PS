import numpy as np
from PIL import Image
import os
import torch
import cv2
from Transformer_multi_res_7 import Transformer_multi_res_7




def resize(img, expected_size):
    img = cv2.resize(img,
                     expected_size,
                     interpolation=cv2.INTER_LANCZOS4)
    return img


def resize_with_padding(img, expected_size):
    if type(img)==np.ndarray:
        if img.dtype==np.uint16:
            img1 = img.astype(np.uint8)
        else:
            img1 = img
        img1 = Image.fromarray(img1)
    else:
        img1 = img
    img1.thumbnail((expected_size[0], expected_size[1]))

    delta_width = expected_size[0] - img1.size[0]
    delta_height = expected_size[1] - img1.size[1]
    pad_width = delta_width // 2
    pad_height = delta_height // 2
    padding = (pad_width,
               pad_height,
               delta_width - pad_width,
               delta_height - pad_height)
    img2 = np.zeros((expected_size[0], expected_size[1], 3))

    if padding[3]!=0 and padding[2]!=0:
        img2[padding[1]:-padding[3],
             padding[0]:-padding[2]] = img
    elif padding[3]!=0:
        img2[padding[1]:-padding[3],
             padding[0]:] = img
    elif padding[2]!=0:
        img2[padding[1]:,
             padding[0]:-padding[2]] = img
    else:
        img2[padding[1]:,
             padding[0]:] = img
    return img2, padding



def depadding(img, padding):
    img = np.array(img)
    if padding[3]!=0 and padding[2]!=0:
        img = img[padding[1]:-padding[3],
                  padding[0]:-padding[2]]
    elif padding[3]!=0:
        img = img[padding[1]:-padding[3],
                  padding[0]:]
    elif padding[2]!=0:
        img = img[padding[1]:,
                  padding[0]:-padding[2]]
    else:
        img = img[padding[1]:,
                  padding[0]:]
    return img

def normal_to_rgb(img):
    return (((img+1)/2)*255).astype(np.uint8)
   

def get_nb_stage(shape):
    max_shape = np.max(shape)
    nb_stage = np.ceil(np.log2(max_shape/32))+1
    nb_stage = int(nb_stage)
    return nb_stage
    
    
def load_imgs_mask(path, 
                   nb_img,
                   calibrated=False,
                   filenames=None,
                   max_size=None):
    
    if filenames is None:
        possible_file = os.listdir(path)
    else:
        possible_file = filenames

    temp = []
    
    for file in possible_file:
        if ".png" in file and "mask" not in file and "Normal" not in file and "normal" not in file:
            temp.append(file)
        elif ".jpg" in file and "mask" not in file and "Normal" not in file and "normal" not in file:
            temp.append(file)
        elif ".TIF" in file and "mask" not in file and "Normal" not in file and "normal" not in file:
            temp.append(file)
        elif ".JPG" in file and "mask" not in file and "Normal" not in file and "normal" not in file:
            temp.append(file)
            
            
    file_mask = os.path.join(path, "mask.png")
    if os.path.exists(file_mask):
        mask = cv2.imread(file_mask)
    else:
        file_img_example = os.path.join(path, temp[0])
        img_example = cv2.imread(file_img_example)
        mask = np.ones(img_example.shape, 
                       dtype=np.uint8)
    
    if max_size is not None:
        if mask.shape[0]>max_size or mask.shape[1]>max_size:
            mask = cv2.resize(mask,
                              (max_size, max_size))
            
    original_shape = mask.shape
    
    coord = np.argwhere(mask[:,:,0]>0)
    x_min, x_max = np.min(coord[:,0]), np.max(coord[:,0])
    y_min, y_max = np.min(coord[:,1]), np.max(coord[:,1])

    x_max_pad = mask.shape[0] - x_max
    y_max_pad = mask.shape[1] - y_max
        
    mask = mask[x_min:x_max,
                y_min:y_max]
        
    nb_stage = get_nb_stage(mask.shape)
    size_img = 32*2**(nb_stage-1)
    
    mask, _ = resize_with_padding(mask,
                                      expected_size=(size_img,
                                                     size_img))
    mask = (mask>0)
    mask = mask[:,:,0]
    
    
    imgs = []
            
    if nb_img is None or nb_img>=len(temp) or nb_img==-1:
        files = np.array(temp)
    else:
        files = np.random.choice(temp, nb_img, replace=False)
        
    for file in files:
        file1 = os.path.join(path, file)
        img = cv2.imread(file1, 
                         cv2.IMREAD_UNCHANGED)
        if len(img.shape)==2:
            img = np.expand_dims(img, -1)
            img = np.concatenate((img, img, img),
                                 axis=-1)
        
        if max_size is not None:
            if img.shape[0]>max_size or img.shape[1]>max_size:
                img = cv2.resize(img,
                                 (max_size, max_size))
                
        img = img[x_min:x_max,
                  y_min:y_max]
            
        img, padding = resize_with_padding(img=img,
                                           expected_size=(size_img, size_img))

        
        img = img.astype(np.float32)
        mean_img = np.mean(img, -1)
        mean_img = mean_img.flatten()
        mean_img1 = np.mean(mean_img[mask.flatten()])
        img = img/mean_img1
        
        imgs.append(img)
    
    
    imgs = np.array(imgs)
    imgs = np.moveaxis(imgs,
                       -1,
                       0)
    imgs = torch.from_numpy(imgs).unsqueeze(0).float()
    
    if calibrated:
        dirs_file = os.path.join(path,
                                 "light_directions.txt")
        dirs_all = np.loadtxt(dirs_file)
        dirs = []
        for key in files:
            key = int(key.split(".")[0])-1
            d = dirs_all[key]
            dirs.append([d[0],
                         d[1],
                         d[2]])
        
        dirs = np.array(dirs)
        dirs = torch.from_numpy(dirs).movedim(1,0).unsqueeze(0)
        dirs.unsqueeze_(-1).unsqueeze_(-1)
        
        dirs = dirs.expand_as(imgs)[:,:,:,:]
    
        imgs = torch.cat([imgs, dirs], 1).float()
    
    
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    
    return imgs, mask, padding, [x_min, x_max_pad, y_min, y_max_pad], original_shape



def load_model(path_weight, cuda,
               calibrated, mode_inference=False): 
    if calibrated:
        file_weight = os.path.join(path_weight, "model_calibrated.pth")
    else:
        file_weight = os.path.join(path_weight, "model_uncalibrated.pth")
    
    if calibrated:
        model = Transformer_multi_res_7(c_in=6)
    else:
        model = Transformer_multi_res_7(c_in=3)
        
    model.load_weights(file=file_weight)
    model.eval()
    if mode_inference:
        model.set_inference_mode(use_cuda_eval_mode=cuda)
    elif cuda:
        model.cuda()
    return model


def process_normal(model, imgs, mask):
    nb_stage = get_nb_stage(mask.shape)
    x = {}
    x["imgs"] = imgs
    x["mask"] = mask

    with torch.no_grad():
        a = model.process(x,
                          nb_stage)
        normal = a["n"].squeeze().movedim(0,-1).numpy()
    return normal
        
