import torch
import numpy as np
import cv2


def decrease_size_batch(imgs, binary=False,
                        channel_first=True, scale=1,
                        f=0.5, size=None):
    results = []
    type_input = type(imgs)
    use_cuda = imgs[0].is_cuda
    
    if len(imgs.shape)==5:
        imgs = torch.movedim(imgs, 1,
                             0)
        for temporal in imgs:
            results.append(decrease_size_batch(imgs=temporal,
                                               binary=binary,
                                               channel_first=channel_first,
                                               scale=scale,
                                               f=f,
                                               size=size))
            
    else:
        for img in imgs:
            img = img.cpu().detach().numpy()
            
            if channel_first:
                img = np.moveaxis(img, 0, -1) 
            if binary:
                img = img[:,:,0]*1.0
            if size is None:
                img = decrease_size_img(img=img,
                                        scale=scale,
                                        f=f)
            else:
                img = cv2.resize(img, size)
            if binary:
                img = (img>0)
                img = np.expand_dims(img, -1)
            if channel_first:
                if len(img.shape)==2:
                    img = np.expand_dims(img, -1)
                img = np.moveaxis(img, -1, 0)
            img = torch.from_numpy(img)
            if use_cuda:
                img = img.cuda()
            results.append(img)
        
    if type_input==torch.Tensor:
        results = torch.stack(results)
    if len(imgs.shape)==5:
        results = torch.movedim(results, 
                                0,
                                1)
    return results

def decrease_size_img(img, scale=1, f=2):
    for i in range(scale):
        size_x = int(img.shape[1]/f)
        size_y = int(img.shape[0]/f)
        img = cv2.resize(img, (size_x, size_y))
    return img