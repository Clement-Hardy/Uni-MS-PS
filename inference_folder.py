import os
import argparse
from utils import load_model
from run import run


parser = argparse.ArgumentParser()
parser.add_argument("--nb_img", type=int, default=-1, help="number of images")
parser.add_argument("--folder_save", type=str,
                    default='inference', help="path_to_save_results")
parser.add_argument("--path_obj", type=str,
                    required=True, help= 'path_to_imgs')
parser.add_argument('--cuda', action='store_true')
args = parser.parse_args()
        

mode_inference = True
model = load_model(path_weight="weights",
                   cuda=args.cuda,
                   mode_inference=mode_inference)

for folder in os.listdir(args.path_obj):
    obj_name = folder
    path_obj = os.path.join(args.path_obj,
                            folder)
    if len(obj_name)==0:
        obj_name = args.path_obj.split(os.sep)[-2]
    
    run(model=model,
        path_obj=path_obj,
        nb_img=args.nb_img,
        folder_save=args.folder_save,
        obj_name=obj_name)