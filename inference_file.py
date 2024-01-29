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

obj_name = args.path_obj.split(os.sep)[-1]
if len(obj_name)==0:
    obj_name = args.path_obj.split(os.sep)[-2]
    
model = load_model(path_weight="weights",
                   cuda=args.cuda,
                   mode_inference=mode_inference)

run(model=model,
    path_obj=args.path_obj,
    nb_img=args.nb_img,
    folder_save=args.folder_save,
    obj_name=obj_name)