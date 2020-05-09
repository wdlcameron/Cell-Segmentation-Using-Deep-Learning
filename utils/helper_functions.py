import numpy as np
from segmentation_dataclass import SegMultiImageList
from fastai.vision import get_transforms

#Support Functions
def get_y_lambda_function(root_path, mask_folder = 'Mask_Norm', image_folder = 'FullImage'):
    return lambda filename: root_path/mask_folder/filename.relative_to(root_path/image_folder).parent/(filename.name.replace('ome.tif', 'png'))

def load_core_paths(root_path):
    mask_path = root_path/"Mask" ; mask_path.mkdir(exist_ok = True)
    seg_path = root_path/"FullImage" ; seg_path.mkdir(exist_ok = True)
    codes_path = root_path/"codes.txt"
    model_path = root_path/'Models' ; model_path.mkdir(exist_ok = True)
    figures_path = root_path/'Figures' ; figures_path.mkdir(exist_ok = True)

    codes = np.loadtxt(codes_path, dtype = 'str')
    return (mask_path, seg_path, codes_path, model_path, figures_path, codes)

def get_data(root_path, tfms= get_transforms(), size= 400, bs = 4):
    seg_path = root_path/"FullImage"
    codes_path = root_path/"codes.txt"
    get_mask_func = get_y_lambda_function(root_path)
    
    codes = np.loadtxt(codes_path, dtype = 'str')
    #name2id = {code:number for number,code in enumerate(codes)}

    data = (SegMultiImageList
             .from_folder(seg_path, recurse = True)
             .split_by_rand_pct()
             .label_from_func (get_mask_func, classes = codes)
             .transform(tfms = tfms, size = size, tfm_y = True, padding_mode = 'border')
             .databunch(bs = bs, num_workers = 0))
    
    return data
