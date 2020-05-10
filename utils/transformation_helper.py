from .segmentation_dataclass import open_image_custom
#from fastai.vision import custom_tfms
import numpy as np
import torch
from functools import partial
import matplotlib.pyplot as plt


def get_example_image(example_image_path, channel_codes):
    return open_image_custom(example_image_path, ch = channel_codes)

def plots_f(rows, cols, width, height, filepath, custom_tfms, channel_codes, **kwargs):
    """ Plot multiple possible output of the transforms"""
    get_img = partial(get_example_image, filepath, channel_codes)
    fig, axes = plt.subplots(rows,cols,figsize=(width,height))
    [get_img().apply_tfms(custom_tfms, padding_mode = 'border', **kwargs).resize((3, width,height)).show(ax=ax) for i,ax in enumerate(axes.flatten())]
    return fig

def one_hot_encoding_array (x, scale=1):
    array = torch.tensor(np.zeros(8))
    array[x] = scale
    return array