import matplotlib.pyplot as plt
import numpy as np
import imageio
import pathlib
from pathlib import Path
from .DeepAnalysis import deep_analysis


"""
These functions rely on a cache which contains the following:

Filename                
Segmentation Images     Shape [H,W,3]
Detection Image         Shape [H,W,3]
ROI List                Shape [ROIs]
Cell Images             Shape [ROIs][h,w,sub_channels]
Subcellular Images      Shape [ROIs][Channel][h,w,sub_channels]
Cell Masks              Shape [ROIs][h,w]
Subcellular Masks       Shape [ROIs][Channel][Subchannels][h,w]
Premask Images          Shape [ROIs][h,w,3]

where:
h,w are the dimensions of the ROI
H,W are the dimensions of the original image

"""



def output_debug_single_ROI(cache, ROI, output_path = Path(r"Debug_Output"), filename = "debug_ROIs.png"):
    
    output_path.mkdir(exist_ok = True, parents = True)
    img_filename = cache['Filename']
    cell_masks = cache['Cell Masks']
    subcell_masks = cache['Subcellular Masks']
    cell_images = cache['Cell Images']
    subcell_images = cache['Subcellular Images']


    num_roi = len(subcell_images)

    assert ROI<num_roi, f"ROI {ROI} not available. There are only {num_roi} ROIs for this image"

    num_channels = len(subcell_images[0])
    img_per_channel = [len(x) for x in cache['Subcellular Masks'][0]]


    rows = num_channels + 1
    #3+Num_Channels is the requiement for the first row
    #2*max_img_per_channel is the requirement for the rest of the rows
    columns = max(2*max(img_per_channel), 3+num_channels)

    h, w, ch = cell_images[0].shape

    size_factor = 10

    fig, ax = plt.subplots(rows, columns, figsize = (20,20))

    #Output the brightfield Information
    cell_image = np.divide(cell_images[ROI], np.amax(cell_images[ROI]))
    ax[0][0].imshow(cell_image)
    ax[0][1].imshow(cell_masks[ROI])
    ax[0][2].imshow(np.multiply(cell_image, np.expand_dims(cell_masks[ROI], 2)))

    for i in range(num_channels):
        
        subcell_img = np.divide(subcell_images[ROI][i], np.amax(subcell_images[ROI][i]))
        ax[0][3+i].imshow(np.multiply(subcell_img, np.expand_dims(cell_masks[ROI], 2)))



    #Output the individual ROIs
    for i, (subcell_image, subcell_mask) in enumerate(zip(subcell_images[ROI], subcell_masks[ROI])):
        for col in range(img_per_channel[i]):

        #cell_image = np.divide(cell_images[r], np.amax(cell_images[r]))

        #ax[r][0].imshow(cell_image)
        #ax[r][1].imshow(cell_masks[r])
        
            sub_img = np.divide(subcell_image, np.amax(subcell_image))
                
            ax[i+1][col*2].imshow(sub_img[:,:,i])
            ax[i+1][col*2+1].imshow(subcell_mask[i])

    fig.suptitle(f"ROI {ROI} for image: {img_filename}")

    fig.savefig(output_path/filename)









def output_debug_ROIs(cache, output_path = Path(r"Debug_Output"), filename = "debug_ROIs.png"):
    

    output_path.mkdir(exist_ok = True, parents = True)
    cell_masks = cache['Cell Masks']
    subcell_masks = cache['Subcellular Masks']
    cell_images = cache['Cell Images']
    subcell_images = cache['Subcellular Images']
    premask_images = cache['Premask Images']
    
    num_roi = len(subcell_images)
    num_channels = len(subcell_images[0])

    rows = num_roi
    columns = num_channels + 3 #subchannels + cell image and mask


    h, w, ch = cell_images[0].shape

    size_factor = 10


    fig, ax = plt.subplots(rows, columns, figsize = (20,20))

    for r in range(rows):

        cell_image = np.divide(cell_images[r], np.amax(cell_images[r]))
        premask_image = np.divide(premask_images[r], np.amax(premask_images[r]))

        ax[r][0].imshow(cell_image)
        ax[r][1].imshow(premask_image)
        ax[r][2].imshow(cell_masks[r])
        
        for i, (subcell_image, subcell_mask) in enumerate(zip(subcell_images[r], subcell_masks[r])):
            sub_img = np.divide(subcell_image, np.amax(subcell_image))
            
            ax[r][i+3].imshow(sub_img)
            ax[r][i+3].imshow(subcell_mask[0], alpha = 0.5)
            
    fig.savefig(output_path/filename)


            
def output_debug_images(cache, output_dir = Path("Debug_Output")):
    cell_masks = cache['Cell Masks']
    subcell_masks = cache['Subcellular Masks']
    cell_images = cache['Cell Images']
    subcell_images = cache['Subcellular Images']

    num_channels = len(subcell_images[0])

    cell_output_path = output_dir/'Debug Outputs (Cells)'; cell_output_path.mkdir(exist_ok = True, parents = True)
    subcell_output_path = output_dir/'Debug Outputs (Subcells)'; subcell_output_path.mkdir(exist_ok = True, parents = True)

    channel_folders = list(map(str, range(num_channels)))


    for folder_name in channel_folders:
        (subcell_output_path/folder_name).mkdir(exist_ok = True)



    for i, cell_img in enumerate(cell_images):
        imageio.volwrite(str(cell_output_path/f'{i}.ome.tif'), cell_img)

    for i, img_array in enumerate(subcell_images):
        for channel, img in enumerate(img_array):
            imageio.volwrite(str(subcell_output_path/channel_folders[channel]/f'{i}.ome.tif'), img)





def output_debug_summary(cache, output_path = Path(r"Debug_Output"), filename = "debug_summary.png"):
    output_path.mkdir(exist_ok = True, parents = True)
    
    num_channels = len(cache['Segmentation Images'])

    fig, ax = plt.subplots(3,2, figsize = (30,40))


    flat_ax = ax.flatten()

    flat_ax[0].imshow(cache['Detection Image'])
    flat_ax[1].imshow(cache['Detection Image'])

    for i, axes in enumerate(flat_ax[2:]):
        axes.imshow(cache['Segmentation Images'][i])
        
    fig.savefig(output_path/filename)



def single_image_analysis(filename, Parameters, dataframes = {}):
    
    
    current_Image = Parameters.ImageClass(Parameters.data_path,
                                              filename, Parameters.channel_array,
                                              Parameters.image_type_array, Parameters.channel_thresholds)
    
    
    dataframes, cache = deep_analysis(current_Image, Parameters, dataframes)
    
    return (current_Image, cache)