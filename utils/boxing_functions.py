import os
import numpy as np
import pandas as pd
from pathlib import Path
import imageio

def find_files(path, suffix, directory_list = {}, double_subpath = True):

    
    for subpath in path.iterdir():
        if subpath.is_file() and subpath.suffix == suffix:
            
            if double_subpath == True:
                file_path = Path(subpath.parent.name)/subpath.name
                root_path = subpath.parent.parent
            else:
                file_path = subpath.name
                root_path = subpath.parent
                
                
            
            if root_path in directory_list:
                directory_list[root_path][file_path] = None

            else:
                directory_list[root_path] = {file_path : None}
                
        elif subpath.is_dir() and not subpath.name == "Segmented_Image":
            directory_list = find_files(subpath, suffix, directory_list, double_subpath)
      
    return (directory_list)

def iterate_through_files(directory_list, Parameters, dataframes = []):
    """
    The purpose of this function is to iterate through all of the files in the directory list and analyze
        the relevant files.
        
        
    Input:
    directory_list is a dictionary of root directories containing dictionaries of subdirectories
        In most cases, there will be one subdirectory per root_directory, however this may not
        always be the case
    
    
    Parameters - the overall imaging parameters passed through to the analysis
        .suffix - the suffix used to identify images of interest
    
    
    dataframes - the list of dataframes. In most cases, you will be creating a new one
    
    
    """   
    
    for root_dir in directory_list:
        #note: there will usually only be one subdirectory per root_dir       
        for files in directory_list[root_dir]:
            ROIs = read_ROI_File(root_dir/files)
            if (not ROIs.empty):
                ROIs = process_ROI_dataframe(ROIs)
                subsegment_image(Parameters, ROIs, root_dir, files)

    return (dataframes)
            

def open_channel(channel, reader, div):
    
    if reader.get_length() == 1:
        import_img = np.asarray(reader.get_data(0))
        x1, x2, x3 = import_img.shape
        if x1<5: import_img = import_img[channel, :,:]
        else: import_img = import_img[:,:, channel]
    
    else:
        import_img = np.asarray(reader.get_data(channel))
        
    if div: import_img = np.divide(import_img, np.amax(import_img))
    return (import_img)

def subsegment_image(Parameters, ROIs, root_dir, sub_path):

    """
    Use this module to prepare the images to help create ground truths of the cells    
    
    """
    
    channels = Parameters["jpeg_channels"]
    suffix = Parameters["suffix"]
    move_to_front = Parameters.get('move_to_front', False)
    ground_truths = Parameters.get("ground_truths", False) 
    if ground_truths: ground_truth_suffix = Parameters["ground_truth_suffix"]
    black_masks = Parameters.get("black_masks", False)
    debug = Parameters.get("debug_mode", False)
    

    if debug: print (root_dir, sub_path)
    if debug: print("subSegmenting image.... :)")


    
    filepath = root_dir/sub_path
    sub_JPEG_dir = root_dir/'Segmented_JPEGs'/sub_path.parent
    sub_cell_dir = root_dir/'Segmented_Image'/sub_path.parent
    sub_gt_dir = root_dir/'Segmented_Ground_Truth'/sub_path.parent
    sub_bm_dir = root_dir/'Segmented_Black_Mask'/sub_path.parent

    
    
    for dirs in [sub_JPEG_dir, sub_gt_dir, sub_cell_dir, sub_bm_dir]:
        if (not os.path.isdir(dirs)):
            os.makedirs(dirs)
    
    
    
    #Import the image for the JPEG
    
    reader = imageio.get_reader(str(filepath))
    images = []
    #Preprocess the image
    for channel in channels:
        #img = np.asarray(reader.get_data(channel))
        img = open_channel(channel, reader, False)


        #Should normalization happen after subsegmentation????
        img = img/np.amax(img)
        img = img*255
        img = img.astype('uint8')
        images.append(img)
        

    merged_img = np.dstack([images[0], images[1], images[2]])

    height, width, depth = merged_img.shape
    
    #Import the image for the real output
    full_image = imageio.volread(str(filepath))


    if move_to_front:
        for i, channel in enumerate(channels):
            if debug: print(f"swapping {i} with {channel}")
            full_image[i,:,:] = full_image[channel,:,:]
    
    
    
    if ground_truths:
        gt_filepath = filepath.parent/filepath.name.replace(suffix, ground_truth_suffix)
        if debug: print ("GT", gt_filepath)
        gt_img = imageio.imread(gt_filepath)
        gt_img *= 255
    
    

    if debug: print (ROIs)
    
    
    for i in range(len(ROIs)):
        x_min = max(0, int(ROIs.loc[i, 'xmin']))
        x_max = min(width, int(ROIs.loc[i, 'xmax']))
        y_min = max (0, int(ROIs.loc[i, 'ymin']))
        y_max = min(height, int(ROIs.loc[i, 'ymax']))
        
        subJPEGimage = merged_img[y_min:y_max, x_min: x_max, :]
        imageio.imwrite(str(sub_JPEG_dir/sub_path.name.replace(suffix, f'{i}.jpg' )), subJPEGimage)
        
        subimage = full_image[:, y_min:y_max, x_min: x_max]
        imageio.volwrite(str(sub_cell_dir/sub_path.name.replace(suffix, f'{i}{suffix}' )), subimage)    

        if black_masks:
            subblackmask = np.zeros(subimage.shape[1:])
            imageio.imwrite(str(sub_bm_dir/sub_path.name.replace(suffix, f'{i}.png' )), subblackmask)

        if debug: print (merged_img.shape, x_min, x_max, y_min, y_max, subJPEGimage.shape, subimage.shape,) 
                
        if ground_truths:
            gt_subimage = gt_img[y_min:y_max, x_min: x_max].copy()
            if debug: print (gt_img.shape, x_min, x_max, y_min, y_max, gt_subimage.shape,) 
            imageio.imwrite(str(sub_gt_dir/sub_path.name.replace(suffix, f'{i}.png' )), gt_subimage)

                
               
   
    
    return(None)

def read_ROI_File(filepath):
    
    text_path = str(filepath).replace(".ome.tif", "--labels.txt")
    
    if os.path.isfile(text_path):
        ROI_List = pd.read_csv(text_path, index_col = False)
        
    else:
        print (f"Warning: No ROI File found for {filepath}")
        ROI_List = pd.DataFrame()
          
    return (ROI_List)

def process_ROI_dataframe(ROIs):    
        
    #Switch X and Y for python standards
    
    ROIs['xmin'] = ROIs['X'] - ROIs['W']/2
    ROIs['xmax'] = ROIs['X'] + ROIs['W']/2
    ROIs['ymin'] = ROIs['Y'] - ROIs['H']/2
    ROIs['ymax'] = ROIs['Y'] + ROIs['H']/2

    return (ROIs)