import numpy as np
import pandas as pd
import cv2
import imageio
from .ROIs import *




def merge_array_channels(image, channels):
    merged_image = np.zeros(image[:,:,0].shape)
    num_channels = len(channels)
    
    for channel in channels:
        subchannel_image = image[:,:,channel]
        merged_image += np.divide(subchannel_image, np.amax(subchannel_image)*num_channels)

    return merged_image



def create_pd_dataframe():
    return pd.DataFrame(columns=pd.MultiIndex.from_tuples([(-1, -1,  "Image"), (-1,-1,"Index"), (-1, -1, "ROI")]))



def find_ROI_match (Parameters, current_Image, ROI, dataframe):
    
    ROI_match_threshold = Parameters.ROI_match_threshold
    filename = current_Image.filename
    filepath = current_Image.file_path
    
    
    
    match_index = 0
    match_IoU = 0
    for index, compROI in enumerate(dataframe[(-1, -1, "ROI")]):
        IoU = calc_overlap(ROI, compROI)
        if (IoU > max(ROI_match_threshold, match_IoU)):
            #print (f"Found a match with {match_IoU} overlap!")
            match_index = index
            match_IoU = IoU

    if match_IoU:
        row_index = match_index #modify the current value
        #change the ROI in the comparison row??

    else:
        #append a new column
        row_index = len(dataframe.index)
        new_row = pd.DataFrame({(-1, -1, "ROI"): [ROI],
                                (-1, -1, "Image"): filename,
                                (-1, -1, "Index"): row_index,
                                (-1, -1, "Filepath"): filepath})
        dataframe = dataframe.append(new_row, ignore_index = True, sort=False)

    return(row_index, dataframe)  



def read_image(reader, start_index, channels):
    image_array = []
    print ("Start Index", start_index, "channels:", channels)
    for channel in channels:
        image_array.append(reader.get_data(start_index+channel))
    image = np.dstack(image_array)
    
    #expand dimensions
    if len(channels) == 1:
        image = np.expand_dims(image, axis=2)
    
    return (image)




def output_image_parameters(current_Image):
    
    timepoints = current_Image.timepoints 
    upper_dir_name = current_Image.directory.parent.name
    print (f'current image is :{current_Image.filename} in {upper_dir_name} and it has {timepoints} timepoints')
    
    



def load_dataframe(current_Image, dataframes):
    filename = current_Image.filename
    if filename not in dataframes:
        new_dataframe = create_pd_dataframe()
        dataframes.update({filename: new_dataframe})
    
    current_dataframe = dataframes[filename]
    all_columns = current_dataframe.columns
       
    master_column_offset = 0
    if len(all_columns)>3:
        master_column_offset = max(all_columns[3:], key = lambda x: x[0])[0] + 1
        
    return (master_column_offset, current_dataframe)
    
    
    
    
def reduce_to_int8(image):

    image = np.dstack([image, image*0.9, image*0.8])
    image = ((image/np.amax(image))*255).astype('uint8')
    return (image)



def create_debug_images(image, ROI_channels, subimage_channels):
    
    #This takes the middle ROI channel (usually in focus)
    #and the first of each fluorescence image (usually the brightest)
    
    debug_seg = reduce_to_int8(image[:,:,ROI_channels[1]])
    
    debug_subcell_array = []
    for channel in (np.cumsum(subimage_channels)-subimage_channels[0]):
        debug_subcell_array.append(reduce_to_int8(image[:,:,channel]))
    

    return (debug_seg, debug_subcell_array)

    
    

                               
def annotate_image (image, ROI, mask, width = 5, label_confidence = True, color = (255, 0, 0)):
    x_corner = ROI.x
    y_corner = ROI.y
    x_len = ROI.x_length
    y_len = ROI.y_length
    
    confidence = ROI.confidence
    
    classification = ROI.classification
    
    
    image = cv2.rectangle(image,
                         (x_corner,y_corner), 
                         (x_corner+x_len, y_corner+y_len), 
                         color, width)

    
    if classification and confidence and label_confidence:
        cv2.putText(img = image, text = f'Con: {confidence:.2f}', 
                    org = (x_corner, max(y_corner-5, 0)), 
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1, 
                    color = (255,255,255))
        
        
    
    subRegion = image[y_corner:y_corner+y_len, x_corner: x_corner+x_len, 2]
    subRegion[mask!=0] = 255
    
    return(image)
    

def export_debug_images(Parameters, current_Image, det_image, image_array, roi_list):
    
    root_dir = Parameters.data_path
    suffix = Parameters.suffix  
    directory = current_Image.directory
    filename = current_Image.filename 

    #Output the ROI Detections
    debug_det_path = root_dir/'debug_det'/directory.relative_to(root_dir)
    debug_det_path.mkdir(parents = True, exist_ok = True)
    debug_det_save_path = debug_det_path/filename.name.replace(suffix, f'-det.jpg')
    imageio.imwrite(str(debug_det_save_path), det_image)
    
    #Output the Segmentation Results
    debug_seg_path = root_dir/'debug_seg'/directory.relative_to(root_dir)
    debug_seg_path.mkdir(parents = True, exist_ok = True)
    for channel, debug_image in enumerate(image_array):
        debug_save_path = debug_seg_path/filename.name.replace(suffix, f'{channel}.jpg')
        imageio.imwrite(str(debug_save_path), debug_image)


    roi_array = np.zeros((len(roi_list), 4))

    for i, ROI in enumerate(roi_list):
        roi_array[i, :] = [ROI.x, ROI.y, ROI.x_length, ROI.y_length]
    roi_df = pd.DataFrame(roi_array, columns = ['X', 'Y', 'xlen', 'ylen'])
    debug_roi_save_path = debug_det_path/filename.name.replace(suffix, f'-det.csv')
    roi_df.to_csv(debug_roi_save_path)                      
        
                    