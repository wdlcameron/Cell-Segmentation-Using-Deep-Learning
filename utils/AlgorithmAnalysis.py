import pandas as pd
import numpy as np


def image_Analysis(current_Image, Parameters, dataframes):
    """
    Analyze the current file.  Whether you are using machine learning or an algorithm, much of the process is the same
    
    current_Image:
        directory
        filename
        file_path
        timepoints
        total_frames
        width
        length
        channel_array
        channel_offset
        image_type_array
        images_per_timepoint
        
        dir_name
        upper_dir_name
        
    
    prev_images is a dictionary of all of the most current values from the previous images
        If a cell is found in the first image but not the second, it will still show up in the prev_images for the third image 
    
    
    dataframes is a list of all the dataframes associated with the analysis.
        different groups of data should have different dataframes (aka, data for a different graph)
        
        Dataframes is a list of dictionaries, each containing a dataframe for each image that it is analyzing.
        Once the analysis is complete, the dictionary will be replaced with a single entry dictionary
        {"Total": pd.DataFrame}
    
       
    """
        
    #g_factor = Parameters.g_factor    

        
    #Import the relevant parameters from the ImagingParameters Class   
    root_dir = Parameters.data_path
    suffix = Parameters.suffix
    debug_mode = Parameters.debug_mode
    ROI_match_threshold = Parameters.ROI_match_threshold
    machine_learning_mode = Parameters.machine_learning_mode
    segmentation_outputs = Parameters.segmentation_outputs
    
    if machine_learning_mode:

        segmentation_models = Parameters.segmentation_models
        segmentation_weights = Parameters.segmentation_weights
        segmentation_network_size = Parameters.segmentation_network_size
    
    
    
    
    #Import the relevant parameters from the ImageClass Class    
    directory = current_Image.directory
    total_frames = current_Image.total_frames
    timepoints = current_Image.timepoints
    filepath = current_Image.file_path
    filename = current_Image.filename
    channel_offset = current_Image.channel_offset
    image_type_array = current_Image.image_type_array
    images_per_timepoint = current_Image.images_per_timepoint
    upper_dir_name = current_Image.upper_dir_name
    channel_thresholds = current_Image.channel_thresholds
    
    
    
    general_strings = ['Image', 'ROI']
    anisotropy_strings = ['Para', 'Perp', 'AniPixel', 'AniAvg']
    intensity_strings = ['Intensity']
    
    
    para_intensity_str = f'Para'
    perp_intensity_str = f'Perp'
    anipix_intensity_str = f'AniPixel'
    aniavg_intensity_str = f'AniAvg'

    intensity_string = "Intensity"
    
    image_string = "Image"
    image_ROI_string = "ROI"
    

        
    if filename not in dataframes:
        new_dataframe = create_pd_dataframe()
        dataframes.update({filename: new_dataframe})
    
    
    current_dataframe = dataframes[filename]
    
    
    all_columns = current_dataframe.columns
    
    master_column_offset = 0
    if len(all_columns)>3:
        master_column_offset = max(all_columns[3:], key = lambda x: x[0])[0] + 1
    
        
    print (f'current image is :{current_Image.filename} in {upper_dir_name} and it has {timepoints} timepoints and length df: {len(dataframes)}')
    for time in range(timepoints):
        
        #for string in general_strings:
        #    current_dataframe[(t, channel, string)] = np.nan
        
        for channel, (image_type, offset) in enumerate(zip(image_type_array, channel_offset)):
            #print (f"Current time is: {t}, anisotropy value is {anisotropy} and offset is {offset}")
            total_offset = time * images_per_timepoint + offset
            t = time + master_column_offset
            
            if image_type == "Anisotropy" or image_type == "Intensity":
            

                #Open the image
                
                Images = {}
 
                if image_type == "Anisotropy":
                    Images['Open'] = np.asarray(bioformats.load_image(filepath, c = 0, z=0, t=total_offset + 0, rescale=False))
                    Images['Para'] = np.asarray(bioformats.load_image(filepath, c = 0, z=0, t=total_offset + 1, rescale=False))
                    Images['Perp'] = np.asarray(bioformats.load_image(filepath, c = 0, z=0, t=total_offset + 2, rescale=False))

                    raw_image = np.dstack((Images['Open'], Images['Para'], Images['Perp']))
                    
                    anisotropy_image = calculate_anisotropy(raw_image, Parameters)
                    
                    Images['AniPixel'] = anisotropy_image[:,:,2].copy()
                                       
                    image_for_ROIs = Images['Open']
                    
                    for string in anisotropy_strings:
                            current_dataframe[(t, channel, string)] = np.nan
                    
                    
                    
                    
                elif image_type == "Intensity":
                    Images['Intensity'] = np.asarray(bioformats.load_image(filepath, c = 0, z=0, t=total_offset + 0, rescale=False))
                    image_for_ROIs = Images['Intensity']
                    
                    for string in intensity_strings:
                        current_dataframe[(t, channel, string)] = np.nan
                        

                debug_path = directory.replace(root_dir, f'{root_dir}{os.sep}debug_ml')
                if not os.path.isdir(debug_path):
                        os.makedirs(debug_path)
                debug_save_path = f'{debug_path}{os.sep}{filename}'.replace(suffix, f'{channel}.jpg')
            
            
                
                #Find ROIs and iterate through them
                
                if machine_learning_mode == True:
                    if image_type == "Anisotropy":
                        ROI_list = detect_ROIs(Parameters, raw_image, debug_path = debug_save_path)
                    elif image_type == "Intensity":
                        ROI_list = detect_ROIs(Parameters, np.dstack([image_for_ROIs, 
                                                                      image_for_ROIs*0.35, 
                                                                      image_for_ROIs*0.3]), 
                                               debug_path = debug_save_path)
                
                else:
                    ROI_list = find_ROIs(image_for_ROIs, 
                                         Parameters, 
                                         threshold = channel_thresholds[channel])

                
                #Debug Mode: Output an image with all the ROIs
                if debug_mode == True:
                    debug_path = directory.replace(root_dir, f'{root_dir}{os.sep}debug')
                    if not os.path.isdir(debug_path):
                        os.makedirs(debug_path)
                        
                    debug_image = image_for_ROIs.copy()
                    debug_image = np.dstack([debug_image, debug_image*0.9, debug_image*0.8])
                    debug_image = ((debug_image/np.amax(debug_image))*255).astype('uint8')
                    
                    
                
                #Output the ROIs as seperate images
                #segmentation_outputs = True
                if segmentation_outputs == True:
                    segmentation_path = directory.replace(root_dir, f'{root_dir}{os.sep}segmentation')
                    if not os.path.isdir(segmentation_path):
                        os.makedirs(segmentation_path)

              
                
                

                for idx, ROI in enumerate(ROI_list):
                    match_index = 0
                    match_IoU = 0
                    for index, compROI in enumerate(current_dataframe[(-1, -1, "ROI")]):
                        #print ("Index is ", index)
                        #print (current_dataframe["ROI"])
                        current_dataframe.head(5)
                        IoU = calc_overlap(ROI, compROI)
                        if (IoU > max(ROI_match_threshold, match_IoU)):
                            match_index = index
                            match_IoU = IoU
                    
                    if match_IoU:
                        row_index = match_index #modify the current value
                        #change the ROI in the comparison row??
                        
                    else:
                        #append a new column
                        row_index = len(current_dataframe.index)
                        new_row = pd.DataFrame({(-1, -1, "ROI"): [ROI],
                                                (-1, -1, "Image"): filename,
                                                (-1, -1, "Index"): row_index})
                        current_dataframe = current_dataframe.append(new_row, ignore_index = True, sort=False)


                    #Use the ROI to append useful information to the dataframe    
                    x_corner = ROI.x
                    y_corner = ROI.y
                    x_len = ROI.x_length
                    y_len = ROI.y_length
                    
                    
                    
                    ROI_data = {}                    
                    data_inputs = {}
                    
                    
                    """
                    Insert Thresholding Module Here!!!  Faster to threshold the sub ROIs than the whole image 
                    
                    
                    
                    
                    """
                    
                    
                    
                    
                    
                    
                    
                    
                    if image_type == "Anisotropy":
                        
                        
                        if machine_learning_mode:
                            
                            seg_data = raw_image[y_corner:y_corner+y_len, x_corner: x_corner+x_len,:].copy()
                            
                            #If you have only loaded one network, use it for every channel
                            #Otherwise, load the appropriate network
                            if len(segmentation_weights) == 1:
                                seg_network = segmentation_models[0]

                            else:
                                seg_network = segmentation_models[channel]
                                
                                
                            ROI.data = segment_ROIs(seg_data, seg_network, segmentation_network_size)


                        
                        image_types = ['Para', 'Perp', 'AniPixel']
                        subRegion = {}
                        subRegionAvg = {}
                                               
                        for types in image_types:
                            subRegion[types] = Images[types][y_corner:y_corner+y_len, x_corner: x_corner+x_len].copy()
                            subRegion[types] = np.multiply(subRegion[types], ROI.data)
                            data_inputs[types] = np.mean(subRegion[types][subRegion['Para']!=0])

                            
                                
                        para_value = data_inputs['Para']
                        perp_value = data_inputs['Perp']
                        data_inputs['AniAvg'] = (para_value - perp_value)/(para_value + 2*perp_value)
                        
                        
                        for types in data_inputs:
                            current_dataframe.at[row_index, (t, channel, types)] = data_inputs[types]

                        
                        #Export ROIs for training...
                        if segmentation_outputs == True:
                            
                            subRegion['Open'] = Images['Open'][y_corner:y_corner+y_len, x_corner: x_corner+x_len].copy()
                            ROI_image = np.dstack([subRegion["Open"].copy(), subRegion["Para"].copy(), subRegion["Perp"].copy()])
                            ROI_image = ((ROI_image/np.amax(ROI_image))*255).astype('uint8')
                            seg_save_path = f"{segmentation_path}{os.sep}{filename}".replace(suffix, f"{channel}-{idx}.jpg")
                            imageio.imwrite(seg_save_path, ROI_image)
                            
                            
                            
                            
                            
                            
                         
                    elif image_type == "Intensity":
                        int_extract = Images['Intensity'][y_corner:y_corner+y_len, x_corner: x_corner+x_len].copy()
                        int_extract = np.multiply(int_extract, ROI.data)
                        int_value = np.mean(int_extract[int_extract!=0])
                        
                        current_dataframe.at[row_index, (t, channel, intensity_string)] = int_value
                        
                        
                    if debug_mode == True:
                        #write the ROI box
                        debug_image = cv2.rectangle(debug_image,
                                                    (x_corner,y_corner), 
                                                    (x_corner+x_len, y_corner+y_len), 
                                                    (255,0,0), 5)
                        
                        debug_subRegion = debug_image[y_corner:y_corner+y_len, x_corner: x_corner+x_len, 2]
                        debug_subRegion[ROI.data!=0] = 255
                        

                        
                if debug_mode == True:
                    debug_save_path = f'{debug_path}{os.sep}{filename}'.replace(suffix, f'{channel}.jpg')
                    imageio.imwrite(debug_save_path, debug_image)
                        
    
    
    #outside the loop, not really useful for anything
    dataframes[filename] = current_dataframe  
    
    return (dataframes)