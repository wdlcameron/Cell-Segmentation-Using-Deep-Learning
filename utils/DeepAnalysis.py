import numpy as np
import pandas as pd
from .feature_extraction import extract_anisotropy_features, extract_intensity_features, extract_area
from .ROIs import *


from .analysis_helpers import *
from .detect import detect_ROIs
from .segment import segment_ROIs
from .yolo_functions import get_channel




def process_masking_array(image, code, channels_to_merge = None):
    if code <20:
        return (image[:,:,code])
    
    elif code == 100: #Merge channels
        return merge_array_channels(image, channels_to_merge)
        
    elif code == 110: #Black image (all zeros with one bright pixel)
        blank_image = np.zeros(image[:,:,0].shape)
        blank_image = np.expand_dims(blank_image, axis = 2)
        blank_image[0,0,0] = 1
        return blank_image

    else:
        raise Exception(f"Do not know how to process this code.  Check the DeepAnalysis.py file")


def premask_processing(image, channel_codes, channels_to_merge):
    
    premask = []
    for channel_code, merge_channels in zip(channel_codes, channels_to_merge):
        #print (channel_code, merge_channels)
        premask.append(process_masking_array(image, channel_code, merge_channels))
        
    #for mask in premask: print (mask.shape)
    return (np.dstack(premask))
        


def deep_analysis(current_Image, Parameters, dataframes):
    
    debug_mode = Parameters.debug_mode
    cell_segmentation_model = Parameters.cell_segmentation_model
    cell_segmentation_size = Parameters.cell_segmentation_size
    subcell_segmentation_models = Parameters.subcell_segmentation_models
    image_organelle_array = Parameters.current_image_organelle_array
    ROI_detection_channels = Parameters.detection_channels
    mask_detection_channels = Parameters.processed_mask_detection_channels
    mask_merge_channels_array = Parameters.mask_merge_channels_array    #This is a list of channels to merge (ex.[[1,2,3], None, [5,7,9]])
    channel_array = Parameters.channel_array
    image_type_array = Parameters.image_type_array
    images_per_timepoint = Parameters.frames_per_timepoint
    masking_area_threshold_pct = Parameters.CELL_MASKING_AREA_THRESHOLD_PCT
    subcellular_area_threshold_pct = Parameters.SUBCELL_AREA_THRESHOLD_PCT

    manual_boxing = Parameters.manual_boxing

    expansion_factor = Parameters.ROI_expansion_factor
    
    
    filename = current_Image.filename
    filepath = current_Image.file_path
    reader = current_Image.reader
    timepoints = current_Image.timepoints 

   
    
    #Hook in functions to be used by deep_analysis (TBA)

    output_image_parameters(current_Image)     #Output information about the image
    #Load the correct dataframe
    master_column_offset, current_dataframe = load_dataframe(current_Image, dataframes)

    
    debug_cache = {}
    for time in range(timepoints):
        #total_offset = time * images_per_timepoint
        t = time + master_column_offset
        

        
        """
        Analysis Steps are as follows...
        
        0) Open the image
        1) Get ROIs
        2) Iterate through ROIs
            A) Find ROI match
            B) Mask Cells
            C) Collect informations from Masks (Cell Type, Area, Size, etc...)
            D) Apply mask to image
            E) Iterate through channels
                i) ID Cell (Organelle, Anything there?, etc.)
                ii) Check that ID matches organelle tag
                iii) Segment ROIs (either the normal or new)
                iv) Get information from segmentation

            E) Add information to the dataframe
        
        
        
        We can hook in all of these functions from the Parameters class
        
        """
        
        
        #0) Open entire image
        raw_image = read_image(reader, time*images_per_timepoint, range(images_per_timepoint)) #Shape (H, W, Channels)


        #0b) Determine which model to use for subsegmentation
        #Note: This is now handled in the base prorgam
        organelle_array = image_organelle_array


        
        
        #1) Get ROIs and create the debug images if necessary
        #detection_raw_image = raw_image[:, :, ROI_detection_channels]      
        

        detection_raw_image = []
        for channel in ROI_detection_channels:
            detection_raw_image.append(get_channel(channel, reader, True, dropout_chance = 0.0))

        detection_raw_image = np.dstack(detection_raw_image) 





        
        #
        if manual_boxing: ROI_list = create_ROI_list_from_file(filepath)
            #ROI_list = read_ROI_File(filepath)
            #if (not ROI_list.empty):
            #    ROI_list = process_ROI_dataframe(ROI_list)
            #    ROI_list = create_ROI_list(ROI_list)

            #else:
            #    ROI_list = None

        else: ROI_list = detect_ROIs(Parameters, detection_raw_image, expansion_factor = expansion_factor)
        

        
        
        if debug_mode: 
            debug_seg, debug_array = create_debug_images(raw_image, ROI_detection_channels, channel_array)
            for ch in range(detection_raw_image.shape[2]):
                debug_seg[:,:,ch] = ((detection_raw_image[:,:,ch]/np.amax(detection_raw_image[:,:,ch]))*255).astype('uint8')
        
        """
        #Load the analysis images:
        image_array = []
        for channel, (num_frames, offset) in enumerate(zip(channel_array, channel_offset)):
            image_array.append(read_image(reader, t+offset, range(num_frames)))
        """
        
        #2) Iterate through the ROIs
        
        
        
        overall_cellular_masks = []
        overall_subcellular_masks = []
        overall_subcellular_images = []
        overall_subcellular_segmentations = []
        overall_cellular_images = []
        overall_premask_images = []
        
        
         
        for ROI in ROI_list:
            x_corner = ROI.x
            y_corner = ROI.y
            x_len = ROI.x_length
            y_len = ROI.y_length
            
            subimage = raw_image[y_corner:y_corner+y_len, x_corner:x_corner+x_len, :]
            if debug_mode == True: debug_seg = annotate_image(debug_seg, ROI, np.zeros(ROI.data.shape), width = 1, label_confidence = False)
            
            
            # A) Find ROI Match (Can change this to be more sophisticated later)
            row_index, current_dataframe = find_ROI_match(Parameters, current_Image, ROI, current_dataframe)
            
            
            mask_inputs = {} # create a dictionary that will be updated
            
            # B) Mask Cells
            
            #Replace with the mask function

            #Get proper image for masking

            #premask_image = premask_processing(subimage, mask_detection_channels, mask_merge_channels_array)


            premask_raw_image = []
            for channel in mask_detection_channels:
                premask_raw_image.append(get_channel(channel, reader, True, dropout_chance = 0.0))

            premask_image = np.dstack(premask_raw_image)[y_corner:y_corner+y_len, x_corner:x_corner+x_len, :] 
    
            cell_mask = segment_ROIs(premask_image, cell_segmentation_model, cell_segmentation_size)
            ROI.data = cell_mask
            if debug_mode == True: debug_seg = annotate_image(debug_seg, ROI, ROI.data , width = 1, label_confidence = False)
                
                
            overall_cellular_masks.append(cell_mask)
            overall_premask_images.append(premask_image)
            #overall_cellular_images.append(subimage[:,:,ROI_detection_channels].copy())
            overall_cellular_images.append(detection_raw_image[y_corner:y_corner+y_len, x_corner:x_corner+x_len, :].copy())
            

            
            # C) Extract Information From Mask

            mask_inputs.update(extract_area(Parameters, cell_mask))
            mask_inputs.update({'X_Corner': x_corner, 'Y_Corner':y_corner, 'Width': x_len, 'Height': y_len})

            mask_inputs.update({'ROI':ROI})

            for types in mask_inputs:
                current_dataframe.at[row_index, (t, -1, types)] = mask_inputs[types]

            
            
            # D) Apply the mask to the entire image
            
            
       
            

            
            # E) Iterate through the channels...
            
            
            
            
            subcellular_masks = []
            subcellular_images = []
            subcellular_segmentations = []
            
            
            offset_array = np.cumsum(channel_array)-channel_array[0]
            for channel, (start, channel_len, image_type) in enumerate(zip(offset_array, channel_array, image_type_array)):
                data_inputs = {}
                if (image_type=="Anisotropy" or image_type == "Intensity"):
 
                    subcell_channels = range(start, start+channel_len)
                    subcell_image = subimage[:,:, subcell_channels]
        
            
            
                    # i) ID Cell (Organelle, Anything there?, etc.)
                    # cell_id = identify_subcellular_compartment(subcell_image)

                
                
                    # ii) Check that ID matches organelle tag
                    # assert it is the same as expected (only if organelle check is there...)
                    
                    
                    
                    # iii) Segment ROIs (either the normal or new)
                    #New approach is to segment each individual image.
                    #Right not it uses the first and applies a mask throughout.



                    
                    #print(subcell_segmentation_models, organelle_array, channel)
                    
                    masked_subimage = subcell_image
                    subcell_segmentation_model, subcell_segmentation_size = subcell_segmentation_models[organelle_array[channel]]
                    subcellular_segmentation = []
                    subcellular_mask_outputs = []
                    for ch in subcell_channels:
                        masked_subimage = subimage[:,:, ch]



                        subimage_segmentation = segment_ROIs(np.dstack([masked_subimage,masked_subimage,masked_subimage]), 
                                                                subcell_segmentation_model, 
                                                                subcell_segmentation_size)


                        ROI.add_subsegmentation(subimage_segmentation)

                        subcellular_mask_outputs.append(subimage_segmentation)

                        subimage_segmentation = np.multiply(masked_subimage, subimage_segmentation)
                        subcellular_segmentation.append(subimage_segmentation)

                        #subcellular_mask_segmentation = np.expand_dims(subimage_segmentation, axis=2)   
                        
                                     

                    
                    segmented_image = np.dstack(subcellular_segmentation)
                    #subcellular_mask = np.expand_dims(subcellular_segmentation, axis=2)
                    #segmented_image = np.multiply (masked_subimage, subcellular_mask)

                    
                    
                    subcellular_segmentations.append(segmented_image.copy())
                    subcellular_images.append(subcell_image.copy())
                    subcellular_masks.append(subcellular_mask_outputs)
                    
                    
                    # iv) Get information from segmentation
                    #These should be updating a dictionary

                    data_inputs.update(extract_area(Parameters, segmented_image))
                    
                    
                    if image_type.lower() == "anisotropy":
                        data_inputs.update(extract_anisotropy_features(Parameters, segmented_image))


                    elif image_type.lower() == "intensity":
                        data_inputs.update(extract_intensity_features(Parameters, segmented_image))


                    if data_inputs['Segmented_Percentage'] > subcellular_area_threshold_pct: #Only update if you have detected enough pixels...
                        for types in data_inputs:
                            current_dataframe.at[row_index, (t, channel, types)] = data_inputs[types]
                    else:
                        for types in data_inputs:
                            current_dataframe.at[row_index, (t, channel, types)] = np.NaN
                        
                        
                    
                    if debug_mode == True: debug_array[channel] = annotate_image(debug_array[channel], ROI, 
                                                                                 subcellular_mask_outputs[0], width = 1)
                
            overall_subcellular_masks.append(subcellular_masks)
            overall_subcellular_images.append(subcellular_images)
            overall_subcellular_segmentations.append(subcellular_segmentations)
                    
        debug_cache['Filename'] = filename
        debug_cache['Segmentation Images'] = debug_array
        debug_cache['Detection Image'] = debug_seg
        debug_cache['ROI List'] = ROI_list
        debug_cache['Cell Images'] = overall_cellular_images
        debug_cache['Subcellular Images'] = overall_subcellular_images
        debug_cache['Subcellular Segmentations'] = overall_subcellular_segmentations
        debug_cache['Cell Masks'] = overall_cellular_masks
        debug_cache['Subcellular Masks'] = overall_subcellular_masks
        debug_cache['Premask Images'] = overall_premask_images
        
        
        if debug_mode == True: export_debug_images(Parameters, current_Image, debug_seg, debug_array, ROI_list)
    
    dataframes[filename] = current_dataframe  
    
    return (dataframes, debug_cache)
    
