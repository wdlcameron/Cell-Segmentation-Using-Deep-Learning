import os

def parse_text_file(text_path):
    
    skip_characters = ["'", "#", "\n"]
    replace_characters = ['[', ']', "'", '"']
    
    for char in ['/', '\\']:
        text_path = text_path.replace(char, os.sep)

    print (text_path)




    config_variables = {}
    with open(text_path, 'r') as f:
        for lines in f:
            if lines[0] not in skip_characters:           
                param_input = "".join(lines.split())
                variable, value = param_input.split('=')
                #print (variable, value, param_input)
                config_variables.update({variable: value})


    float_variables = {'numerical_aperture',
                       'index_of_refraction',
                       'magnification',
                       'gFactor',
                       'max_filter_region',
                       'max_threshold', 
                       'conf_thresh', 
                       'nms_thresh', 
                       'thresh_tolerance', 
                       'IoU_match_thresh', 
                       'ROI_match_threshold', 
                       'CELL_MASKING_AREA_THRESHOLD_PCT', 
                       'SUBCELL_AREA_THRESHOLD_PCT'}
    
    int_variables = {'classes', 
                     'ROI_expansion_factor',
                     'local_max_avg_radius', 
                     'max_neighborhood_size', 
                     'min_ROI_size', 
                     'segmentation_network_size'}
    
    binary_variables = {'debug_mode', 
                        'root_dir_same_treatment', 
                        'machine_learning_mode', 
                        'use_cuda', 
                        "segmentation_outputs",
                        'use_GPU_for_detection', 
                        'manual_boxing'}
    
    array_variables = {'channel_thresholds': "int", 
                       'image_organelle_array':'str',
                       'channel_array': 'int',
                       'image_type_array': 'str', 
                       'network_size':'int', 
                       'labels': 'str',
                       'detection_channels': 'int',
                       'mask_detection_channels': 'int',
                       'FLUORESCENCE_CHANNELS':'str'}
    
    path_variables = {'weights_path', 'mask_path'}

    dictionary_variables = {'subcell_segmentation_model_paths':'str'}
    
    #Don't need this for anthing
    string_variables = {'suffix', 
                        'cell_mask_weights', 
                        'BRIGHTFIELD_NAME',
                        'FLUORESCENCE_NAME', 
                        'model_selection'}
    
    
    for variables in config_variables:
        
        #Convert Floats
        if variables in float_variables:
            config_variables[variables] = float(config_variables[variables])
            
        elif variables in int_variables:
            config_variables[variables] = int(config_variables[variables])

        #Convert Binary Variables
        elif variables in binary_variables:
            config_variables[variables] = (config_variables[variables].lower() == "true")

        #Convert to arrays with the appropriate variable type
        elif variables in array_variables:
            var_type = array_variables[variables]
            for c in replace_characters:
                config_variables[variables] = config_variables[variables].replace(c,"")
            config_variables[variables] = config_variables[variables].split(',')
            if var_type.lower() == "int":
                config_variables[variables] = [int(x) for x in config_variables[variables]]
                
            #elif var_type.lower() == "str":
            #    config_variables[variables] = [x.lower for x in config_variables[variables]]
                
        
        
        
        
        elif variables in path_variables:
            for char in ['/', '\\']:
                config_variables[variables] = config_variables[variables].replace(char, os.sep)
                
        elif variables in string_variables:
            pass

        elif variables in dictionary_variables:
            var_type = dictionary_variables[variables]
            for c in replace_characters:
                config_variables[variables] = config_variables[variables].replace(c,"")
            temp_dict = {}
            for items in config_variables[variables].split(','):
                key, value = items.split(':')
                temp_dict.update({key:value})
            config_variables[variables] = temp_dict

            
        
        else:
            print (f'unknown variables found: {variables}')
    
    return config_variables