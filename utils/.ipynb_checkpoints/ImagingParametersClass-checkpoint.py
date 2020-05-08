#from fastai.vision import *
#from fastai.metrics import error_rate    #May not need this here...
from .fastai_helper import *
from .DeepAnalysis import deep_analysis


import imageio
import logging
import cv2
import torch
from torchvision import transforms as tf
import brambox.boxes as bbb
import lightnet as ln

import numpy as np
import re


class ImagingParameters():
    
    """
    Subclass containing the additional information found in each image.  Variables are slotted to lower 
    memory usage.

    
    """
    class ImageClass():
        __slots__ = ['directory', 'filename', 'file_path', 'reader', 'ROI_list', 'channel_offset',
                     'total_frames', 'height', 'width', 'timepoints']
        def __init__(self, directory, filename, channel_array, image_type_array, channel_thresholds):
            self.directory, self.filename = directory , filename
            self.file_path = directory/filename
            self.ROI_list = []
            self.reader = imageio.get_reader(str(self.file_path))
            self.total_frames = self.reader.get_length()
            self.height, self.width = self.reader.get_data(0).shape

            assert (len(channel_array) == len(image_type_array)), "Anisotropy and Channel arrays are not of equal length"
            
            self.channel_offset = []
            images_per_timepoint = np.sum(channel_array)
            assert self.total_frames >= images_per_timepoint, "You have declared more frames in your channel_array than are present.  Check your config file"
            self.timepoints = self.total_frames // images_per_timepoint

    class Experiment():
        __slots__ = ['path', 'name', 'files', 'models', 'results']
        def __init__(self, path):
            self.path = path
            self.name = path.name
            self.files = {}

        def add_files(self, file_dict):
            self.files.update(file_dict)

        def set_models(self, models):
            self.models = models

        def detect_models(self, detection_model):
            #'Detection', 'Filenames', 'User'
            pass

        def infer_model_from_name(self, name):
            #Can make this more sophisticated later...
            lookup = {'cyto':'Cytoplasm',
                      'mito':'Mitochondria', 
                      'br': 'None', 
                      'nuc':'Nuclear', 
                      'mem': 'Membrane', 
                      'er': 'ER',
                      'per': 'Peroxisome',
                      'whole': 'WholeCell'}
            
            
            self.models = []
            p = re.compile('\[(.*?)\]')
            string_list = p.findall(name)
            #print (self.name, string_list)
            string_list = string_list[0].split(' ')

            for string in string_list:
                lw_string = string.lower()
                if lw_string in lookup: self.models.append(lookup[lw_string])
                else: print(f'Could not find {lw_string} in lookup!')
                
        
        def get_file(self, folder_idx = 0, file_idx = 0):
            folder = list(self.files.keys())[folder_idx]
            file = list(self.files[folder].keys())[file_idx]
            
            return(folder, file)


            
   

    #Initialize the ImageParameters class
    def __init__(self, data_path, **kwargs): 
        self.data_path = data_path
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            #print (key, value, getattr(self, key))
            
            
        self._valid_image_types = ['Anisotropy', 'Intensity']
        self.channels_by_type = self.find_channels_by_type()
        self.assert_variable_sets()   #Make sure all the important variables are present




        """
        Steps to assembling an experiment:
        1) Find all files
            - Find by recursion through the root folder
        2) Split files into experiments
            - Split by folder
        3) Choose which models are associated with each experiment
            - From config file
            - From base fodler name
            - From automatic detection

        """


        #Step 1: Find all Files
        self.directory_list = self.find_files(self.data_path, {})

        #Step 2: Organize into experiments
        self.experiments = self.group_experiments_using_folders(self.directory_list)

        #Step 3: Find models for each experiment
        self.find_models_for_experiment(self.model_selection)
        

        self.dataframes = []
        self.experimental_results = []
        self.frames_per_timepoint = self.determine_frames_per_timepoint(self.channel_array)
        
        if (self.machine_learning_mode == True):
            self.initializeROINetwork()
            self.initializeSegmentationNetwork()

        if "Anisotropy" in self.image_type_array:
            self.initializeAnisotropyVariables()
              
        #Choose Analyzer (Hook function in)...
        if self.machine_learning_mode: self.analyze_image = deep_analysis
        else: self.analyze_image = image_Analysis
            



    def iterate_through_experiments(self):
        #experimental_results = {}
        for name, experiment in self.experiments.items():
            print(f'Currently analyzing the "{name}" group of directories')
            self.current_experiment = experiment
            
            
            self.load_organelle_models(experiment)            
            """#Destroy the unneeded models:
            for model in self.subcell_segmentation_models.copy():
                if model not in experiment.models: #Note, this is slow due to the fact that experiment.models is a list!!!
                    self.deleteSegmentationNetwork(model)

                
                    print (f'Deleted model {model}.  Check produces: {model in self.subcell_segmentation_models}')

            
            
            #Load the relevant models
            for organelle in experiment.models:
                if organelle not in self.subcell_segmentation_models:
                    self.loadSegmentationNetwork(organelle)
                    print (f"Loaded the {organelle} network")
            """


            self.current_image_organelle_array = experiment.models
            dataframe = self.iterate_through_files(experiment.files)
            experiment.results = dataframe
            #The experimental results dictionary will be phased out...
            #experimental_results.update({name: dataframe})

        #self.experimental_results = experimental_results
        return self.experiments
        #return experimental_results

        
        
        
    def load_organelle_models(self, experiment):
    #Destroy the unneeded models:
        for model in self.subcell_segmentation_models.copy():
            if model not in experiment.models: #Note, this is slow due to the fact that experiment.models is a list!!!
                self.deleteSegmentationNetwork(model)
                print (f'Deleted model {model}.  Check produces: {model in self.subcell_segmentation_models}')

        #Load the relevant models
        for organelle in experiment.models:
            if organelle not in self.subcell_segmentation_models:
                self.loadSegmentationNetwork(organelle)
                print (f"Loaded the {organelle} network")
                
        
        

    def analyze_single_file(self, experiment, folder, file):
        analyze_image = self.analyze_image
        
        #Prepare the correct parameters
        self.current_image_organelle_array = experiment.models
        self.load_organelle_models(experiment)
        
        #Create an image class instance for the file of interest
        current_Image = self.ImageClass(folder, file, self.channel_array,
                                        self.image_type_array, self.channel_thresholds)

        #Analyze the image
        dataframe, debug_cache = analyze_image(current_Image = current_Image,
                                              Parameters = self,
                                              dataframes = {})
        
        return (dataframe, debug_cache)
        
        
        
        
        
        
        
        
        
        
        

    def find_models_for_experiment(self, model_code):
        for name, experiment in self.experiments.items():
            if model_code == 'User':
                experiment.set_models(self.image_organelle_array)

            elif model_code == 'Detection':
                experiment.detect_models()

            elif model_code == 'Filenames':
                experiment.infer_model_from_name(experiment.name)
            
            else:
                raise Exception(f"You have provided an invalid detection mode: {model_code}")


    def determine_frames_per_timepoint(self, channel_array):
        return (np.sum(channel_array))
        
    def assert_variables(self, variables, identifier = ""):
            for attribute in variables:
                assert hasattr(self, attribute), f"Error: {identifier} attribute {attribute} was not set.  Check your config file"

                
    def assert_variable_sets(self):
        #Assert that the core variables are all set correctly.
        core_variables = ['channel_array', 'channel_thresholds', 'suffix', 'machine_learning_mode', 'ROI_match_threshold']
        print (core_variables)
        self.assert_variables(core_variables, "Core")
        
        if (self.machine_learning_mode == True):
            #variables for the yolo network
            network_variables = ['classes', 'network_size', 'labels', 'conf_thresh', 'nms_thresh', 'use_cuda']
            self.assert_variables(network_variables, "Network")
            
            #variables for the Unet segmentation
            segmentation_variables = ['cell_segmentation_weights', 'subcell_segmentation_model_paths', 'mask_detection_channels']
            self.assert_variables(segmentation_variables, "Segmentation")
            
        else:
            automatic_detection_variables = ['max_neighborhood_size', 
                                             'max_threshold', 
                                             'local_max_avg_radius', 
                                             'thresh_tolerance', 
                                             'min_ROI_size', 
                                             'IoU_match_thresh']
            self.assert_variables(automatic_detection_variables, "Local Max Detection")
        
    def print_experiments(self, show_every = 10):
        for i, (name, experiment) in enumerate(self.experiments.items()):
            sub_paths = experiment.files
            print(f'Experiment {i} is: {name}')
            print (f'This experiment will use the following models: {experiment.models}')
            for root_dir, sub_dirs in sub_paths.items():
                print (f'\t|-> {root_dir}')
                for i, sub_dir in enumerate(sub_dirs):
                    if not (i%show_every): 
                        print (f'\t\t|-> {sub_dir}')
                        if not show_every == 1: print(f'\t\t|->(...) x {show_every-1}')
                        


    def print_experiment_summary(self, show_every = 10):
        print ("Your Experiment will be organized as follows:\n")       
        self.print_experiments(show_every)

        print('\n\nYour cell masking network will use the following codes for its deep analysis:')
        for det_channel, merge_channels in zip(self.processed_mask_detection_channels, self.mask_merge_channels_array):
            print (f"\tCode: {det_channel}, which will merge the following channels: {merge_channels}")




    def find_files (self, path, directory_list = {}, double_subpath = True):
        for subpath in path.iterdir():
            if subpath.is_file() and subpath.suffixes == ['.ome', '.tif']:
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

            elif subpath.is_dir():
                directory_list = self.find_files(subpath, directory_list, double_subpath)

        return (directory_list)

    
    def group_experiments_using_folders(self, directory_lists):
        experiments = {}
        for items, directory_list in sorted(directory_lists.items()):           
            subdirectory = items.relative_to(self.data_path)
            if subdirectory.parent == Path(""): #If it is in the main folder, it should be its own experiment
                current_experiment = self.Experiment(path = items)
                current_experiment.add_files({items:directory_list})
                experiments[subdirectory] = current_experiment
            else:
                superdirectory = subdirectory.parent
                if superdirectory not in experiments: #Create a new dictionary if it doesn't exist
                    current_experiment = self.Experiment(path = superdirectory)
                current_experiment.add_files({items:directory_list})
                experiments[superdirectory] = current_experiment 
        return (experiments)
    
    
    

    
    
    def iterate_through_files(self, directory_list, dataframes = {}):
        """
        The purpose of this function is to iterate through all of the files in the directory list and then
        send them to the relevant analysis program.  It assumes that this is a single group of image sets 
        that were taken using the same cells 
        """ 
        
        #dataframes = self.dataframes

        suffix = self.suffix
        debug_mode = self.debug_mode
        machine_learning_mode = self.machine_learning_mode
        dataframes = {}
        analyze_image = self.analyze_image
        
        for root_dir in sorted(directory_list):
            print("Root", root_dir)
            
            #note: there will usually only be one subdirectory per root_dir
            for files in sorted(directory_list[root_dir]):                  
                
                current_Image = self.ImageClass(root_dir,
                                          files, self.channel_array,
                                          self.image_type_array, self.channel_thresholds)

                
                #Analyze the image

                dataframes, debug_cache = analyze_image(current_Image = current_Image,
                                              Parameters = self,
                                              dataframes = dataframes)


        #consolidate the dataframes
        dataframes = self.consolidate_dataframes(dataframes)

        #self.dataframes = dataframes
        return (dataframes)
    
    
    def choose_file(self, exp_idx = 0, folder_idx = 0, file_idx = 0):
        experiment = list(self.experiments.values())[exp_idx]
        #folder = list(experiment.files.keys())[folder_idx]
        #file = list(experiment.files[folder].keys())[file_idx]
        folder, file = experiment.get_file(folder_idx, file_idx)
        return experiment, folder, file

            
            
    def consolidate_dataframes(self, dataframes):   #Combine all dataframes into one
        final_dataframe = pd.DataFrame([])
        for filenames, df_interest in dataframes.items():
            if len(final_dataframe)==0:
                #print ("need a new dataframe")
                final_dataframe = df_interest
            else:
                #print ("Appending")
                final_dataframe = final_dataframe.append(df_interest, ignore_index = True, sort = False)
        return (final_dataframe)
    
    
    
    
    def initializeAnisotropyVariables(self):
        #Assert that the correct variables are defined
        ani_variables = ['numerical_aperture', 'index_of_refraction', 'magnification', 'gFactor']
        self.assert_variables(ani_variables, "Anisotropy")

        asin = math.asin(self.numerical_aperture/self.index_of_refraction) 
        cos = math.cos(asin)

        kA = (2.0 - 3.0 * cos + cos * cos * cos) / (6.0 * (1.0 - cos))
        kB = (1.0 - 3.0 * cos + 3.0 * cos * cos - cos * cos * cos) / (24.0 * (1.0 - cos))
        kC = (5.0 - 3.0 * cos - cos * cos - cos * cos * cos) / (8.0 * (1.0 - cos))

        self.kA = kA
        self.kB = kB
        self.kC = kC
        print (kA, kB, kC)
    
    
    def initializeROINetwork(self):
        #These will eventually be in a configuration file
        self.log = logging.getLogger('lightnet.detect')
              

        #Use the GPU if available and wanted
        self.device = torch.device('cpu')
        if self.use_cuda:
            if torch.cuda.is_available():
                self.log.debug('CUDA enabled')
                self.device = torch.device('cuda')
            else:
                self.log.error('CUDA not available')
        
        self.network = ln.models.Yolo(self.classes,
                                      conf_thresh = self.conf_thresh,
                                      nms_thresh = self.nms_thresh,)
    
        self.network.postprocess.append(ln.data.transform.TensorToBrambox(self.network_size, self.labels))
        self.network.load(self.weights_path)
    
        self.network = self.network.to(self.device)
        
        
    
    def find_network_size(self, network):
        #This will be replaced by a proper function later...
        test = open_image_array(np.ones((3,11,11)))
        _, o2, _  = network.predict(test)
        return (o2.shape[1])
    
    
        
    def initializeSegmentationNetwork(self):
        
        """Once this works, play around with changing the size here.  Can you load 
        the existing weights into a larger or smaller network???
        """
        
        #This will keep all the fastai models on the CPU (using less memory) - we can change to GPU if it is too slow...
        if not self.use_GPU_for_detection:
            defaults.device = torch.device('cpu')


        model_path = Path.cwd()/self.mask_path
        

        self.mask_merge_channels_array = self.find_mask_merge_channels()

        

        
        
        self.cell_segmentation_model = load_learner(model_path, self.cell_segmentation_weights)
        self.cell_segmentation_size = self.find_network_size(self.cell_segmentation_model)  #Run an image of ones through the network to determine what the size is..
        
        
        self.subcell_segmentation_models = {}
        #self.subcell_segmentation_sizes = []

        """for model_name, model_filename in self.subcell_segmentation_model_paths.items():
            loaded_model = load_learner(model_path, model_filename)
            model_size = self.find_network_size(loaded_model)
            self.subcell_segmentation_models[model_name] = (loaded_model, model_size)
        """



    def loadSegmentationNetwork(self, model_name):
        skip_codes = {'None', 'none', ''}

        if model_name not in skip_codes:
        
            model_path = Path.cwd()/self.mask_path
            model_filename = self.subcell_segmentation_model_paths[model_name]

            loaded_model = load_learner(model_path, model_filename)
            model_size = self.find_network_size(loaded_model)
            self.subcell_segmentation_models[model_name] = (loaded_model, model_size)

    def deleteSegmentationNetwork(self, model_name):
        model, _ = self.subcell_segmentation_models[model_name]
        model.destroy()
        del self.subcell_segmentation_models[model_name]


    def clearSegmentationNetworks(self):
        pass
    
    def find_mask_merge_channels(self):

        #print(self.channels_by_type, self.mask_detection_channels)
        
        mask_merge_channels_array = []
        processed_mask_detection_channels = []
        for channel_code in self.mask_detection_channels:
            if channel_code < 100:
                mask_merge_channels_array.append(None)
                processed_mask_detection_channels.append(channel_code)


            elif channel_code == 100:  #Merge all fluorescence
                mask_merge_channels_array.append(self.channels_by_type[self.FLUORESCENCE_NAME])
                processed_mask_detection_channels.append(100)

            elif channel_code == 104: #Merge all brightfield
                mask_merge_channels_array.append(self.channels_by_type[self.BRIGHTFIELD_NAME])
                processed_mask_detection_channels.append(100)

            elif channel_code == 111: #Merge first of all fluorescence
                cumulative_channels = 0
                chan_array = []
                for img_type, channels in zip (self.image_type_array, self.channel_array):
                    if img_type in self.FLUORESCENCE_CHANNELS:
                        chan_array.append(cumulative_channels)
                    cumulative_channels += channels
                mask_merge_channels_array.append(chan_array)
                processed_mask_detection_channels.append(100)

            elif channel_code == 110: #Blank Channel
                mask_merge_channels_array.append(None)
                processed_mask_detection_channels.append(110)

            else:
                raise Exception(f"Do not know how to process code {channel_code}")
        self.processed_mask_detection_channels = processed_mask_detection_channels
        return (mask_merge_channels_array)





    def find_channels_by_type(self):
        cumulative_channels = 0    
        channels_by_type = {}
        for img_type, channels in zip (self.image_type_array, self.channel_array):
            ch_array = [x for x in range(cumulative_channels, cumulative_channels+channels)]
            channels_by_type[img_type] = (channels_by_type[img_type]+ch_array) if img_type in channels_by_type else ch_array   
            cumulative_channels += channels
            
        fluorescence_channels = self.FLUORESCENCE_CHANNELS

        channels_by_type[self.FLUORESCENCE_NAME] = []

        for ch in fluorescence_channels:
            if ch in channels_by_type: channels_by_type[self.FLUORESCENCE_NAME]+= channels_by_type[ch]

        return(channels_by_type)

                    


