import os
import math
import sys
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import imageio

import skimage

import xml.etree.cElementTree as ET

from fastai.vision import *
from fastai.metrics import error_rate

def find_files(path, directory_list = {}, double_subpath = True):

    
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
            directory_list = find_files(subpath, directory_list, double_subpath)
      
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

    subsegmentation = Parameters["subsegmentation"]
  
    
    for root_dir in directory_list:
        #note: there will usually only be one subdirectory per root_dir
        annotation_dir = root_dir/'Annotations'
        image_dir = root_dir/'JPEGImages'
        
        print (annotation_dir)
        print (root_dir)
        
        for files in directory_list[root_dir]:

            #if files.suffix == (".tif"):  #This is unneccessary since we already did the check before...

            ROIs = read_ROI_File(root_dir/files)
            #Process the image if there is an associated ROI file
            #print (ROIs)
            if (not ROIs.empty):
                ROIs = process_ROI_dataframe(ROIs)
                segment_ROI_image(Parameters, ROIs, root_dir, files)

                if subsegmentation == True:
                    subsegment_image(Parameters, ROIs, root_dir, files)

    
    return (dataframes)
            

def segment_ROI_image(Parameters, ROIs, root_dir, sub_path):
    
    """
    Use this to create the segmentations around each ROI (more intact cells) used to train the YOLO Component of the dataframe
    

    
    """
    ch = Parameters["channels"]
    subimage_height = Parameters["subimage_height"] 
    subimage_width = Parameters["subimage_width"]


    print (root_dir, sub_path)
    print("Segmenting around the ROIs image.... :)")    

    
   
    
    filepath = root_dir/sub_path
    annotations_dir = root_dir/'Annotations'/sub_path.parent
    image_dir = root_dir/'JPEGImages'/sub_path.parent
    
    print ("IM", image_dir)
    
    
    for dirs in [annotations_dir, image_dir]:
        if (not os.path.isdir(dirs)):
            os.makedirs(dirs)
    
    
    #Preprocess the image
    
    reader = imageio.get_reader(str(filepath))
    image_open = np.asarray(reader.get_data(ch[0]))
    image_para = np.asarray(reader.get_data(ch[1]))
    image_perp = np.asarray(reader.get_data(ch[2]))
    
    
    img = np.dstack([image_open, image_para, image_perp])
    img = img/np.amax(img)        
    img = img*255
    img = img.astype('uint8')
    plt.imshow(img)
    
    height, width, depth = img.shape
    print ("Shape", img.shape)
    print (ROIs)
    
    
    for i in range(len(ROIs)):
        x_min = int(ROIs.loc[i, 'xmin'])
        x_max = int(ROIs.loc[i, 'xmax'])
        y_min = int(ROIs.loc[i, 'ymin'])
        y_max = int(ROIs.loc[i, 'ymax'])
        
        
        
        
        #x_length = x_max - x_min
        #y_length = y_max - y_min
        
        
        #Padding can be negative!
        #x_pad = (subimage_width - x_length)//2
        #y_pad = (subimage_height - y_length)//2
        
        x_centroid = (x_max + x_min)//2
        y_centroid = (y_max + y_min)//2
        
        print (f"Stats: X:{x_min}, {x_max}, {x_centroid} Y:{y_min}, {y_max}, {y_centroid}")

        xmin = max(0, (x_centroid - subimage_width//2))
        xmax = min(width, (x_centroid + subimage_width//2))
        ymin = max(0, (y_centroid - subimage_height//2))
        ymax = min(height, (y_centroid + subimage_height//2))
        
        subimage = img[ymin:ymax, xmin:xmax, :]

        subROIs = ROIs[(ROIs['X']>xmin) & 
                       (ROIs['X']<xmax) & 
                       (ROIs['Y']>ymin) & 
                       (ROIs['Y']<ymax)].copy()


        print ("Stats:", "X", xmin, xmax, "Y", ymin, ymax, subimage.shape, len(subROIs))

        #If ROI list is not empty           
        if len(subROIs)>0:

            #mod ROIs to fit the new size
            subROIs['xmin'] = subROIs['xmin'] - xmin
            subROIs['xmax'] = subROIs['xmax'] - xmin
            subROIs['ymin'] = subROIs['ymin'] - ymin
            subROIs['ymax'] = subROIs['ymax'] - ymin

            #Check for any truncations
            subROIs['Truncated'] = ((subROIs['xmin']<0) | (subROIs['xmax']>xmax) | 
                                    (subROIs['ymin']<0) | (subROIs['ymax']>ymax))


            #print (i, j, xmin, xmax, ymin, ymax, len(subROIs))
            print (subROIs)

            #Save the jpeg files
            JPEG_filename = image_dir/sub_path.name.replace('.ome.tif', f'{i}.jpg')
            imageio.imwrite(str(JPEG_filename), subimage)
            

            #Output the labels
            labels_filename = annotations_dir/sub_path.name.replace('.ome.tif', f'{i}--labels.xml')
            labels = {'Height': subimage.shape[0], 
                      'Width': subimage.shape[1], 
                      'Filename' : (sub_path.name.replace('.ome.tif', f'{i}.jpg')) , 'Folder': str(sub_path.parent)}          
            output_labels (labels, subROIs, labels_filename)
                
    
    return(None)

def subsegment_image(Parameters, ROIs, root_dir, sub_path):
    
    
    """
    Use this module to prepare the images to help create ground truths of the cells
    
    
    
    
    """
    
    channels = Parameters["subChannels"]
    suffix = Parameters["suffix"]
    move_to_front = Parameters['move_to_front']
    
    print (root_dir, sub_path)
    print("subSegmenting image.... :)")
    
    ground_truths = Parameters["ground_truths"]
    ground_truth_suffix = Parameters["ground_truth_suffix"]

    black_masks = Parameters["black_masks"]
    
    
   
    
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
        img = np.asarray(reader.get_data(channel))

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
            print(f"swapping {i} with {channel}")
            full_image[i,:,:] = full_image[channel,:,:]
    
    
    
    if ground_truths == True:
        gt_filepath = filepath.parent/filepath.name.replace(suffix, ground_truth_suffix)
        #filepath.replace(suffix, ground_truth_suffix)
        print ("GT", gt_filepath)
        
        #gt_img = np.asarray(bioformats.load_image(gt_filepath, c = 0, z=0, t=0, rescale=False))
        gt_img = imageio.imread(gt_filepath)
        
        gt_img *= 255
        #gt_img = np.dstack([gt_img, gt_img, gt_img])
    
    

    print (ROIs)
    
    
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

        
        
        
        
        print (merged_img.shape, x_min, x_max, y_min, y_max, subJPEGimage.shape, subimage.shape,) 
        
        
        
        
        if ground_truths == True:
             

            gt_subimage = gt_img[y_min:y_max, x_min: x_max].copy()
            print (gt_img.shape, x_min, x_max, y_min, y_max, gt_subimage.shape,) 
            imageio.imwrite(str(sub_gt_dir/sub_path.name.replace(suffix, f'{i}.png' )), gt_subimage)

                
               
   
    
    return(None)

def read_ROI_File(filepath):
    
    text_path = str(filepath).replace(".ome.tif", "--labels.txt")
    print (text_path)
    
    if os.path.isfile(text_path):
        print ("Exists")
        ROI_List = pd.read_csv(text_path, index_col = False)
        #print (ROI_List)
        
        
    else:
        print ("No ROI File")
        ROI_List = pd.DataFrame()
    
    print (filepath)


    
    
    
    
    #except:
    #    ROI_List = None
    
    
    return (ROI_List)

def process_ROI_dataframe(ROIs):    
    
    
    #Switch X and Y for python standards
    
    ROIs['xmin'] = ROIs['X'] - ROIs['W']/2
    ROIs['xmax'] = ROIs['X'] + ROIs['W']/2
    ROIs['ymin'] = ROIs['Y'] - ROIs['H']/2
    ROIs['ymax'] = ROIs['Y'] + ROIs['H']/2
    
    
    
    
    #print (ROIs)
    
    return (ROIs)

def output_labels(Labels, subROIs, filepath):
    
    
    """
    Outputs the text labels into the VOC Pascal format for Lightnet to use
    
    Inputs:
    Filename
    subROIs - List of ROIs used to 
       
    
    
    
    """
    
    ROIs = subROIs.copy()
    #ROIs = ROIs.rename(columns={'xmin':'ymin', 'ymin':'xmin', 'xmax':'ymax', 'ymax':'xmax'})
    
    output_path = filepath

    objects = ['aeroplace', 'cat', 'sheep']
    
    
    annotation = ET.Element('annotation')
    tree = ET.ElementTree(annotation)


    folder = ET.SubElement(annotation, 'folder')
    folder.text = Labels['Folder']
    
    filename = ET.SubElement(annotation, 'filename')
    filename.text = Labels['Filename']

    source = ET.SubElement(annotation, 'source')
    source_database = ET.SubElement(source, 'database')
    source_annotation = ET.SubElement(source, 'annotation')
    source_image = ET.SubElement(source, 'image')
    source_flickrid = ET.SubElement(source, 'flickrid')

    owner = ET.SubElement(annotation, 'owner')
    owner_flickrid = ET.SubElement(owner, 'flickrid')

    
    
    object_subelements = ['name', 'pose', 'truncated', 'difficult', 'bndbox']
    bndbox_subelements = ['xmin', 'ymin', 'xmax', 'ymax']
    
    
    
    
    

    object_count = len(ROIs)
    print ("Object count is", object_count)
    Objects = []


    for i in ROIs.index:
        dictionary = {}

        dictionary['object'] = ET.SubElement(annotation, 'object')


        dictionary['name'] = ET.SubElement(dictionary['object'], 'name')
        dictionary['name'].text = str(objects[int(ROIs.loc[i, 'Class'])])
        dictionary['pose'] = ET.SubElement(dictionary['object'], 'pose')
        dictionary['truncated'] = ET.SubElement(dictionary['object'], 'truncated')
        dictionary['difficult'] = ET.SubElement(dictionary['object'], 'difficult')
        dictionary['bndbox'] = ET.SubElement(dictionary['object'], 'bndbox')

        dictionary['bndbox'].tail = '\n'
        
        
        
        
        
        for element in bndbox_subelements:
            dictionary[element] = ET.SubElement(dictionary['bndbox'], element)
            dictionary[element].text = str(int(ROIs.loc[i, element]))
        
        
        
        

        Objects.append (dictionary)



    tree.write(output_path)
    
    return (None)



    #Create the pickle file for a single folder

def identify(xml_file):
    root = ET.parse(xml_file).getroot()
    folder = root.find('folder').text
    filename = os.path.splitext(root.find('filename').text)[0]
    return f'{folder}{os.sep}{filename}'


def find_subdirectories (PATH, FOLDER_ID, directories = []):
    #print (directories)
    for folders in os.listdir(PATH):
        
        if os.path.isdir(f"{PATH}{os.sep}{folders}"):
            #print ("Sub folders", folders)
            if folders == FOLDER_ID:
                print ("Found One")
                directories.append(f"{PATH}")
            else:
                find_subdirectories(f"{PATH}{os.sep}{folders}", FOLDER_ID, directories)
                
    return directories

def get_filenames(path, suffix, filenames = []):
    files = os.listdir(path)
    for file in files:
        if os.path.isdir(f"{path}{os.sep}{file}"):
            get_filenames(f"{path}{os.sep}{file}", suffix, filenames)
        elif file.endswith(suffix):
            filenames.append(f"{path}{os.sep}{file}")
    return filenames
            
    
def get_datasets(directories, x_suffix, y_suffix, input_folder, output_folder):
    x_data = []
    y_data = []
    
    
    
    for folders in directories:
        data_filenames = get_filenames(f"{folders}{os.sep}{input_folder}", x_suffix, filenames = [])
        for x_file in data_filenames:
            y_file = x_file.replace(input_folder, output_folder)
            y_file = f"{y_file[:-(len(x_suffix))]}{y_suffix}"
            #print (y_file)
            if os.path.isfile(y_file):
                #print ("Got to here")
                x_data.append(x_file)
                y_data.append(y_file)
                
    return (x_data, y_data)
        


def create_training_pickle(ROOT):

    DEBUG = True        # Enable some debug prints with extra information
    #ROOT = '../data'    # Root folder where the VOCdevkit is located

    
    #Can implement findfiles
    

    anno_directory = f"{ROOT}{os.sep}Annotations"

    #find all files for the training set...
    print('Getting training annotation filenames')
    anno_list = []

    
    train = []
    
    
    
    #Note, this function only goes one directory deep...
    
    for folders in [f'{anno_directory}{os.sep}{x}' for x in os.listdir(anno_directory) if os.path.isdir(f'{anno_directory}{os.sep}{x}')]:
        file_list = os.listdir(folders)   
        train += [f'{folders}{os.sep}{x}' for x in file_list if x.endswith('.xml')]

    
    print (train)

    if DEBUG:
        print(f'\t{len(train)} xml files')
        #for items in train:
        #print (items)

    print('Parsing training annotation files')
    #train_annos = bbb.parse('anno_pascalvoc', train, identify)
    train_annos = bbb.parse('anno_pascalvoc', train, identify)


    # Remove difficult for training
    for k,annos in train_annos.items():
        for i in range(len(annos)-1, -1, -1):
            if annos[i].difficult:
                del annos[i]

    print('Generating training annotation file')
    bbb.generate('anno_pickle', train_annos, f'{ROOT}/train.pkl')

    print("Done!!")


def create_training_pickle_from_folders(ROOT, subdirectories, label_id, image_id):

    DEBUG = True        # Enable some debug prints with extra information
    #ROOT = '../data'    # Root folder where the VOCdevkit is located


    train = []
    anno_list = []
    
    train_annos = {}
    for dir in subdirectories:
        anno_directory = f"{ROOT}{os.sep}{dir}{os.sep}Annotations"

        #find all files for the training set...
        print(f'Getting training annotation filenames from {anno_directory}')


        #Note, this function only goes one directory deep right now...

        for sub_folder in [x for x in os.listdir(anno_directory) if os.path.isdir(f'{anno_directory}{os.sep}{x}')]:
            
            print (sub_folder)
            
            full_subpath = f'{anno_directory}{os.sep}{sub_folder}'
            
            #print ("Filenames are...", [x.replace(ROOT, "") for x in get_filenames(folders, '.xml', [])])
            file_list = os.listdir(full_subpath)   
            train = [f'{full_subpath}{os.sep}{x}' for x in file_list if x.endswith('.xml')]
            #print (train)
            
            print (f"Up to {len(train)} files now")
        
        
        
        #Can replace with a lambda statement...
            def identify_temp(xml_file):
                #print (xml_file)
                root = ET.parse(xml_file).getroot()
                folder = root.find('folder').text
                filename = os.path.splitext(root.find('filename').text)[0]
                return f'{dir}{os.sep}{image_id}{os.sep}{sub_folder}{os.sep}{filename}'

            #add the new values to the dictionary
            train_annos.update(bbb.parse('anno_pascalvoc', train, identify_temp))
        




    if DEBUG:
        print(f'\t{len(train)} xml files')
        #for items in train:
        #print (items)

    print('Parsing training annotation files')
    #train_annos = bbb.parse('anno_pascalvoc', train, identify)
    #train_annos = bbb.parse('anno_pascalvoc', train, identify_two)
    
    
    print (train_annos)


    # Remove difficult for training
    for k,annos in train_annos.items():
        for i in range(len(annos)-1, -1, -1):
            if annos[i].difficult:
                del annos[i]

    print('Generating training annotation file')
    bbb.generate('anno_pickle', train_annos, f'{ROOT}/train.pkl')

    print("Done!!")