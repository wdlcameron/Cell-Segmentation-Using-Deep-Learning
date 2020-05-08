import os
import pandas as pd
import numpy as np

class ROIs():
    __slots__ = ['x','y','data','classification','confidence','x_length','y_length','segmentation','subcell_segmentation']
    def __init__(self, x, y, data, classification=None, confidence=None):
        """
            ______x_______
            |            |
           y|            |
            |            |
            |____________|
        
        
        Note: this standard was adopted as it conforms to both bramboxes and CV2 drawing formats.
        For numpy arrays, axis 0 is Y and axis 1 is X.  Therefore indexing should be
        np.array[y1:y1, x1:x2] when dealing with the images as numpy arrays
        
        """       
        
        self.x = x
        self.y = y
        self.data = data
        self.classification = classification
        self.confidence = confidence
        
        y_length, x_length = data.shape
        
        self.x_length = x_length
        self.y_length = y_length

        self.segmentation = None
        self.subcell_segmentation = []

    def add_segmentation(self, segmentation_array):
        self.segmentation = segmentation_array
    
    def add_subsegmentation(self, segmentation_array):
        self.subcell_segmentation.append(segmentation_array)
    
        
    
#Should make this a subfunction of the ROI class



def calc_overlap(ROI1, ROI2, threshold = 0.8):
    """
    Calculate the IoU for two ROIs
    
    
    """
    x1, y1, x1Len, y1Len = ROI1.x, ROI1.y,  ROI1.x_length, ROI1.y_length
    x2, y2, x2Len, y2Len = ROI2.x, ROI2.y,  ROI2.x_length, ROI2.y_length
    
    
    box1_Area = (x1Len * y1Len)
    box2_Area = (x2Len * y2Len)
    #print ("b1 Area", box1_Area, "b2 Area", box2_Area)
    
    
    inter_Area = max(0, (min(x1+x1Len, x2+x2Len)-max(x1, x2)))*max(0, (min(y1+y1Len, y2+y2Len) - max(y1, y2)))
    
    
    
    if (box1_Area or box2_Area):
        IoU = inter_Area / (box1_Area + box2_Area - inter_Area + 0.001)
    else:
        IoU = 0
    
    
    
    
    return (IoU) 



def read_ROI_File(filepath):
    
    text_path = str(filepath).replace(".ome.tif", "--labels.txt")
    #print (text_path)
    
    if os.path.isfile(text_path):
        #print ("Exists")
        ROI_List = pd.read_csv(text_path, index_col = False)
        #print (ROI_List)
        
        
    else:
        #print ("No ROI File")
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

def create_ROI_list(ROI_list):
    processed_ROI_list = []
    for x,y, w, h in zip(ROI_list['xmin'], ROI_list['ymin'], ROI_list['W'], ROI_list['H']):
        data = np.zeros((int(h),int(w)))
        processed_ROI_list.append(ROIs(int(x), int(y), data))
    return processed_ROI_list

def create_ROI_list_from_file(filepath):
    ROI_list = read_ROI_File(filepath)
    if (not ROI_list.empty):
        ROI_list = process_ROI_dataframe(ROI_list)
        ROI_list = create_ROI_list(ROI_list)

    else:
        ROI_list = []

    return (ROI_list)
        
