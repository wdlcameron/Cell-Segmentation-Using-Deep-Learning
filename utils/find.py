import numpy as np
import cv2
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters
from .ROIs import ROIs, calc_overlap


def find_local_maxima (data, neighborhood_size = 4, threshold = 700, region_size = 5, min_maxima_value = 500):
    """
    input:
    
    image = a 2D numpy matrix containing your data
    neighborhood_size = the area used to find the maximum filter (should be larger than your largest object)
    threshold = the minimum difference between your maximum and the lower value 
        in the neighbourhood (the background if the neighborhood size is large enough)
        
    region_size = the region that you should look at when calculating the average intensity for that maxima
    min_maxima_value = the minimum maxima threshold value that determines whether to keep that maxima or not
    
    
    
    output:
    The x and y coordinates of the maxima in the image (empty list if there are none...)
    """    

    #find the maxima value within a specific neighborhood size    
    data_max = filters.maximum_filter(data, neighborhood_size)
    
    #set the maxima as "true" for the pooint where this maxima is true
    maxima = (data == data_max)
    
    #find the minima for comparison
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    
    #Remove all points that are below the minimum threshold
    maxima[diff == 0] = 0


    
    #May not be necessary...
    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    

    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)//2
        y_center = (dy.start + dy.stop - 1)//2    
        
        reg_mean = np.mean(data[y_center-region_size:y_center + region_size,
                                x_center-region_size:x_center+region_size])
        if reg_mean > min_maxima_value:

            x.append(int(x_center))
            y.append(int(y_center))


    return (x, y)




def find_ROIs (image, Parameters, avg_radius = 4, thresh_tolerance = 0.5, threshold = 0, min_ROI_size = 0):
    """
    This function is responsible for finding ROIs within a cell using local maxima
    and acts as an alternative to the machine learning approach.  It can produce
    good results, but some of the parameters must be optimized for each experiment
    
    
    In particular, you need to use
    
    
    """
    
    #variables for local maxima
    
    max_neighborhood_size = Parameters.max_neighborhood_size
    max_threshold= Parameters.max_threshold
    avg_radius = Parameters.local_max_avg_radius
    thresh_tolerance = Parameters.thresh_tolerance
    IoU_match_thresh = Parameters.IoU_match_thresh
    channel_thresholds = Parameters.channel_thresholds
    min_ROI_size = Parameters.min_ROI_size
    


    #Find the x,y points of the local maxima
    seed = np.zeros(image.shape)
    x, y = find_local_maxima (image, 
                              neighborhood_size=max_neighborhood_size, 
                              threshold=max_threshold, min_maxima_value = threshold)

    overall_image = np.zeros(image.shape)


    
    ROI_List = []

    for (x_val, y_val) in (zip(x,y)):
        #Doesn't take care of the fact that y_val or x_val might be close to the edge (less than the tolerance)


        average_value = np.mean(image[y_val- avg_radius: y_val + avg_radius,
                                      x_val- avg_radius: x_val + avg_radius])





        if average_value:
            seed = np.zeros(image.shape)
            seed[y_val, x_val] = 1
            _ , mask = cv2.threshold(image,
                                     average_value * (1-thresh_tolerance),
                                     average_value*(1+thresh_tolerance),
                                     cv2.THRESH_BINARY )

            
            roi_image = ndimage.binary_propagation(seed, mask = mask)


            i, j = np.where(roi_image)
            #y_corner = min(i)
            #x_corner = min(j)
            
            roi_subimage = roi_image[min(i): max(i),
                                     min(j): max(j)]

            
            #only add ROIs that are large enough (should both axes be, or just one??)
            if (max(i)-min(i) > min_ROI_size) and (max(j)-min(j) > min_ROI_size):
                current_ROI = ROIs(x = min(j), y = min(i), confidence = None, classification = None, data = roi_subimage)

                if not ROI_List:
                    #print (current_ROI)
                    ROI_List.append(current_ROI)

                else:           
                    for compROI in ROI_List:
                        IoU_match = 0
                        IoU = calc_overlap(current_ROI, compROI)
                        if IoU>IoU_match_thresh:
                            IoU_match = 1
                            #print ("Already added a similar ROI")
                            break

                    if not IoU_match:
                        #print (current_ROI)
                        ROI_List.append(current_ROI)

                #for testing purposes
                #overall_image = np.add(overall_image, np.multiply(image, roi_image))

        #print (x_val, y_val, average_value, np.sum(roi_image), np.sum(mask), roi_subimage.shape)
        #print (x_corner, y_corner, min(j), max(j),min(i), max(i), roi_subimage.shape)


    #print (f"Numer of ROIS is {len(ROI_List)}")
    #plt.imshow(overall_image)
    
    
    return (ROI_List)
    