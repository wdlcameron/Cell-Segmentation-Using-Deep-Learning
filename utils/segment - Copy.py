import numpy as np
from skimage.measure import block_reduce

from .fastai_helper import open_image_array



def segment_ROIs(data, network, sz, output_all = False):
    
    """
    Steps: 
    
    We import an h x w x 3 shaped vector
    We want to end up with a 3 x sz x sz shaped vector
    
    Concern: artifacts created by rescaling the segmentation up or down
    
    D 1) determine if you need to scale down and by how much
    D 2) Pad the image to the bottom and left so that scaling down is less lossy
    D 3) Block reduce to be within sz x sz
    D 4) Normalize the image (divide by amax)
    D 5) Swap the dimensions so that we have a 1 x 3 x h x w array
    D 6) Place the array in the larger 1 x 3 x sz x sz array
    D 7) Feed it into the network
    D 8) Expand the output by the scaling factor (use np.kron)
    D 9) Output the original size (h x w) (ie. do not include the padding)
    
    """
    
    if output_all: print (f"Original shape is {data.shape} and network size is {sz}")
    
    h, w, c = data.shape


    #Determine how to modify the image
    if max(h/sz, w/sz) >1:      #If either axis is larger than the size, scale down
        scale_factor = 1 + max(h//sz, w//sz)
    elif min([np.log2(sz/x)//1 for x in (h,w)]) > 0: #If the image can be doubled, do it
        scale_factor = 1/2**min([np.log2(sz/x)//1 for x in (h,w)])
    else:     #Otherwise do nothing
        scale_factor = 1    
    #Can also write this as scale_factor = (1 + max(h//sz, w//sz))/min([np.log2(sz/x)//1 for x in (h,w)]) but this is simpler


    if output_all: print (f"The scale factor is {scale_factor}")
    if output_all: print (f"The starting shape is {data.shape}")



    if scale_factor > 1:
        padded_array = np.zeros((h+h%scale_factor,
                                w+w%scale_factor,
                                3))
        padded_array[:h,:w,:] = data
        data = block_reduce(padded_array, (scale_factor, scale_factor, 1), func = np.mean)

    elif scale_factor <1:
        data = np.kron(data, np.ones((int(1/scale_factor), int(1/scale_factor), 1)))



    

    data = np.moveaxis(data, -1, 0)     #New shape is (3, sz, sz)
    
    new_c, new_h, new_w = data.shape
  
    if output_all: print (f"Before the input array, shape is {data.shape}")

    
    input_array = np.zeros((3,sz,sz))
    input_array[:, :new_h, :new_w] = data.copy()

    if output_all: print ("In the input array", input_array.shape)
    
    
    input_array =  open_image_array(input_array)
    if output_all: print ("After Open Image", input_array.shape)
    (o1, o2, o3) = network.predict(input_array)
    
    prediction = np.asarray(o2.squeeze())
    
    if prediction.ndim == 2: prediction = np.expand_dims(prediction, 0)
    
    
    if output_all: print (f"The shape of the prediction is {prediction.shape}")
    
    
    if scale_factor > 1:
        prediction = np.kron(prediction, np.ones((1, scale_factor, scale_factor)))

    elif scale_factor < 1:
        prediction = block_reduce(prediction, (int(1/scale_factor), int(1/scale_factor), 1), func = np.mean)
    
    if output_all: print("Right before segmentation the results are:", prediction.shape)
    final_segmentation = prediction[0, :h, :w]>0
    
    if output_all: print ("The shape of the final segmentation is", final_segmentation.shape)
    if output_all: print ("The maximum value of the prediction is", np.amax(prediction[0, :h, :w]))
    if output_all: print ("The mean value is: ", np.mean(prediction[0, :h, :w]))

    return (final_segmentation)
    

