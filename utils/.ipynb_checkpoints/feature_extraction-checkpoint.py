import numpy as np

def extract_anisotropy_features (Parameters, image, mask=None):
    """Given an image and a mask, return a dictionary with the relevant anisotropy features"""
    
    data_inputs = {}
    
    Ka, Kb, Kc = Parameters.kA, Parameters.kB, Parameters.kC
    
    
    h, w, channels = image.shape
    
    if channels == 2:
        channel_types = ["Para", "Perp"]
    elif channels == 3:
        channel_types = ["Open", "Para", "Perp"]
    
    
    for index, channel in enumerate(channel_types):
        
        data_inputs[channel] = np.sum(image[:,:, index])/np.count_nonzero(image[:,:, index])


    #Additional parameters
    para_value = data_inputs['Para']
    perp_value = data_inputs['Perp']
    data_inputs['AniAvg'] = (para_value - perp_value)/(para_value + 2*perp_value)
    
    #With corrections
    data_inputs['Ix'] = Ix = ((Ka+Kb)*perp_value - (Ka+Kc)*para_value)/(Ka*Kb + Kb*Kb - Ka*Kc - Kc*Kc)
    data_inputs['Iy'] = Iy = (Kb*para_value - Kc*perp_value)/(Ka*Kb + Kb*Kb - Ka*Kc - Kc*Kc)
    data_inputs['AniAvg'] = (Ix - Iy)/(Ix + 2*Iy)
    

    
    return (data_inputs)




def extract_intensity_features(Parameters, image, mask=None):
    """Given an image and a mask, return a dictionary with the relevant intensity features"""
    data_inputs = {}
    
    channel_types = ["Intensity"]
    for index, channel in enumerate(channel_types):
        data_inputs[channel] = np.sum(image[:,:, index])/np.count_nonzero(image[:,:, index])
    
    
    return (data_inputs)


def extract_area(Parameters, image, mask = None):
    data_inputs = {}

    #if image.ndim ==  3: h, w, _ = image.shape
    #else: h,w = image.shape
    
    non_zero = np.count_nonzero(image)

    data_inputs['Segmented_Area'] = non_zero

    data_inputs['Segmented_Percentage'] =  non_zero/image.size

    return data_inputs
