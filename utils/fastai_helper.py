from fastai.vision import *
from fastai.metrics import error_rate    #May not need this here...
import imageio

"""
Helper functions for FASTAI V1

"""

def acc_segmentation(input_image, target, void_code = 255):
    target = target.squeeze(1)
    mask = target != void_code
    return (input_image.argmax(dim=1)[mask]==target[mask]).float().mean()


#Get the mask filenames from the image filenames

get_y_func = lambda filename: filename.parent.parent/'Mask_Norm'/(filename.name.replace('ome.tif', 'png'))
    



CURRENT_CHANNELS = [9,12,15]        

def open_channel(channel, reader, div):
    if reader.get_length() == 1:
        import_img = reader.get_data(0)
        import_img = np.asarray(import_img[:,:,channel])
    else:
        import_img = np.asarray(reader.get_data(channel)) 
    if div: import_img = np.divide(import_img, np.amax(import_img))
    return (import_img)
    
        
def merge_channels(channels, reader, div):
    num_channels = len(channels)
    merged_image = np.zeros(reader.get_data(0).shape)
    for ch in channels:
        merged_image += open_channel(ch, reader, div)/num_channels
    return (merged_image)


#Note, this is very specific to my FASTai trainer on a 16 channel dataset...
def get_channel(channel, reader, div):
    if channel<20:
        return (open_channel(channel, reader, div))
    
    elif channel == 100: #Merge All Fluorescence Channels
        return (merge_channels([0,3,6], reader, div))
        
    elif channel == 101: #Random Lower BR Channel
        channel =  (np.random.randint(9, 12))
        return (open_channel(channel, reader, div))
    
    
    elif channel == 102: #Random Upper BR Channel
        channel =  (np.random.randint(13, 16))
        return (open_channel(channel, reader, div))
    
    
    elif channel == 103: #Random BR Channel
        channel =  (np.random.randint(13, 16))
        return (open_channel(channel, reader, div))
    
    elif channel == 104: #Merge all BR Channels
        return (merge_channels(range(9, 16), reader, div))
    
    elif channel == 107:  #Any random channels
        channel =  (np.random.randint(0, 16))
        return (open_channel(channel, reader, div))
    
    elif channel == 110: #Black Channel
        return(np.zeros(reader.get_data(0).shape))
        
        
def open_multi_image(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
    "Return `Image` object created from image in file `fn`."
    """with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        x = PIL.Image.open(fn).convert(convert_mode)
    
    """
    ch = CURRENT_CHANNELS  #This is sloppy coding.  I Should have an channel input, but this is only used for testing purposes
    reader = imageio.get_reader(str(fn))
    image = []

    for channel in ch:
        import_img = get_channel(channel, reader, div)
        image.append(import_img)
    
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a =  torch.from_numpy( a.astype(np.float32, copy=False) )
    #if after_open: x = after_open(x)
    x = a
    return cls(x)


def open_image_array(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:

    a = np.asarray(fn)
    if a.ndim==2 : a = np.expand_dims(a,2)
    if div: a = np.divide(a, np.amax(a))
    a =  torch.from_numpy(a.astype(np.float32, copy=False) )
    #if after_open: x = after_open(x)
    return cls(a)


class SegMultiImageList(SegmentationItemList):
    def open(self, fn):
        "Open image in `fn`, subclass and overwrite for custom behavior."
        return open_multi_image(fn, convert_mode=self.convert_mode, after_open=self.after_open)
