from fastai.vision import *
import imageio

CURRENT_CHANNELS = [0,3,6]        
SPECIAL_DROPOUT_RATE = 0.0
DROPOUT_RATE = 0.0


def open_channel(channel, reader, div):
    
    if reader.get_length() == 1:
        import_img = np.asarray(reader.get_data(0))
        x1, x2, x3 = import_img.shape
        #print("Exception...", import_img.shape)
        if x1<5: import_img = import_img[channel, :,:]
        else: import_img = import_img[:,:, channel]
    
    else:
        import_img = np.asarray(reader.get_data(channel))
        
    if div: import_img = np.divide(import_img, np.amax(import_img))
    return (import_img)
     
def merge_channels(channels, reader, div, dropout_rate = 0.0):
    num_channels = len(channels)
    merged_image = np.zeros(reader.get_data(0).shape)
    for ch in channels:
        if np.random.uniform() > dropout_rate:
            temp_channel = open_channel(ch, reader, div)
            merged_image += temp_channel/np.amax(temp_channel)/num_channels
            #print ("Kept channel", ch)
        #else: print ("Dropped channel", ch)
    return (merged_image)


def get_channel(channel, reader, div, dropout_chance = DROPOUT_RATE):
    if channel<=50:
        return (open_channel(channel, reader, div))
    
    elif channel == 100: #Merge first of all Fluorescence Channels
        return (merge_channels([0,3,6], reader, div))
        
    elif channel == 101: #Random Lower BR Channel
        channel =  (np.random.randint(9, 12))
        return (open_channel(channel, reader, div))
    
    
    elif channel == 102: #Random Upper BR Channel
        channel =  (np.random.randint(13, 16))
        return (open_channel(channel, reader, div))
    
    elif channel == 103: #Random BR Channel
        channel =  (np.random.randint(9, 16))
        return (open_channel(channel, reader, div))
    
    elif channel == 104: #Merge all BR Channels
        return (merge_channels(range(9, 16), reader, div))
    
    
    elif channel == 107:  #Any random channels
        channel =  (np.random.randint(0, 16))
        return (open_channel(channel, reader, div))
    
    elif channel == 108:  #One of the fluorescent channels
        fluorescent_channels  = [0,3,6]
        channel = np.random.choice(fluorescent_channels, 1)
        return (open_channel(channel, reader, div))
    
    elif channel == 110: #Black Channel
        return(np.zeros(reader.get_data(0).shape))
    
    elif channel == 111: #Dropout Merge of Fluorescent Channels
        return (merge_channels([0,3,6], reader, div, SPECIAL_DROPOUT_RATE))
    
    #Externaldata new channel setups (FLUORESCENT [0,1], Below OOF [2,3,4], Infocus [5], Above OOF [6,7,8])
    elif channel == 121: #Dropout Merge of Fluorescent Channels
        return (merge_channels([0,1], reader, div, SPECIAL_DROPOUT_RATE))
    
    elif channel == 122: #Below OOF random
        channel =  2
        return (open_channel(channel, reader, div))
    
    elif channel == 123: #Above OOF random
        channel =  (np.random.randint(4, 7))
        return (open_channel(channel, reader, div))
    
    
    elif channel >= 200:     #Lots of room for errors with this...  Will default to the 200 code if it pases the dropout test
        if np.random.uniform()>dropout_chance: 
            #print ("Defaulting to", channel-200)
            return (get_channel(channel - 200, reader, div))
        else:
            return (get_channel(110, reader, div))
        
        

        
        
def open_image_array(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
   #Assumes an input of (h, w, c)
    
    a = np.asarray(fn)
    if a.ndim==2 : a = np.expand_dims(a,2)
    #a = np.moveaxis(a, -1, 0)  #move last axis (the number of channels) to the front
    if div: a = np.divide(a, np.amax(a))
    a =  torch.from_numpy( a.astype(np.float32, copy=False) )

    #if after_open: x = after_open(x)
    x = a
    return cls(x)


        
def open_image_2(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
    "Return `Image` object created from image in file `fn`."
    """with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        x = PIL.Image.open(fn).convert(convert_mode)
    
    """
    ch = CURRENT_CHANNELS  #This is sloppy coding.  I Should have an channel input, but this is only used for testing purposes
    dropout_rate = DROPOUT_RATE

    reader = imageio.get_reader(str(fn))
    
    image = []

    for channel in ch:
        import_img = get_channel(channel, reader, div, dropout_rate)
        image.append(import_img)

    image_together = np.dstack(image)
    
    
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a =  torch.from_numpy( a.astype(np.float32, copy=False) )
    #if after_open: x = after_open(x)
    x = a
    #print(type(cls(x))) 
    return cls(x)


class SegMultiImageList(SegmentationItemList):
    def open(self, fn):
        "Open image in `fn`, subclass and overwrite for custom behavior."
        return open_image_2(fn, convert_mode=self.convert_mode, after_open=self.after_open)


