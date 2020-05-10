from fastai.vision import *
import imageio

# CURRENT_CHANNELS = [0,3,6]        
# SPECIAL_DROPOUT_RATE = 0.0
# DROPOUT_RATE = 0.0


def open_channel(channel, reader, div):
    """
    Uses the reader to open the indicated channel.  If div is true, divide the channel by its maximum value to normalize it in the range 0-1
    """
    
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
    """
    Merge the channels, applying a dropout test to each one
    channels:     a list of channels you would like to merge
    reader:       the reader used to open the image
    div:          set to True if you would like to scale your values to the range 0-1 (passthrough variable)
    dropout_rate: 0 for no dropout, 1 for complete dropout of each channel
    """
    num_channels = len(channels)
    merged_image = np.zeros(reader.get_data(0).shape)
    for ch in channels:
        if np.random.uniform() > dropout_rate:
            temp_channel = open_channel(ch, reader, div)
            merged_image += temp_channel/np.amax(temp_channel)/num_channels
    return (merged_image)


def get_channel(channel, reader, div, dropout_chance = 0, merge_dropout_chance = 0):
    """
    Output a channel based on the 
    channel:   the code used to detemine how the channel will be loaded
    reader:    the reader used to open the image
    div:       set to True if you would like to scale your values to the range 0-1 (passthrough variable)
    dropout_chance:       0 for no dropout, 1 for complete dropout of each channel.  This will be used for channel codes >=200
    merge_dropout_chance: 0 for no dropout, 1 for complete dropout of each channel.  This is a passthrough variable for the merge-based codes  
    """
    
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
        return (merge_channels([0,3,6], reader, div, merge_dropout_chance))
    
    #Externaldata new channel setups (FLUORESCENT [0,1], Below OOF [2,3,4], Infocus [5], Above OOF [6,7,8])
    elif channel == 121: #Dropout Merge of Fluorescent Channels
        return (merge_channels([0,1], reader, div, merge_dropout_chance))
    
    elif channel == 122: #Below OOF random
        channel =  2
        return (open_channel(channel, reader, div))
    
    elif channel == 123: #Above OOF random
        channel =  (np.random.randint(4, 7))
        return (open_channel(channel, reader, div))
    
    elif channel >= 200:
        if np.random.uniform()>dropout_chance: 
            return (get_channel(channel - 200, reader, div))
        else:
            return (get_channel(110, reader, div))
        
        
  
def open_image_array(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
    """
    Outputs an Image (by default) instance from an array.  Alternative to open_image, which accepts a filename
    fn:        input array
    div:       set to True if you would like to scale your values to the range 0-1
    class:     image class initialized using your processed array
    """
    a = np.asarray(fn)
    if a.ndim==2 : a = np.expand_dims(a,2)
    if div: a = np.divide(a, np.amax(a))
    a =  torch.from_numpy( a.astype(np.float32, copy=False) )
    x = a
    return cls(x)


        
def open_image_custom(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None, ch= [0,0,0], channel_dropout = 0, merge_dropout=0)->Image:
    """
    Outputs an Image (by default) instance from a filename using the loading codes
    fn:       input array
    div:      set to True if you would like to scale your values to the range 0-1
    class:    image class initialized using your processed array
    ch:       an array of loading codes to outline how each of the output channels should be processed
    channel_dropout:  the chance that channels whose codes are >=200 will be replaced by a black channel
    merge_dropout:    for merge-based loading codes, the chance that each input channel will be replaced by a black channel during the merge
    """
    reader = imageio.get_reader(str(fn))    
    image = []
    for channel in ch:
        import_img = get_channel(channel, reader, div, channel_dropout, merge_dropout)
        image.append(import_img)
    a = np.asarray(image)
    if a.ndim==2 : a = np.expand_dims(a,2)
    a =  torch.from_numpy( a.astype(np.float32, copy=False) )
    x = a
    return cls(x)


class SegMultiImageList(SegmentationItemList):
    CURRENT_CHANNELS = [[0,3,6]]        
    SPECIAL_DROPOUT_RATE = [0.0]
    DROPOUT_RATE = [0.0]

    def open(self, fn):
        "Open image in `fn`"
        return self.open_image_2(fn, convert_mode=self.convert_mode, after_open=self.after_open)


            
    def open_image_2(self, fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
            after_open:Callable=None)->Image:
        """
        Outputs an Image (by default) instance from a filename using the loading codes
        fn:       input array
        div:      set to True if you would like to scale your values to the range 0-1
        class:    image class initialized using your processed array
        """
        ch = self.CURRENT_CHANNELS[0]
        merge_dropout = self.SPECIAL_DROPOUT_RATE[0]
        channel_dropout = self.DROPOUT_RATE[0]
        reader = imageio.get_reader(str(fn))
        image = []
        for channel in ch:
            import_img = get_channel(channel, reader, div, channel_dropout, merge_dropout)
            image.append(import_img)
        a = np.asarray(image)
        if a.ndim==2 : a = np.expand_dims(a,2)
        a =  torch.from_numpy( a.astype(np.float32, copy=False) )
        x = a
        return cls(x)

    def set_training_parameters(self, loading_channels, channel_dropout, merge_dropout): 
        """
        Set the training parameters that will be used during the open_image_2 class method.
        """
        self.CURRENT_CHANNELS[0], self.DROPOUT_RATE[0], self.SPECIAL_DROPOUT_RATE[0] =  loading_channels, channel_dropout, merge_dropout

