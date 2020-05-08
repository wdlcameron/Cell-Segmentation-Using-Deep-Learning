import torch
import cv2
import numpy as np
import brambox.boxes as bbb
import lightnet as ln

from torchvision import transforms as tf
from .ROIs import ROIs

def post_transform(boxes, scale, pad):
    for box in boxes:
        box.x_top_left -= pad[0]
        box.y_top_left -= pad[1]

        box.x_top_left *= scale
        box.y_top_left *= scale
        box.width *= scale
        box.height *= scale
    return boxes



def output_ROIs (detections, im_h, im_w, segmentation = None, expansion_factor = 1):
    ROI_list = []
    for det in detections:

        #Note, the pixels are rounded down, but it shouldn't matter since the segmentation network will
        #do the pixel by pixel segmentation
        x = int(det.x_top_left)
        y = int(det.y_top_left)
        height = int(det.height)
        width = int(det.width)
        
        #implement padding (pixel based)
        #pad = (expansion_factor - 1)*10
        pad = expansion_factor
        x -= pad
        y -= pad
        height += 2*pad
        width += 2*pad


        #print(x, y, width, height, expansion_factor)
        #Ensure that the ROIs do not go over the edge of the image.  Alternatively, don't use ROIs that go over the edge...
        if x<0:
            width = width+x
            x=0
        if y<0:
            height = height+y
            y=0
        height = min((im_h - y), height)
        width = min((im_w - x), width)
        
        print ("x", int(det.x_top_left),
               "y", int(det.y_top_left), 
               det.class_label, 
               det.confidence,
               "w", det.width,
               "h", det.height,
              "new_H", height, "new_W", width )
        
        
        #There are some boundary cases where the entire object will be detected off the screen
        if (height >0) and (width >0):
            
            
            
            fake_data = np.ones((height,width))


            roi = ROIs(x = x,
                       y = y, 
                       classification = det.class_label, 
                       confidence = det.confidence, 
                       data = fake_data)  #Use the data as ones for now... replace with segmentation later


            ROI_list.append(roi)
        
    return (ROI_list)




def detect_ROIs(Parameters, image, debug_path = None, expansion_factor = 1):
    
    #Import the relevant parameters:
    network_size = Parameters.network_size
    network = Parameters.network
    net_w, net_h = Parameters.network_size
    device = Parameters.device
    log = Parameters.log

    
    save_check = True
    show_check = False
    show_label = True
    use_cuda = True
    img = image.copy()
    
    #Maybe this should already be set.. 
    network.training = False
    
    im_h, im_w = image.shape[:2]

       
    device = torch.device('cpu')
    if use_cuda:
        if torch.cuda.is_available():
            log.debug('CUDA enabled')
            device = torch.device('cuda')
        else:
            log.error('CUDA not available')

    img = img/np.amax(img)
    img = img*255
    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Transform the image and prep it for the 
    img_tf = ln.data.transform.Letterbox.apply(img, dimension = network_size)
    img_tf = tf.ToTensor()(img_tf)
    img_tf.unsqueeze_(0)
    img_tf = img_tf.to(device)


    # Run the image through the neural net
    with torch.no_grad():
        output = network(img_tf)
        
    if im_w == net_w and im_h == net_h:
        scale = 1
    elif im_w / net_w >= im_h / net_h:
        scale = im_w/net_w
    else:
        scale = im_h/net_h
        
    pad = int((net_w - im_w/scale) / 2), int((net_h - im_h/scale) / 2)

    
    #Convert the boxes into ROIs
    converted_boxes = []
    for b in output:
        converted_boxes.append(post_transform(b, scale, pad))
    
    output = converted_boxes
    
    image_markup = bbb.draw_boxes(img, output[0], color = (255,0,0), show_labels=show_label)
    if Parameters.debug_mode == True:
        if show_check:
            cv2.imshow('image', image_markup)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
        if save_check and debug_path:
            cv2.imwrite(debug_path, image_markup)

    ROI_list = output_ROIs(output[0], im_h, im_w, expansion_factor = expansion_factor)
    
    return ROI_list