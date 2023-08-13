import cv2
import numpy as np

def one_hot_to_str(one_hot_encoded):
    if one_hot_encoded == [1, 0, 0]:
        return 'red'
    elif one_hot_encoded == [0, 1, 0]:
        return 'yellow'
    else:
        return 'green'

def standardize_input(image):
    
    ## TODO: Resize image and pre-process so that all "standard" images are the same size  
    copy_im = np.copy(image)
    
    # Define how many pixels to slice off the sides of the original image
    # crop_percent = 10
    # row_crop = int(copy_im.shape[1] * crop_percent /100)
    # col_crop = int(copy_im.shape[0] * crop_percent /100)
    
    # Using image slicing, subtract the row_crop from top/bottom and col_crop from left/right   
    # cropped_copy_im = image[row_crop:-row_crop, col_crop:-col_crop, :]
    standard_im = cv2.resize(copy_im, (8, 32), interpolation=cv2.INTER_AREA)
    
    return standard_im

def red_mask_feature(rgb_image):
    
    #Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    #red_mask1 = cv2.inRange(hsv, (0,70,50), (10,255,255))
    #red_mask2 = cv2.inRange(hsv, (170,70,50), (180,255,255))
    #red_mask = cv2.bitwise_or(red_mask1,red_mask2)
        
        
    lower_red = np.array([150,140,140])
    upper_red = np.array([180,255,255])
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    

    red_result = cv2.bitwise_and(rgb_image, rgb_image, mask = red_mask)
     
    #find out hsv values based on each of the 3 parts
    
    h,s,v = create_hsv(red_result)
    
    
    return h, s, v
    _
def rgb_feature(rgb_image):
    
    # Define our color selection boundaries in RGB values
    lower_red = np.array([50,0,0]) 
    upper_red = np.array([255,200,200])

    # Define the masked area
    red_mask = cv2.inRange(rgb_image, lower_red, upper_red)
    red_result = cv2.bitwise_and(rgb_image, rgb_image, mask = red_mask)
     
    #slice top
    up = red_result[0:10, :, :]
    r = create_rgb(red_result)
    
    return r
    
def create_rgb(rgb_image):
    
    ## Add up all the pixel values and calculate the average brightness
    area =rgb_image.shape[0]*rgb_image.shape[1] #pixels
    
    #r channel
    r_sum_brightness = np.sum(rgb_image[:,:,0])
    r_avg = r_sum_brightness/area
    
    
    return r_avg

def just_slice(rgb_image):
    
    image = np.copy (rgb_image)
    
    #slice into 3 parts, up, middle, down
    up = image[0:10, :, :]
    middle = image[11:20, :, :]
    down = image[21:32, :, :]
    
    #find out hsv values based on each of the 3 parts
    
    h_up, s_up, v_up = create_hsv(up)
    h_middle, s_middle, v_middle = create_hsv(middle)
    h_down, s_down, v_down = create_hsv(down)
    
    #v in hsv can detect whether theres value in up,middle or down

    if  v_up> v_middle and v_up> v_down:# and s_up>s_middle and s_up>s_down:
            
        return [1,0,0] #red
   
    elif  v_middle > v_down:# and s_middle>s_down:
        return [0,1,0] #yellow
 
    return [0,0,1] #green

def create_hsv(rgb_image):
    
    ## TODO: Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    ## Add up all the pixel values and calculate the average brightness
    area =hsv.shape[0]*hsv.shape[1] #pixels
    
    #H channel
    h_sum_brightness = np.sum(hsv[:,:,0])
    h_avg = h_sum_brightness/area
    
    #S channel
    s_sum_brightness = np.sum(hsv[:,:,1])
    s_avg = s_sum_brightness/area
    
    #V channel
    v_sum_brightness = np.sum(hsv[:,:,2])
    v_avg = v_sum_brightness/area
    
    
    return h_avg, s_avg, v_avg


def create_mask_image(rgb_image,label):
    
    #Convert image to HSV color space
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    
    #analyze histogram
    if label == 'red':
        red_mask1 = cv2.inRange(hsv, (0,30,50), (10,255,255))
        red_mask2 = cv2.inRange(hsv, (150,40,50), (180,255,255))
        mask = cv2.bitwise_or(red_mask1,red_mask2)
        
    elif label == 'yellow':
        mask = cv2.inRange(hsv, (10,10,110), (30,255,255))
    
    #green
    else:
        mask = cv2.inRange(hsv, (45,40,120), (95,255,255))
    
    res = cv2.bitwise_and(rgb_image,rgb_image,mask = mask)
    
    return res

def create_feature(rgb_image):

    h,s,v = create_hsv(rgb_image)
    image = np.copy (rgb_image)
    
    #apply mask
    red_mask = create_mask_image(image,'red')
    yellow_mask = create_mask_image(image,'yellow')
    green_mask = create_mask_image(image,'green')
    
    #slice into 3 parts, up, middle, down
    up = red_mask[0:10, :, :]
    middle = yellow_mask[11:20, :, :]
    down = green_mask[21:32, :, :]
    
    
    #find out hsv values based on each of the 3 parts
    
    h_up, s_up, v_up = create_hsv(up)
    h_middle, s_middle, v_middle = create_hsv(middle)
    h_down, s_down, v_down = create_hsv(down)
    
    #v in hsv can detect whether theres value in up,middle or down

    if  v_up> v_middle and v_up> v_down:# and s_up>s_middle and s_up>s_down:
            
        return [1,0,0] #red
   
    elif  v_middle > v_down:# and s_middle>s_down:
        return [0,1,0] #yellow
 
    return [0,0,1] #green