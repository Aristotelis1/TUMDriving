import os
import cv2

try:
    from trafficlights.utils import create_feature, red_mask_feature, rgb_feature, just_slice, standardize_input, one_hot_to_str
except ImportError:
    from utils import create_feature, red_mask_feature, rgb_feature, just_slice, standardize_input, one_hot_to_str

def estimate_label(bgr_image):

    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2BGR)
    ## classify the image and output a one-hot encoded label
    rgb_image = standardize_input(rgb_image)
    
    predicted_label = create_feature(rgb_image)
    h,s,v = red_mask_feature(rgb_image)
    r = rgb_feature(rgb_image)
    res = just_slice(rgb_image)
    
    
    if predicted_label != [1,0,0] and h>=0 and r> 0:
        
        if h>0:
        
            predicted_label =  [1,0,0] #red
            
            
            
        if predicted_label!= res and h==0.0:
                
            predicted_label = res           
                
    return one_hot_to_str(predicted_label)

if __name__ == "__main__":
    # Path to the directory containing the images
    image_folder = "../data/trafficlights/"

    # List all files in the directory
    image_files = os.listdir(image_folder)

    # Loop through each image file
    for image_file in image_files:
        # Construct the full path to the image
        image_path = os.path.join(image_folder, image_file)

        # Read the image using cv2
        image = cv2.imread(image_path)
        if image is not None:
            predicted_label = estimate_label(image)
            print(predicted_label)
        else:
            print(f"Unable to read image: {image_path}")
