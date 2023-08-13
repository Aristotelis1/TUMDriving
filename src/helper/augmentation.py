import random
import cv2
import numpy as np

def random_blur(image, max_kernel_size=5):
    """
    Applies a random blur effect to the input image.

    Parameters:
        image (numpy array): The input image (numpy array) to be blurred.
        max_kernel_size (int): The maximum size of the Gaussian blur kernel.
                               The actual kernel size will be a random integer
                               between 1 and max_kernel_size.

    Returns:
        numpy array: The blurred image.
    """
    # Randomly choose a kernel size for Gaussian blur
    kernel_size = random.randint(1, max_kernel_size) * 2 + 1

    # Apply Gaussian blur with the randomly chosen kernel size
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    return blurred_image

def random_contrast(image, min_factor=0.5, max_factor=1.5):
    # Generate a random contrast factor within the specified range
    contrast_factor = np.random.uniform(min_factor, max_factor)

    # Adjust the pixel intensity values using the contrast factor
    adjusted_image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

    return adjusted_image

def random_brightness(image, min_value=-30, max_value=30):
    # Generate a random brightness value within the specified range
    brightness_value = np.random.randint(min_value, max_value + 1)

    # Adjust the pixel intensity values using the brightness value
    adjusted_image = np.clip(image.astype(np.int16) + brightness_value, 0, 255).astype(np.uint8)

    return adjusted_image

def crop_and_get_center_image(image):
    sub_image_width = image.shape[1] // 3

    # Calculate the starting and ending indices for the middle portion
    start_index = sub_image_width
    end_index = sub_image_width * 2

    # Crop the middle portion of the image
    cropped_img = image[:, start_index:end_index, :]

    return cropped_img

def crop_sky(image):

    return image[65:, :, :]

def rgb2yuv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)