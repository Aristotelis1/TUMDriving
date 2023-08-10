
def crop_and_get_center_image(image):
    sub_image_width = image.shape[1] // 3

    # Calculate the starting and ending indices for the middle portion
    start_index = sub_image_width
    end_index = sub_image_width * 2

    # Crop the middle portion of the image
    cropped_img = image[:, start_index:end_index, :]

    return cropped_img