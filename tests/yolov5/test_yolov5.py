import time
import cv2
import torch
from transforms import crop_and_get_center_image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5n')
model = model.to("mps")
model.conf = 0.30  # NMS confidence threshold
model.classes = [2, 5, 9]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs

# Images
# for f in 'zidane.jpg', 'bus.jpg':
#     torch.hub.download_url_to_file('https://ultralytics.com/images/' + f, f)  # download 2 images
# im1 = Image.open('zidane.jpg')  # PIL image
# im2 = cv2.imread('bus.jpg')[..., ::-1]  # OpenCV image (BGR to RGB)
for i in range(50, 150):
    image_path = f"/Users/aristotelistsoytsanis/Downloads/Mac/Train/Data/Dataset1/IMG/CapturedImage{i}.jpg"
    image_original = cv2.imread(image_path)[..., ::-1]  # OpenCV image (BGR to RGB)
    image = crop_and_get_center_image(image_original)
    image = cv2.resize(image, (360, 360), interpolation=cv2.INTER_AREA)
    # cv2.imshow("img", image)
    # cv2.waitKey(20)

    # Inference
    start_time = time.time()
    with torch.no_grad():
        results = model([image], size=360) # batch of images
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000
    print("Inference time (ms): ", inference_time)
    # results.save()
    # print(results.xyxy)
    for bbox in results.xyxy[0]:
        if int(bbox[5])==9: # traffic light
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            traffic_light = image[ymin:ymax, xmin:xmax]
            cv2.imwrite(f"trafficlights/traffic_light{i}{xmin}.jpg", traffic_light)
            # cv2.imshow("traffic_light", traffic_light)
            cv2.waitKey(0)
    # results.xyxy

# Results
# results.print()
# results.show()  # or .show()

# results.xyxy[0]  # im1 predictions (tensor)
# results.pandas().xyxy[0]  # im1 predictions (pandas)
#      xmin    ymin    xmax   ymax  confidence  class    name
# 0  749.50   43.50  1148.0  704.5    0.874023      0  person
# 1  433.50  433.50   517.5  714.5    0.687988     27     tie
# 2  114.75  195.75  1095.0  708.0    0.624512      0  person
# 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
