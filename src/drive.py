# This server used to receive images and current vehicle data from the client, passes the images to a trained classifier to predict new vehicle commands, 
# and then sends these commands back to Unity to control the car for autonomous driving.

# now we only test the communication between the client and server at first, 
# so we just give the fixed command to see wether the car will be controlled and wether the images succesffly recieved
# lately we can also add code to realize autonomous driving

import socketio
# concurrent networking
import eventlet
# web server gateway interface
import eventlet.wsgi
from flask import Flask
import base64
import cv2
import numpy as np
# from io import BytesIO
import time
import os
import torch
from ultralytics import YOLO

from networks.resnet import resnet_model
from networks.nvidia import NetworkNvidia
from tests.yolov5.transforms import crop_and_get_center_image
from helper.cv_utils import calculate_iou
from trafficlights.classifier import estimate_label

import random

# Initialize Socket.IO server
sio = socketio.Server()
app = Flask(__name__)

frame_count = 0
frame_count_save = 0
prev_time = 0
fps = 0
model = None


@sio.on("send_image")
def on_image(sid, data):
    # print("image received")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    #make the variables global to calculate the fps 
    global frame_count, frame_count_save, prev_time, fps
    #print("image recieved!")
    img_data = data["image"]
    img_bytes = base64.b64decode(img_data)
    # Decode image from base64 format
    img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)

    brake = 0
    throttle = 0.3
    steering_angle = 0.0
    # show the recieved images on the screen
    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        img = crop_and_get_center_image(img)
        img_yolo = cv2.resize(img, (360, 360), interpolation=cv2.INTER_AREA)
        start_time = time.time()
        with torch.no_grad():
            results = yolo([img_yolo], size=360)
        end_time = time.time()
        # inference_time = (end_time - start_time) * 1000 # convert to ms
        # print("Inference time: ", inference_time)
        
        # Calculate the center of the image
        image_height, image_width, _ = img_yolo.shape
        image_center_x = image_width // 2
        image_center_y = image_height // 2
        brake_area = (image_center_x - 50, image_center_y, image_center_x + 60, image_center_y + 100)
        cv2.rectangle(img_yolo, (brake_area[0], brake_area[1]), (brake_area[2], brake_area[3]), (0, 0, 255), 2)
        for bbox in results.xyxy[0]:
            xmin = int(bbox[0])
            ymin = int(bbox[1])
            xmax = int(bbox[2])
            ymax = int(bbox[3])
            prob = bbox[4]
            if int(bbox[5] == 9):
                cv2.rectangle(img_yolo, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)  # Blue rectangle
                traffic_light_image = img_yolo[ymin: ymax, xmin:xmax]
                try:
                    if estimate_label(traffic_light_image) == "red":
                        print("Red light detected...FULL BRAKE! --- probability: ", prob, " shape: ", traffic_light_image.shape)
                        brake = 1
                except Exception as e:
                    print(str(e), " --- ", traffic_light_image.shape)
                    i = random.randint(1, 20)
                    cv2.imwrite(f"data/trafficlights/traffic_lights_{i}.jpg", traffic_light_image)
            if int(bbox[5]) == 2 or int(bbox[5]) == 5:
                if calculate_iou(brake_area, (xmin, ymin, xmax, ymax)) > 0.02:
                    # print("IOU: ", calculate_iou(brake_area, (xmin, ymin, xmax, ymax)))
                    print("Car/Bus is in the brake area...FULL BRAKE! --- probability: ", prob)
                    brake = 1

                cv2.rectangle(img_yolo, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Green rectangle
        send_control(steering_angle, throttle, brake)
        cv2.namedWindow("image from unity", cv2.WINDOW_NORMAL)
        cv2.imshow("image from unity", img_yolo)
        key = cv2.waitKey(1)
        if key == 27:
            cv2.destroyAllWindows()
            return

    else:
        print("Invalid image data")

    # # create a variable to identify every frame of image for the lately image-save
    # frame_count_save += 1

    # Calculate and print fps
    frame_count += 1
    elapsed_time = time.time() - prev_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        # print(f"FPS: {fps:.2f}")
        prev_time = time.time()
        frame_count = 0

    # # split the merged image
    # _, width, _ = img.shape
    # single_width = width // 3

    # img1 = img[:, :single_width, :]
    # img2 = img[:, single_width : 2 * single_width, :]
    # img3 = img[:, 2 * single_width :, :]

    # # save the images
    # save_folder = os.path.expanduser("~/Desktop/receivedImage")
    # os.makedirs(save_folder, exist_ok=True)
    # filename = os.path.join(save_folder, f"image{frame_count_save:02d}.jpg")
    # cv2.imwrite(filename, img)

# listen for the event "vehicle_data"
@sio.on("vehicle_data")
def vehicle_command(sid, data):
    steering_angle = float(data["steering_angle"])
    throttle = float(data["throttle"])
    brake = float(data["brake"])
    velocity = float(data["velocity"])
    if data:
        # throttle = 0.3
        if velocity >= 12.0:
            steering_angle = 0.0
            throttle = 0.0
            brake = 0.0
            send_control(steering_angle, throttle, brake)
        else:
            send_control(0.0, 0.275, 0)
    else:
        # send the data to unityClient
        # sio.emit("manual", data={})
        print("data is empty")


@sio.event
def connect(sid, environ):
    # sid for identifying the client connected，environ
    print("Client connected")
    send_control(0, 0, 0)

# Define a data sending function to send processed data back to unity client
def send_control(steering_angle, throttle, brake):
    # print(f"steering: {steering_angle}, throttle: {throttle}")
    sio.emit(
        "control_command",
        data={
            "steering_angle": steering_angle.__str__(),
            "throttle": throttle.__str__(),
            "brake": brake.__str__(),
        },
        skip_sid=True,
    )

@sio.event
def disconnect(sid):
    # implement this function, if disconnected
    print("Client disconnected")


app = socketio.Middleware(sio, app)
# Connect to Socket.IO client
if __name__ == "__main__":

    print("Loading model...")
    # model = resnet_model()
    # model = NetworkNvidia()
    # model.load_state_dict(torch.load("weights/nvidia/MSE_Steering/nvidia_50.pt", map_location=torch.device('mps')))
    # model = model.to('mps')
    
    # model.eval()

    # YOLOV5 Model
    # yolo = YOLO('yolov8n.pt')
    yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    yolo = yolo.to("mps")
    yolo.conf = 0.50  # NMS confidence threshold
    yolo.classes = [2, 5, 9]  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO

    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)