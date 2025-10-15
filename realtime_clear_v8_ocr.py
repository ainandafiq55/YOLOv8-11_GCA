import cv2
import math 
import time
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import argparse
import re
import psutil
from paddleocr import PaddleOCR
from openpyxl import Workbook
from ultralytics import YOLO
from numpy.fft import fft2, ifft2
from scipy.stats import norm
from scipy.signal import convolve2d

def gaussian(M, std):
    n = np.arange(0, M) - (M - 1.0) / 2.0
    return np.exp(-0.5 * (n / std) ** 2)

def blur(img, mode = 'box', kernel_size = 3):
    # mode = 'box' or 'gaussian' or 'motion'
    dummy = np.copy(img)
    if mode == 'box':
        h = np.ones((kernel_size, kernel_size)) / kernel_size ** 2
    elif mode == 'gaussian':
        h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
        h = np.dot(h, h.transpose())
        h /= np.sum(h)
    elif mode == 'motion':
        h = np.eye(kernel_size) / kernel_size
    dummy = convolve2d(dummy, h, mode = 'valid')
    return dummy

def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s = img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

def gaussian_kernel(kernel_size = 3):
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    h = np.dot(h, h.transpose())
    h /= np.sum(h)
    return h

def rgb2gray(rgb):
    gray = np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
    gray_3channel = np.stack((gray,)*3, axis=-1)
    return gray_3channel

def rgba2rgb(rgba):
    return rgba[:, :, :3]

def mergeRGB(img_r, img_g, img_b):
    img_rgb = np.stack([img_r, img_g, img_b], axis=-1)
    return img_rgb

# plt imread to cv2
def plt2cv2(img):
    max_val = np.max(img)
    img = (img/max_val * 255).astype(np.uint8)
    return img

def laplacian(img):
    # laplacian type
    c = 1 
    #get laplacian image
    LaplacianImage = cv2.filter2D(src=img,ddepth=-1,kernel=LaplacianFilter)    
    #sharpen the image
    img = img + c*LaplacianImage
    return img

# for laplacian    
LaplacianFilter = np.array([[0,-1,0],
                            [-1,4,-1],
                            [0,-1,0]])

# for wiener
kernel = gaussian_kernel(3)

# True label of OCR path
label_path = r"./assets/dictionary/list_ruang_nama.xlsx"
label_dict = pd.read_excel(label_path)

# Room number correction algorithm
def correct(text): 
    match = text
    text = text.replace(" ","")
    for idx, row in label_dict.iterrows():
        unique_label = row['unique']
        if unique_label.replace(" ", "") in text : match = (row['label'])
    return match

# Setup Argument Parsing
parser = argparse.ArgumentParser(description='Run OCR with optional preprocessing.')
parser.add_argument("--weight", default="yolov8_gca", help='Weight PATH for detection is mandatory')
parser.add_argument('--video', action='store', default=0,help='Video source, 0 for camera')
parser.add_argument('--cpu', action='store_true', default=0, help='Using CPU instead of GPU')
parser.add_argument('--preprocess', default="none", help="select deblur method: 'laplacian', 'wiener'")

args = parser.parse_args()

# Initialize Excel file
excel_data = []
wb = Workbook()
ws = wb.active
ws.append(['Label','Frame', 'Object_Index', 'Object_Detected', 'OCR_Result', 'True/False','Computation_Time', 'FPS'])

# OCR Model Load
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Evaluation Variable
total_cer = 0 #character error rate
total_true = 0 #true prediction
total_obj = 0 #detected object count

total_detection_time = 0
total_ocr_time = 0

# save true label
label = os.path.splitext(args.video)[0]
label = correct(label)

# start webcam
vid_folder = f"./assets/videos/{args.video}"
cap = cv2.VideoCapture(vid_folder)

#set open camera size
if (cap.isOpened() == False):  
    print("Error reading video file") 
frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4))   
size = (frame_width, frame_height)

# Decide filename
save_name = f"{args.video}_{args.weight}"
save_name = os.path.basename(save_name)
# Video filename writer
writer = cv2.VideoWriter(f'{save_name}_walk_.avi',cv2.VideoWriter_fourcc(*'MJPG'),25, size) 

#convert weight args to PATH
weight = f"./weights/{args.weight}.pt"

# load model
model = YOLO(weight)

# object classes
classNames = ["plate"]

# calculate frame
frame_count=1

# for calculate fps
prev_frame_time = 0
new_frame_time = 0
total_fps = 0
min_fps = 999
max_fps = 0
fps = 30 #default for first frame

# for calculate RAM
total_RAM = 0

# For detection label
object_detected = False

while True:
    success, img = cap.read()
    if not success : break
    print("Frame -->", frame_count)
    results = model(img)
    frame_count+=1
    object_index = 0
    
    start_total_time = time.time()
    object_detected = False # Reset to detected object false if no detection yet
    True_False = False #Room recognizing is false if the OCR not match with the label

    # process each detected objects
    for r in results:

        boxes = r.boxes
        for box in boxes:
            
            # confidence
            confidence = math.ceil((box.conf[0]*100))/100
            if confidence >= 0.5 : #based on mAP 0.5

                # set to true for excel
                object_detected = True
                # object index for excel
                object_index += 1

                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                # Crop image
                final_image = img[int(y1):int(y2),int(x1):int(x2)]

                # Set image for OCR
                image_ocr = final_image
                # Proses OCR berdasarkan flag preprocess
                if args.preprocess == "none":
                    image_ocr = image_ocr
                elif args.preprocess == "original":
                    #resize
                    resized_image = cv2.resize(image_ocr, (192, 48), interpolation=cv2.INTER_AREA)
                    image_ocr = plt2cv2(resized_image)
                elif args.preprocess == "laplacian":
                    #divide each img to r, g, b
                    img_r = image_ocr[:,:,0]
                    img_g = image_ocr[:,:,1]
                    img_b = image_ocr[:,:,2]
                    #laplacian
                    laplacian_imgr = laplacian(img_r)
                    laplacian_imgg = laplacian(img_g)
                    laplacian_imgb = laplacian(img_b)
                    laplacian_img = mergeRGB(laplacian_imgr,laplacian_imgg,laplacian_imgb)
                    image_ocr = laplacian_img
                elif args.preprocess == "wiener":
                    #divide each img to r, g, b
                    img_r = image_ocr[:,:,0]
                    img_g = image_ocr[:,:,1]
                    img_b = image_ocr[:,:,2]
                    #wiener
                    wiener_imgr = wiener_filter(img_r, kernel, K = 30)
                    wiener_imgg = wiener_filter(img_g, kernel, K = 30)
                    wiener_imgb = wiener_filter(img_b, kernel, K = 30)
                    wiener_img = mergeRGB(wiener_imgr, wiener_imgg, wiener_imgb)
                    image_ocr = plt2cv2(wiener_img)
                elif args.preprocess == "grayscale":
                    image_ocr = rgb2gray(image_ocr)
                elif args.preprocess == "inversion":
                    image_ocr = cv2.bitwise_not(image_ocr)
                else:
                    # Tidak menggunakan preprocessing
                    image_ocr = image_ocr

                # OCR process
                # Hitung waktu mulai OCR
                start_time = time.time()
                

                # Paddle text recognition
                ocr_result = ""
                if image_ocr is None or image_ocr.shape[1] == 0 or image_ocr.shape[0] == 0:
                    print("Image not loaded properly or has zero width/height.")
                else :
                    ocr_result = ocr.ocr(image_ocr, cls=True)
                output = ""
                if ocr_result is not None :
                    for idx in range(len(ocr_result)):
                        res = ocr_result[idx]
                        if res is not None :
                            for line in res:
                                output += line[1][0] + " "
                ocr_result = output

                # Calculate OCR time and add to total time
                end_time = time.time()
                ocr_time = end_time - start_time
                total_ocr_time += ocr_time

                # Tampilkan waktu yang dibutuhkan untuk proses OCR gambar ini
                print(f"OCR time for frame {frame_count}, at object {object_index}: {ocr_time:.4f} sec")

                raw_recognized_text = ocr_result
                #correct the recognized text
                recognized_text = correct(ocr_result)

                # The True_False became true if recognized text matched with the true label
                if recognized_text == label : True_False = True

                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # show recognized text on bbox
                object_text = recognized_text
                
                # put text on bounding box
                org = [x1,y1]
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2
                cv2.putText(img, object_text, org, cv2.FONT_HERSHEY_SIMPLEX, fontScale, color, thickness)

                # Total time computation
                object_total_time = time.time() - start_total_time
                
                # Record at excel file only at each 5th frame
                # ['Label','Frame', 'Object_Index', 'Object_Detected', 'OCR_Result', 'True/False','Computation_Time', 'FPS']
                if frame_count%5 == 0 :
                    excel_data.append([label, frame_count, object_index, object_detected, object_text, True_False,object_total_time, fps])

                True_False = False #put back to false after write in excel

    # Total time computation
    frame_total_time = time.time() - start_total_time

    # FPS calculation
    new_frame_time = time.time()
    fps = 1/((new_frame_time+0.001)-prev_frame_time) #0.001 to prevent zero division
    prev_frame_time = new_frame_time

    if fps > max_fps: max_fps=fps #update max fps
    if fps < min_fps: min_fps=fps #updat min fps
    total_fps += fps

    fps = int(fps)
    cv2.putText(img, f"fps: {fps:.2f}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # RAM calculation
    process = psutil.Process()
    RAM_usage = (process.memory_info().rss)/1000000  # in bytes 
    total_RAM += RAM_usage

    # Record at excel file with no OCR results if no detection, and at 5th frame
    if (not object_detected) and (frame_count%5 == 0): excel_data.append([label, frame_count, object_index, object_detected, "", True_False, frame_total_time, fps])

    writer.write(img)
    # Show image
    cv2.imshow('Webcam', img) 

    # if q key pressed, then exit the real time camera
    if (cv2.waitKey(1) == ord('q')):
        break

# average fps
print("Average FPS -->", total_fps/frame_count)
print("Min FPS -->", min_fps)
print("Max FPS -->", max_fps)

# average RAM
print("Average RAM -->", total_RAM/frame_count)

# Write data to Excel
for row in excel_data:
    ws.append(row)

wb.save(f"{save_name}_realtime_data_walk.xlsx")

# Release cap
cap.release()
cv2.destroyAllWindows()

#command: python realtime_clear_v8_ocr.py --weight yolov8-ghost-casppf --preprocess inversion --video "F 3.1.mp4"