import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import ttkbootstrap as ttk
DIR = r"D:/MiniProject" 
PROTOTXT = os.path.join(DIR, "colorization_deploy_v2.prototxt") 
MODEL = os.path.join(DIR, "colorization_release_v2.caffemodel")
POINTS = os.path.join(DIR, "pts_in_hull.npy")
    
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)

class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

root = ttk.Window(themename="cyborg") 
root.title("AI-Powered Image Colorization")
root.geometry("900x600")


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg;*.MP4")])
    
    if not file_path:
        return  
    image = cv2.imread(file_path)
    if image is None:
        print("Error: Unable to load image.")
        return


    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50  


    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))


    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")


    original_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    colorized_img = Image.fromarray(cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB))


    original_img.thumbnail((400, 400))
    colorized_img.thumbnail((400, 400))

    img1 = ImageTk.PhotoImage(original_img)
    img2 = ImageTk.PhotoImage(colorized_img)


    label_original.config(image=img1)
    label_original.image = img1
    label_colorized.config(image=img2)
    label_colorized.image = img2


title_label = ttk.Label(root, text="AI-Powered Image Colorization", font=("Helvetica", 20, "bold"))
title_label.pack(pady=10)

desc_label = ttk.Label(root, text="Upload a grayscale image and watch it come to life in color!", font=("Arial", 12))
desc_label.pack(pady=5)

btn_upload = ttk.Button(root, text="Upload Image", command=upload_image, bootstyle="primary", padding=10)
btn_upload.pack(pady=20)

frame = ttk.Frame(root)
frame.pack()

label_original = ttk.Label(frame, text="Original Image", font=("Arial", 12, "italic"))
label_original.pack(side="left", padx=10)

label_colorized = ttk.Label(frame, text="Colorized Image", font=("Arial", 12, "italic"))
label_colorized.pack(side="right", padx=10)

root.mainloop()