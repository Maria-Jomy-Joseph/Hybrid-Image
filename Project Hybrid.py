import cv2
import numpy as np


def cross_correlation_2d(img, kernel):
    # input
    m, n = kernel.shape

    output = np.empty(img.shape)

    # keep the image into 3 dimensions
    if len(img.shape) == 3:
        height, width, channel = img.shape
    else:
        height, width = img.shape
        channel = 1
        img = np.expand_dims(img, axis=2)

    # set up a new workplace adding size of kernels and images
    newpad = np.zeros((m + height - 1, n + width - 1, channel), dtype=img.dtype)

    m1 = int((m - 1) / 2)
    n1 = int((n - 1) / 2)
    height = int(height)
    width = int(width)
    # put the image into the workplace
    newpad[m1:m1 + height, n1:n1 + width] = img

    matrix = m * n
    kernel = kernel.reshape(-1)
    # calculate the output image
    for i in range(width):
        for j in range(height):
            cross_image = np.reshape(newpad[j:j + m, i:i + n], (matrix, channel))
            output[j, i] = np.dot(kernel, cross_image)

    return output


def convolve_2d(img, kernel):

    return cross_correlation_2d(img, np.fliplr(np.flipud(kernel)))


def gaussian_blur_kernel_2d(sigma, width, height):
    
    # make the range of i and j (X and Y) btw -width/2 and width/2+1, -height/2 and height/2+1
    x, y = int(width / 2), int(height / 2)
    x1, y1 = x + 1, y + 1
    X = np.arange(-x, x1, 1.0) ** 2
    Y = np.arange(-y, y1, 1.0) ** 2

    X = np.exp(-X / (2 * sigma * sigma))
    Y = np.exp(-Y / (2 * sigma * sigma)) / (2 * sigma * sigma * np.pi)
    output = np.outer(X, Y)

    normalize = np.sum(Y) * np.sum(X)
    return output / normalize

def low_pass(img, sigma, size):
    
    return convolve_2d(img, gaussian_blur_kernel_2d(sigma, size, size))


def high_pass(img, sigma, size):
    
    return img - low_pass(img, sigma, size)


def create_hybrid_image(img1, img2, sigma1, size1, high_low1, sigma2, size2,
                        high_low2, mixin_ratio):
    
    high_low1 = high_low1.lower()
    high_low2 = high_low2.lower()

    if img1.dtype == np.uint8:
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

    if high_low1 == 'low':
        img1 = low_pass(img1, sigma1, size1)
    else:
        img1 = high_pass(img1, sigma1, size1)

    if high_low2 == 'low':
        img2 = low_pass(img2, sigma2, size2)
    else:
        img2 = high_pass(img2, sigma2, size2)

    img1 *= 2 * (1 - mixin_ratio)
    img2 *= 2 * mixin_ratio
    hybrid_img = (img1 + img2)
    return (hybrid_img * 255).clip(0, 255).astype(np.uint8)

import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename

root = tk.Tk()
root.geometry("600x600")
root.config(background="#FFFFFF")
e1 =tk.Label(root)
e1.grid(row=0,column=0)
e2 =tk.Label(root)
e2.grid(row=0,column=2)
e3 =tk.Label(root)
e3.grid(row=4,column=1)

path1 = StringVar()
def file1():
    f_types =[('JPG File', '*.jpg'),('PNG File','*.png')]
    filename=askopenfilename(filetypes=f_types)
    if filename is not None:
        img=Image.open(filename) # read the image file
        img=img.resize((200,200)) # new width & height
        img=ImageTk.PhotoImage(img)
        path1.set(filename)
        e1.image=img
        e1['image']=img
        
path2 = StringVar()
def file2():
    f_types =[('JPG File', '*.jpg'),('PNG File','*.png')]
    filename=askopenfilename(filetypes=f_types)
    
    if filename is not None:
        img=Image.open(filename) # read the image file
        img=img.resize((200,200)) # new width & height
        img=ImageTk.PhotoImage(img)
        path2.set(filename)
        e2.image=img
        e2['image']=img

def file3():
    image1= cv2.imread(path1.get())
    image2_read= cv2.imread(path2.get())
    height = image1.shape[0]
    width = image1.shape[1]
    dim=(width,height)
    image2=cv2.resize(image2_read,dim,interpolation = cv2.INTER_AREA)
    filename=create_hybrid_image(image1, image2, 7, 12, "high", 7, 12,"low", 0.5)
    height, width = filename.shape[:2]
    ppm_header = f'P6 {width} {height} 255 '.encode()
    data = ppm_header + cv2.cvtColor(filename, cv2.COLOR_BGR2RGB).tobytes()
    imgmade= tk.PhotoImage(width=width, height=height, data=data, format='PPM')
    e3.image=imgmade
    e3['image']=imgmade
    
    
openfirst = tk.Button(root, text ="Load first Image",command= lambda : file1()).grid(row=1,column=0)
opensecond=tk.Button(root, text ="Load second Image",command= lambda : file2()).grid(row=1,column=2)
createhybrid=tk.Button(root, text ="Show Hybrid Image",command= lambda : file3()).grid(row=3,column=1)
root.mainloop()
