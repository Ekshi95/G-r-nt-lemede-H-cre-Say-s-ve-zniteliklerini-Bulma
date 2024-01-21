# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 20:10:19 2024

@author: Talip
"""
import numpy as np
import cv2 
from matplotlib import pyplot as plt


path = "C:/Users/Talip/Desktop/FINAL/hucre.png"
img = cv2.imread(path,0)

th1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,17,19)
    
    
I = th1>70
I=I.astype(np.uint8)
I = 255*I

kernel = np.ones((5,9), np.uint8)
kernel2=np.ones((7,9),np.uint8)
erosion = cv2.erode(I, kernel, iterations = 1)
dilation = cv2.dilate(erosion, kernel, iterations=1)


edges = cv2.Canny(dilation, 0, 255)


dilation2 = cv2.dilate(edges, kernel2, iterations=1)
opening = cv2.morphologyEx(dilation2, cv2.MORPH_OPEN, kernel2)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)


marked_imgs, comp, stats, centroids = cv2.connectedComponentsWithStats(closing, connectivity=8)
sizes = stats[1:, -1]; marked_imgs = marked_imgs - 1

cx=np.zeros(20)  
cy=np.zeros(20)
area=np.zeros(20)  
perimeter=np.zeros(20, dtype=int) 
aspect_ratio=np.zeros(20) 
equi_diameter=np.zeros(20) 
orientation = np.zeros(20)
circularity = np.zeros(20)
compactness = np.zeros(20)
area_to_perimeter_ratio = np.zeros(20)

for i in range(0, marked_imgs):
    img2 = np.zeros((comp.shape),dtype = np.uint8)
    img2[comp == i ] = 255
    # moments
    contours,hierarchy = cv2.findContours(img2, 1, 2)
    cnt = contours[0]
    M = cv2.moments(cnt)
    # center
    cx[i] = round(M['m10']/M['m00'])
    cy[i] = round(M['m01']/M['m00'])
    # contour area
    area[i] = cv2.contourArea(cnt)
    # perimeter
    perimeter[i] = cv2.arcLength(cnt,True)
    # aspect ratio
    x,y,w,h = cv2.boundingRect(cnt)
    aspect_ratio[i] = float(w)/h
    # eq dia
    are = cv2.contourArea(cnt)
    equi_diameter[i] = np.sqrt(4*are/np.pi)
    # orientation
    angle = 0.5 * np.arctan2((2 * M['mu11']), (M['mu20'] - M['mu02']))
    orientation[i] = np.degrees(angle)
    # circularity
    circularity[i] = (4 * np.pi * area[i]) / (perimeter[i] ** 2)
   # compactness
    compactness[i] = (perimeter[i] ** 2) / area[i]
   # area to perimeter ratio
    area_to_perimeter_ratio[i] = area[i] / perimeter[i]


min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img,mask = img2)


plt.figure(figsize=(30, 30))    
plt.subplot(6, 1, 1)
plt.imshow(img, cmap="gray")

plt.subplot(6, 1, 2)
plt.imshow(th1,cmap="gray")

plt.subplot(6, 1, 3)
plt.imshow(dilation,cmap="gray")

plt.subplot(6, 1, 4)
plt.imshow(edges,cmap="gray")

plt.subplot(6, 1, 5)
plt.imshow(closing,cmap='gray')

plt.subplot(6, 1, 6)
plt.imshow(comp,cmap='nipy_spectral')

plt.show()

