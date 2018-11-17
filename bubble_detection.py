# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 14:11:01 2018

@author: randu
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import fct

def fill_holes(imInput, threshold):
    """
    The method used in this function can be find at
    https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/

    """

    # Threshold.
    th, thImg = cv2.threshold(imInput, threshold, 255, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    imFloodfill = thImg.copy()

    # Get the mask.
    h, w = thImg.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0).
    cv2.floodFill(imFloodfill, mask, (0,0), 255)

    # Invert the floodfilled image.
    imFloodfillInv = cv2.bitwise_not(imFloodfill)

    # Combine the two images.
    imOut = thImg | imFloodfillInv

    return imOut





##### Load the image.
img = cv2.imread('7.jpg',0)

##### smoothing
gaus = cv2.GaussianBlur(img,(5,5),0)

##### threshold
th = cv2.adaptiveThreshold(gaus,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

filled = fill_holes(th, 50)

kernel_op = np.ones((10,10),np.uint8)
cl = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_op)



#method, dp, minDist, param1, param2, minRadius, maxRadius
p = [cv2.HOUGH_GRADIENT, 1, 30, 12, 12, 2, 50] # normalement ~70
circles = cv2.HoughCircles(image=cl, method=p[0], dp=p[1], minDist=p[2], param1=p[3], param2=p[4], minRadius=p[5], maxRadius=p[6])

all_radius_circle = list(circles[0][:,2])


img_circles = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
det = 255*np.ones([img.shape[0], img.shape[1]], np.uint8)

for i in range(len(circles[0])):
    # draw the outer circle
    cv2.circle(img_circles,(circles[0][i][0],circles[0][i][1]),circles[0][i][2],(255,0,0),2)
    cv2.circle(det,(circles[0][i][0],circles[0][i][1]),circles[0][i][2],(0,0,0),2)
# display img
plt.figure(figsize=[20, 15])
plt.imshow(img_circles, cmap='gray')
plt.title('hough')
plt.show()

kernel_op = np.ones((5,5),np.uint8)
cl = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel_op)

nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(cl)

plt.figure(figsize=[20,15])
plt.subplot(121)
plt.imshow(labels)
plt.title('original labels')

for i in range(len(circles[0])):
    cv2.circle(labels,(circles[0][i][0],circles[0][i][1]),circles[0][i][2],(0,0,0),-1)
    cv2.circle(labels,(circles[0][i][0],circles[0][i][1]),circles[0][i][2],(0,0,0),2)
    
plt.subplot(122)
plt.imshow(labels)
plt.title('labels - detected hough circles')
plt.show()  


for i in range(1, nlabels):
    if stats[i][4]>20:
        try:
            I = np.zeros([cl.shape[0], cl.shape[1]], np.uint8)
            I[labels==i] = 255  
            #plt.imshow(I, cmap='gray')
            im2,contours,hierarchy = cv2.findContours(I, 1, 2)
            cnt = contours[0]
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            all_radius_circle.append(radius)
            center = (int(x),int(y))
            radius = int(radius)
            cv2.circle(img_circles,center,radius,(255,0,0),2)
            cv2.circle(det,center,radius,(0,0,0),2)
        except IndexError:
            continue


# display img
fig = plt.figure(figsize=[40, 20])
plt.subplot(121)
plt.imshow(img_circles, cmap='gray')
plt.title('circles detected')
plt.subplot(122)
plt.hist(all_radius_circle, int(2*np.ceil(np.max(all_radius_circle))), [0, int(np.ceil(np.max(all_radius_circle)))])
plt.xlabel('radius')
plt.ylabel('bubbles')
plt.show()
fig.savefig('X40__r.jpg')




#cv2.imwrite("1_a.jpg", det)


##### stocker tous les cercles
##### tracer duplus petits au plus grand
##### comme ça pas de cheuvauchement

#### si chevauchement importants et rayon/centre similaire, fusionner les bulles? 

##### calculer la distribution des aires / surfaces





