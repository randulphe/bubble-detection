# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import cv2


def image_to_imagette(image, nb_h, nb_w):

    imagette = []
    
    imagette_hauteur_pixel = int(image.shape[0]/nb_h)
    imagette_largeur_pixel= int(image.shape[1]/nb_w)

    idx = 0
    for i in range(nb_h):
        for j in range(nb_w):
            imagette.append(image[i*imagette_hauteur_pixel:(i+1)*imagette_hauteur_pixel, j*imagette_largeur_pixel:(j+1)*imagette_largeur_pixel])
            idx += 1

    return np.array(imagette)




def otsu_thresh(imagette):
    
    imagette_thresh = []
#    thresh = []
    
    for i in range(imagette.shape[0]):
#        tmp_ret, tmp_thresh = cv2.threshold(imagette[i],0,255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        tmp_thresh = cv2.adaptiveThreshold(imagette[i],255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        imagette_thresh.append(tmp_thresh)
#        thresh.append(tmp_ret)
    imagette_thresh = np.array(imagette_thresh)
    
#    return thresh, imagette_thresh
    return imagette_thresh


def circle_detection(imagette_th, p):
    
    circles = []
    
    for i in range(imagette_th.shape[0]):        
        # circles detection
        circles.append(cv2.HoughCircles(image=imagette_th[i], method=p[0], dp=p[1], minDist=p[2], param1=p[3], param2=p[4], minRadius=p[5], maxRadius=p[6]))
    
    return circles




def draw_circle_imagette(imagette_thresh, circles):

    # draw circles on img
    
    imagette_with_circles = []
    
    for i in range(len(circles)):
        
        if type(circles[i]) == type(None):
            continue
        
        circles[i] = np.uint16(np.around(circles[i]))

    for i in range(imagette_thresh.shape[0]):
        imagette_with_circles.append(cv2.cvtColor(imagette_thresh[i],cv2.COLOR_GRAY2BGR))

        if type(circles[i]) == type(None):
            continue

        for j in circles[i][0,:]:
            # draw the outer circle
            cv2.circle(imagette_with_circles[i],(j[0],j[1]),j[2],(255,255,255),-1)
            cv2.circle(imagette_with_circles[i],(j[0],j[1]),j[2],(0,0,0),1)
            # draw the center of the circle
            cv2.circle(imagette_with_circles[i],(j[0],j[1]),2,(0,0,0),1)


    return np.array(imagette_with_circles)



def circles_coord_conversion(nb_h, nb_w, pixel_h, pixel_w, circles):

    imagette_hauteur_pixel = int(pixel_h/nb_h)
    imagette_largeur_pixel = int(pixel_w/nb_w)
    
    circles_coord = []
    
    for i in range(len(circles)):
        
        if type(circles[i]) == type(None):
            continue

        idx_h, idx_w = i//nb_h, i%nb_h

        for j in range(circles[i].shape[1]):
            circles[i][0][j][0] += idx_w*imagette_largeur_pixel
            circles[i][0][j][1] +=  idx_h*imagette_hauteur_pixel

            circles_coord.append([circles[i][0][j][0], circles[i][0][j][1], circles[i][0][j][2]])

    return circles_coord


