'''
Created on 05-Oct-2015

@author: Jim DSouza

Description : Simple classifier that identifies the shape and brand of cans on supermarket shelves
The tolerance rate is adjusted for different images to give highest accuracy
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

img_rgb = cv2.imread('D:\\Redbull\\Model\\shelf_1.jpeg',1)

#cv2.imshow( "Display Window", img_rgb )
#cv2.waitKey(0)

img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

def classify(template, im, clone, threshold, color):
    template = cv2.imread(template,0)
    res = cv2.matchTemplate(im, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.circle(clone, (pt[0] + 40, pt[1]), 15, color, 15)

def regular(shelf_type=1, im=img_rgb):
    if shelf_type == 1:
        clone = im.copy()
        
        classify('D:\\Redbull\\Model\\templates\\regular\\redbull1.png', img_gray, clone, 0.83, (255, 0, 0))
        classify('D:\\Redbull\\Model\\templates\\regular\\redbull2.png', img_gray, clone, 0.88, (255, 0, 0))
        classify('D:\\Redbull\\Model\\templates\\regular\\redbull3.png', img_gray, clone, 0.9, (255, 0, 0))
        classify('D:\\Redbull\\Model\\templates\\regular\\redbull4.png', img_gray, clone, 0.8, (255, 0, 0))
        classify('D:\\Redbull\\Model\\templates\\regular\\redbull5.png', img_gray, clone, 0.9, (255, 0, 0))

        return clone

def zerocal(shelf_type=1, im=img_rgb):
    if shelf_type == 1:
        clone = im.copy()
        
        classify('D:\\Redbull\\Model\\templates\\zerocal\\redbull1.png', img_gray, clone, 0.94, (0, 0, 255))

        return clone

def sugarfree(shelf_type=1, im=img_rgb):
    if shelf_type == 1:
        clone = im.copy()
        
        classify('D:\\Redbull\\Model\\templates\\sugarfree\\redbull1.png', img_gray, clone, 0.95, (0, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\sugarfree\\redbull2.png', img_gray, clone, 0.75, (0, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\sugarfree\\redbull3.png', img_gray, clone, 0.9, (0, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\sugarfree\\redbull4.png', img_gray, clone, 0.9, (0, 255, 0))
        
        return clone
    
def box4(shelf_type=1, im=img_rgb):
    if shelf_type == 1:
        clone = im.copy()
        
        #classify('D:\\Redbull\\Model\\templates\\box4\\box41.png', img_gray, clone, 0.83, (0, 255, 255))
        #classify('D:\\Redbull\\Model\\templates\\box4\\box42.png', img_gray, clone, 0.85, (0, 255, 255))
        #classify('D:\\Redbull\\Model\\templates\\box4\\box43.png', img_gray, clone, 0.8, (255, 100, 0))
        #classify('D:\\Redbull\\Model\\templates\\box4\\box44.png', img_gray, clone, 0.91, (255, 0, 255))
        classify('D:\\Redbull\\Model\\templates\\box4\\box46.png', img_gray, clone, 0.78, (255, 0, 255))
        
        return clone
        
def box12(shelf_type=1, im=img_rgb):
    if shelf_type == 1:
        clone = im.copy()
        
        #classify('D:\\Redbull\\Model\\templates\\box12\\box121.png', img_gray, clone, 0.85, (0, 200, 200))
        #classify('D:\\Redbull\\Model\\templates\\box12\\box122.png', img_gray, clone, 0.80, (200, 0, 200))
        
        return clone

def cans(shelf_type=1, im=img_rgb):
    if shelf_type == 1:
        clone = regular(shelf_type=1, im=im)
        clone = zerocal(shelf_type=1, im=clone)
        clone = sugarfree(shelf_type=1, im=clone)
        clone = box4(shelf_type=1, im=clone)
        #clone = box12(shelf_type=1, im=clone)
        
        
        classify('D:\\Redbull\\Model\\templates\\cans\\cans1.png', img_gray, clone, 0.95, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans2.png', img_gray, clone, 0.95, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans3.png', img_gray, clone, 0.95, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans4.png', img_gray, clone, 0.95, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans5.png', img_gray, clone, 0.95, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans6.png', img_gray, clone, 0.9, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans7.png', img_gray, clone, 0.9, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans8.png', img_gray, clone, 0.9, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans9.png', img_gray, clone, 0.9, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans10.png', img_gray, clone, 0.9, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans11.png', img_gray, clone, 0.9, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans12.png', img_gray, clone, 0.9, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans13.png', img_gray, clone, 0.8, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans14.png', img_gray, clone, 0.75, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans15.png', img_gray, clone, 0.8, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans16.png', img_gray, clone, 0.7, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans17.png', img_gray, clone, 0.8, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans18.png', img_gray, clone, 0.7, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans19.png', img_gray, clone, 0.8, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans20.png', img_gray, clone, 0.8, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans21.png', img_gray, clone, 0.8, (255, 255, 0))
        classify('D:\\Redbull\\Model\\templates\\cans\\cans22.png', img_gray, clone, 0.8, (255, 255, 0))
        
        
        return clone

fig = plt.figure(figsize=(20, 15))
ax = fig.add_subplot(111)
ax.imshow(cans())
plt.show()
