# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 05:31:58 2017

@author: aman
"""

import numpy as np
import cv2
from datetime import datetime
import sys
from thread import start_new_thread as startNT
import os
import tkFileDialog as tkd
import Tkinter as tk
import matplotlib.pyplot as plt
from multiprocessing.dummy import Pool as ThreadPool 
import shutil
from glob import glob
import re
from math import atan2, degrees


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


imDir = "/home/aman/Downloads/opencv/CS_1_20170512_225019_tracked_8_6Classes/"

headColor = [10, 225, 255]
tailColor = [255, 226, 84]
bodyColor = [0, 0, 255]
LegColor  = [130, 255, 79]
legTipcolor = [193, 121, 255]
bgColor = [255, 118, 198]

params = cv2.SimpleBlobDetector_Params()
params.blobColor = 255
params.minThreshold = 5
params.maxThreshold = 255
params.filterByArea = True
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False
params.minArea = 0.1
params.maxArea = 10000
cropBox =100

detector = cv2.SimpleBlobDetector_create(params)


imList = natural_sort(os.listdir(imDir))

os.chdir(imDir)

nClasses = 6
nLegs = 6

legDilate = 1
othersDilate = 5
colors = [
            headColor,
            tailColor,
            bodyColor,
            LegColor,
            legTipcolor,
            bgColor
]
colorDic = {
    'headColor' : headColor,
    'tailColor' : tailColor,
    'bodyColor' : bodyColor,
    'LegColor'  : LegColor,
    'legTipcolor' : legTipcolor,
    'bgColor' : bgColor
}

col = 150
patches = [
            [col,0,0],
            [0,col,0],
            [0,0,col],
            [col,0,col],
            [0, col, col],
            [col,col,0],
            [0,col,0],
            [0,col,0]
            ]

def getBlob(image, dilateKernel, color, nblob):
    '''
    input:
        image: image to be segmented
        dilatKernek: Kernel size for dilating the extracted points having given colors
        color: color to be segmented
        nBlob: number of blobs to be extracted, based on size, starting from maximum size first
    returns:
        an array of blobs containing: 
            0: blob size
            1: blob X coordinate
            2: blob Y coordinate
    '''
    im = np.array(((image[:,:,0] == color[0]) &\
                    (image[:,:,1] == color[1]) &\
                    (image[:,:,2] == color[2]))*255, dtype = 'uint8')
    kernel = np.ones((dilateKernel, dilateKernel), np.uint8)# to dilate
    im = cv2.dilate(im, kernel, iterations=1)
    keypoints = detector.detect(im)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    blobs = []
    for i in xrange(len(keypoints)):
        blobs.append([keypoints[i].size, keypoints[i].pt[0],keypoints[i].pt[1]]) #(size, x-coordiante, y-coordinate)
#    print '==========================='
    blobs = np.array(blobs)
    blobs = (blobs[blobs[:,0].argsort()])#sort by size of the blob
    blobs = blobs[-nblob:]# select only the highest size blobs for nLegs
#    print blobs
    return blobs

legLabels = ['L3','R3','R2','R1','L1','L2']

restLabels = ['head','tail','body']
image = cv2.imread(imList[1])

imgCenter = (image.shape[0]/2+image.shape[1]/2)/2
la = []

#create an empty array which will store all the values of 
#            a) leg blobs coordinates b))leg blob angle w.r.t center of the image
#            c) head blob coordinates d) head blob angle w.r.t center of the image
#            e) tail blob coordinates d) tail blob angle w.r.t center of the image
#            g) body blob coordinates d) body blob angle w.r.t center of the image

# array size would be: (4*3, nLegs, len(imList)) (number of values per leg, number of legs, total frames)

legAngles = np.zeros(((nLegs+3)*3, len(imList)), dtype='float32')

legAnglesHeader = [legLabels[0]+'_x', legLabels[0]+'_y', legLabels[0]+'_angle', 
                   legLabels[1]+'_x', legLabels[1]+'_y', legLabels[1]+'_angle', 
                   legLabels[2]+'_x', legLabels[2]+'_y', legLabels[2]+'_angle', 
                   legLabels[3]+'_x', legLabels[3]+'_y', legLabels[3]+'_angle', 
                   legLabels[4]+'_x', legLabels[4]+'_y', legLabels[4]+'_angle', 
                   legLabels[5]+'_x', legLabels[5]+'_y', legLabels[5]+'_angle', 
                   restLabels[0]+'_x', restLabels[0]+'_y', restLabels[0]+'_angle', 
                   restLabels[1]+'_x', restLabels[1]+'_y', restLabels[1]+'_angle', 
                   restLabels[2]+'_x', restLabels[2]+'_y', restLabels[2]+'_angle', 
                  ]

for im in xrange(len(imList)):
    image = cv2.imread(imList[im])
    legBlobs = getBlob(image, legDilate, colors[4],nLegs)
    headBlobs = getBlob(image, othersDilate, colors[0],1)
    tailBlobs = getBlob(image, othersDilate, colors[1],1)
    bodyBlobs = getBlob(image, othersDilate, colors[2],1)
    #after getting all the blobs and their coordiantes, now find out their respective angles from the center of the image
    if len(legBlobs)!= nLegs:
        print im, len(legBlobs), len(headBlobs), len(tailBlobs), len(bodyBlobs)
    headAngle = degrees(atan2(headBlobs[0][1]-imgCenter, headBlobs[0][2]-imgCenter))
    tailAngle = degrees(atan2(tailBlobs[0][1]-imgCenter, tailBlobs[0][2]-imgCenter))
    bodyAngle = degrees(atan2(bodyBlobs[0][1]-imgCenter, bodyBlobs[0][2]-imgCenter))
    angles = np.zeros((nLegs,3), dtype = 'float')
    for i in xrange(len(legBlobs)):
        angles[i,:2] = legBlobs[i, 1:3] # insert the original blob values, (size, x-coordinate, y-coordinate)
        angles[i,2] = degrees(atan2(legBlobs[i][1]-imgCenter, legBlobs[i][2]-imgCenter))# insert leg angle w.r.t the origin
    angles = angles[angles[:,2].argsort()]#sort the legAngles according to the angle values
    
    for i in xrange(len(legBlobs)):
#        legAngles[i*3+2,i,im] = angles[i,#store leg tip angle in third index
        legAngles[i*3:(i*3)+3,im] = angles[i] # insert the original blob x-coordinate, y-coordinate
        
    legAngles[18:20,im] = headBlobs[0][1:] # insert the original blob x-coordinate, y-coordinate
    legAngles[20,im] = headAngle # insert the original blob x-coordinate, y-coordinate

    legAngles[21:23,im] = tailBlobs[0][1:] # insert the original blob x-coordinate, y-coordinate
    legAngles[23,im] = tailAngle # insert the original blob x-coordinate, y-coordinate

    legAngles[24:26,im] = bodyBlobs[0][1:] # insert the original blob x-coordinate, y-coordinate
    legAngles[26,im] = bodyAngle # insert the original blob x-coordinate, y-coordinate

print 'done tracking, now displaying'
for im in xrange(len(imList)):
    img = np.ones(image.shape, dtype='uint8')*222#create an empty iamge with gray background 
    cv2.circle(img,(int(legAngles[18,im]),int(legAngles[19,im])), 5, (0,0,255), thickness=4)#draw a circle on the detected head blobs
    cv2.circle(img,(int(legAngles[21,im]),int(legAngles[22,im])), 2, (100,255,0), thickness=2)#draw a circle on the detected tail blobs
    cv2.circle(img,(int(legAngles[24,im]),int(legAngles[25,im])), 2, (100,255,0), thickness=2)#draw a circle on the detected body  blobs
    cv2.line(img, (int(legAngles[18,im]),int(legAngles[19,im])),(int(legAngles[21,im]),int(legAngles[22,im])),\
                                                                                (0,0,200), thickness=3)# draw a line from head to tail
    for i in xrange(nLegs):
        cv2.circle(img,(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        cv2.line(img, (int(legAngles[24,im]),int(legAngles[25,im])),(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), tuple(patches[i]))
        cv2.putText(img,legLabels[i], (int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    cv2.imshow('123',np.hstack((img, image)))
    cv2.waitKey(30)

cv2.destroyAllWindows()





def getIm(im):
    global img
    img = np.ones(image.shape, dtype='uint8')*222#create an empty iamge with gray background 
    cv2.circle(img,(int(legAngles[18,im]),int(legAngles[19,im])), 5, (0,0,255), thickness=4)#draw a circle on the detected head blobs
    cv2.circle(img,(int(legAngles[21,im]),int(legAngles[22,im])), 2, (100,255,0), thickness=2)#draw a circle on the detected tail blobs
    cv2.circle(img,(int(legAngles[24,im]),int(legAngles[25,im])), 2, (100,255,0), thickness=2)#draw a circle on the detected body  blobs
    cv2.line(img, (int(legAngles[18,im]),int(legAngles[19,im])),(int(legAngles[21,im]),int(legAngles[22,im])),\
                                                                                (0,0,200), thickness=3)# draw a line from head to tail
    for i in xrange(nLegs):
        cv2.circle(img,(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        cv2.line(img, (int(legAngles[24,im]),int(legAngles[25,im])),(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), tuple(patches[i]))
        cv2.putText(img,legLabels[i], (int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    return img


print 'done tracking, now displaying'
for im in xrange(len(imList)):
    image = cv2.imread(imList[im])
    img = getIm(im)
    cv2.imshow('123',np.hstack((img, image)))
    cv2.waitKey(30)

cv2.destroyAllWindows()
    


def getRaw(im):
    global img
    img = cv2.imread(rawDir+rawList[im])
    for i in xrange(nLegs):
        cv2.circle(img,(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        cv2.putText(img,legLabels[i], (int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))


def button1(buttonState, y):
    print buttonState, y

def button2():
    print 'button2'


ix = 0
iy= 0
def draw_circle(event,x,y,flags,param):
    global ix, iy
    ix = x
    iy = y
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(im,(ix,iy),100,(255,0,0),-1)
        print ix, iy


cv2.namedWindow('main1', cv2.WINDOW_GUI_EXPANDED)
#cv2.namedWindow('main2', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('main1',draw_circle)
cv2.createTrackbar("trackbar1", "main1", 0, len(imList)-1, getRaw)

cv2.createButton('button1', button1, 0, cv2.QT_RADIOBOX,0 )
cv2.createButton('button2', button1, 1, cv2.QT_RADIOBOX,0 )
cv2.createButton('button3', button1, 2, cv2.QT_RADIOBOX,1)
cv2.createButton('button4', button1, 3, cv2.QT_RADIOBOX,1)
cv2.createButton('button5', button1, 4, cv2.QT_RADIOBOX,1)
cv2.createButton('button6', button1, 5, cv2.QT_RADIOBOX,1)
cv2.createButton('button7', button1, 6, cv2.QT_CHECKBOX,0 )
cv2.createButton('button8', button1, 7, cv2.QT_CHECKBOX,0 )

im = 0
img = cv2.imread(rawDir+rawList[0])
for i in xrange(nLegs):
    cv2.circle(img,(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
    cv2.putText(img,legLabels[i], (int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))

while(1):    

    cv2.imshow('main1',img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()




























































