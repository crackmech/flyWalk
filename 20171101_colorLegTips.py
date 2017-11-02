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
rawDir = "/home/aman/Downloads/opencv/CS_1_20170512_225019_tracked_8/"
csvName = rawDir.rstrip('/')+'.csv'


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

rawList = natural_sort(os.listdir(rawDir))

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

leg = 0
nUpdate = 0
ix = 0
iy= 0


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
def getIm(im, lAngles):
    '''
    input:
        'im' refering to the index of the image to be processed
    skeleton image is produced by plotting circles on the legTips, head, tail, body center
    a line is plotted for the body axis, between head and tail centers
    
    '''
    global img
    img = np.ones(image.shape, dtype='uint8')*222#create an empty iamge with gray background 
    cv2.circle(img,(int(lAngles[18,im]),int(lAngles[19,im])), 5, (0,0,255), thickness=4)#draw a circle on the detected head blobs
    cv2.circle(img,(int(lAngles[21,im]),int(lAngles[22,im])), 2, (100,255,0), thickness=2)#draw a circle on the detected tail blobs
    cv2.circle(img,(int(lAngles[24,im]),int(lAngles[25,im])), 2, (100,255,0), thickness=2)#draw a circle on the detected body  blobs
    cv2.line(img, (int(lAngles[18,im]),int(lAngles[19,im])),(int(lAngles[21,im]),int(lAngles[22,im])),\
                                                                                (0,0,200), thickness=3)# draw a line from head to tail
    for i in xrange(nLegs):
        cv2.circle(img,(int(lAngles[i*3,im]),int(lAngles[(i*3)+1, im])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        cv2.line(img, (int(lAngles[24,im]),int(lAngles[25,im])),(int(lAngles[i*3,im]),int(lAngles[(i*3)+1, im])), tuple(patches[i]))
        cv2.putText(img,legLabels[i], (int(lAngles[i*3,im]),int(lAngles[(i*3)+1, im])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    return img



def getRaw(im):
    '''
    returns the legtip labeled raw image for index 'im'
    '''
    global img, ix, iy
    img = cv2.imread(rawDir+rawList[im])
    for i in xrange(nLegs):
        cv2.circle(img,(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        cv2.putText(img,legLabels[i], (int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    ix = 0
    iy = 0

def button1(buttonState, y):
    global leg
    leg = y
    print leg

def update(x,y):
    '''
    update the legtip coordinate based on manual correction from mouse click on the displayed image
    '''
    global nUpdate
    if ix != 0 and iy !=0:
        im = cv2.getTrackbarPos("trackbar1", 'main1')
        legAngles[leg*3,im] = ix
        legAngles[(leg*3)+1,im] = iy
        getRaw(im)
        nUpdate+=1
        print 'updated '+str(nUpdate)+ ' times'

def save(x,y):
    print x,y
    print 'saved'

def draw_circle(event,x,y,flags,param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        ix = x
        iy = y
        cv2.circle(img,(ix,iy),2,(255,0,0),1)
        print ix, iy

def errorcorrect(legAngles, tolerance, maxstep):
    '''
    input:  
        legAngles:  numpy array containing all leg coordinates, their angles and
                    other body segment coordinates and angle
        tolerance:  the pixel diameter which is okay for jitter in legtip coordinates
        maxstep:    maximum distance moved by a legtip in two consecutive frames
    output:
        correctedLegAngles: numpy array containing corrected values for leg tips
        errorList:          list of frames where the values were corrected
        
    '''
    


legLabels = ['L3','R3','R2','R1','L1','L2']

#set Leg Id for labelin legs, sorted the basis of angle from the tail
L1 = 4
L2 = 5
L3 = 0
R1 = 3
R2 = 2
R3 = 1

restLabels = ['head','tail','body']
image = cv2.imread(imList[1])

imgCenter = (image.shape[0]/2+image.shape[1]/2)/2
la = []

#create an empty array which will store all the values of 
#            a,b) leg blob coordinates for each leg
#            c) leg angle for each leg w.r.t center of the image, columns = 0:nLegs*3 = 6*3 = 0:18
#            d) head blob coordinates, head blob angle w.r.t center of the image columns: 19:21
#            e) tail blob coordinates, tail blob angle w.r.t center of the image, columns: 22:24
#            f) body blob coordinates, body blob angle w.r.t center of the image, columns: 25:27

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





cv2.namedWindow('main1', cv2.WINDOW_GUI_EXPANDED)
#cv2.namedWindow('main2', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('main1',draw_circle)
cv2.createTrackbar("trackbar1", "main1", 0, len(imList)-1, getRaw)

legLabels = ['L3','R3','R2','R1','L1','L2']

cv2.createButton('L1', button1, L1, cv2.QT_RADIOBOX,1 )
cv2.createButton('L2', button1, L2, cv2.QT_RADIOBOX,0 )
cv2.createButton('L3', button1, L3, cv2.QT_RADIOBOX,0)
cv2.createButton('R1', button1, R1, cv2.QT_RADIOBOX,0)
cv2.createButton('R2', button1, R2, cv2.QT_RADIOBOX,0)
cv2.createButton('R3', button1, R3, cv2.QT_RADIOBOX,0)
cv2.createButton('Update', update, 6, cv2.QT_NEW_BUTTONBAR,0 )
cv2.createButton('Save', save, 7, cv2.QT_NEW_BUTTONBAR,0 )

im = 0
img = cv2.imread(rawDir+rawList[0])
for i in xrange(nLegs):
    cv2.circle(img,(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
    cv2.putText(img,legLabels[i], (int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))

while(1):    

    cv2.imshow('main1',img)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
    elif k == ord('1'):#key '1' pressed to select L1
        print k
        leg = L1
        update(0,0)
        print leg
    elif k == ord('2'):#key '2' pressed to select L2
        print k
        leg = L2
        update(0,0)
        print leg
    elif k == ord('3'):#key '3' pressed to select L3
        print k
        leg = L3
        update(0,0)
        print leg
    elif k == ord('q'):#key 'q' pressed to select R1
        print k
        leg = R1
        update(0,0)
        print leg
    elif k == ord('w'):#key 'w' pressed to select R2
        print k
        leg = R2
        update(0,0)
        print leg
    elif k == ord('e'):#key 'e' pressed to select R3
        print k
        leg = R3
        update(0,0)
        print leg
    elif k == 32: #spacebar pressed to update the selected coordiantes
        print k
        update(0,0)
        print leg
    elif k == 83: #right arrow pressed for next image
        print k
        im+=1
        getRaw(im)
        cv2.setTrackbarPos('trackbar1','main1', im)
    elif k == 81:  #left arrow pressed for previous image
        print k
        im-=1
        getRaw(im)
        cv2.setTrackbarPos('trackbar1','main1', im)
#    else:
#        print k
    cv2.waitKey(1)
        
cv2.destroyAllWindows()


print 'done tracking, now displaying'
blk = np.ones(image.shape, dtype='uint8')*222#create an empty iamge with gray background 
for im in xrange(len(imList)):
    image = cv2.imread(imList[im])
    img = getIm(im, legAngles)
    for i in xrange(nLegs):
        cv2.circle(blk,(int(legAngles[i*3,im]),int(legAngles[(i*3)+1, im])), 2, tuple(patches[i]), thickness=1)#draw a circle on the detected leg tip blobs        

    cv2.imshow('123',np.hstack((img, image, blk)))
    cv2.waitKey(30)

cv2.destroyAllWindows()



cv2.imshow('123',blk)
cv2.waitKey()

cv2.destroyAllWindows()









































