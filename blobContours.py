# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 06:28:04 2017

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


imDir = "/media/aman/data/flyWalk_data/LegPainting/test/CNN/cm/temp_cropped_42_results/"
rawDir = "/media/aman/data/flyWalk_data/LegPainting/test/CNN/cm/temp_cropped_42/"
csvName = rawDir.rstrip('/')+'.csv'

legLabels = ['L3','L2','L1','R1','R2','R3']

#legLabels = ['L1','R1','R2','R3','L3','L2']

legLabels = ['L3','L2','L1','R1','R2','R3']

#set Leg Id for labelin legs, sorted the basis of angle from the tail
L1 = 2
L2 = 1
L3 = 0
R3 = 5
R2 = 4
R1 = 3



#headColor = [10, 225, 255]
#tailColor = [255, 226, 84]
#bodyColor = [0, 0, 255]
#LegColor  = [130, 255, 79]
#legTipcolor = [193, 121, 255]
#bgColor = [255, 118, 198]


headColor = [1, 138, 58]
tailColor = [126, 1, 177]
bodyColor = [126, 1, 177]
LegColor  = [0, 0, 1]
legTipcolor = [0, 0, 1]
bgColor = [1, 30, 172]

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

nClasses = 6
nLegs = 6

legDilate = 2
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

col = 220
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
bgValue = 0
leg = 0
nUpdate = 0
ix = 0
iy= 0

def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+os.sep


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
#    cv2.imshow('123', image); cv2.waitKey(); cv2.destroyAllWindows()
#    print color
    im = np.array(((image[:,:,0] == color[0]) &\
                    (image[:,:,1] == color[1]) &\
                    (image[:,:,2] == color[2]))*255, dtype = 'uint8')
    kernel = np.ones((dilateKernel, dilateKernel), np.uint8)# to dilate
    im = cv2.dilate(im, kernel, iterations=1)
#    cv2.imshow('123', im); cv2.waitKey(); cv2.destroyAllWindows()
    keypoints = detector.detect(im)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    blobs = []
    for i in xrange(len(keypoints)):
        blobs.append([keypoints[i].size, keypoints[i].pt[0],keypoints[i].pt[1]]) #(size, x-coordiante, y-coordinate)
#    print '==========================='
    blobs = np.array(blobs)
#    print len(blobs)
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
    img = np.ones(image.shape, dtype='uint8')*bgValue#create an empty iamge with gray background 
    cv2.circle(img,(int(lAngles[im, 18]),int(lAngles[im, 19])), 5, (0,0,255), thickness=4)#draw a circle on the detected head blobs
    cv2.circle(img,(int(lAngles[im, 21]),int(lAngles[im, 22])), 2, (100,255,0), thickness=2)#draw a circle on the detected tail blobs
    cv2.circle(img,(int(lAngles[im, 24]),int(lAngles[im, 25])), 2, (100,255,0), thickness=2)#draw a circle on the detected body  blobs
    cv2.line(img, (int(lAngles[im, 18]),int(lAngles[im, 19])),(int(lAngles[im, 21]),int(lAngles[im, 22])),\
                                                                                (0,0,200), thickness=3)# draw a line from head to tail
    for i in xrange(nLegs):
        cv2.circle(img,(int(lAngles[im, i*3]),int(lAngles[im, (i*3)+1])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        cv2.line(img, (int(lAngles[im, 24]),int(lAngles[im, 25])),(int(lAngles[im, i*3]),int(lAngles[im, (i*3)+1])), tuple(patches[i]))
        cv2.putText(img,legLabels[i], (int(lAngles[im, i*3]),int(lAngles[im, (i*3)+1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    return img

fontAngle = 0.4

def getRaw(im):
    '''
    returns the legtip labeled raw image for index 'im'
    '''
    global img, ix, iy
    img = cv2.imread(rawDir+rawList[im])
    cv2.putText(img,'H'+str(legAngles[im, 20]), (int(legAngles[im, 18]),int(legAngles[im, 19])), cv2.FONT_HERSHEY_COMPLEX, fontAngle, (0,0,255))
    cv2.putText(img,'T'+str(legAngles[im, 23]), (int(legAngles[im, 21]),int(legAngles[im, 22])), cv2.FONT_HERSHEY_COMPLEX, fontAngle, (0,0,255))
    for i in xrange(nLegs):
        cv2.circle(img,(int(legAngles[im, i*3]),int(legAngles[im, (i*3)+1])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        cv2.putText(img,legLabels[i]+str(legAngles[im, (i*3)+2]), (int(legAngles[im, i*3]),int(legAngles[im, (i*3)+1])), cv2.FONT_HERSHEY_COMPLEX, fontAngle, tuple(patches[i]))
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
        legAngles[im, leg*3] = ix
        legAngles[im, (leg*3)+1] = iy
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

def getLegtipAngle(legTipCoords, headCoords, tailCoords, bodyCoords):
    '''
    input: 
        legTipCoords:   legtip centroid coordinates (x,y)
        tailCoords:     tail centroid coordinates   (x,y)
        origin:         coordinates of the origin, i.e. center of the image (x,y)
    output:
        legTipAngle:    a numpy array containing legtip coordinates and angle (x,y,theta)
    '''
    legTipAngle = (degrees(atan2(legTipCoords[1]-bodyCoords[1], legTipCoords[0]-bodyCoords[0])))#+360)%360# insert leg angle w.r.t the origin
    headAngle = (degrees(atan2(headCoords[1]-bodyCoords[1], headCoords[0]-bodyCoords[0])))#+360)%360# insert leg angle w.r.t the origin
    return np.array([legTipCoords[0], legTipCoords[1], (legTipAngle-headAngle)], dtype='float32')

def neighbourUpdate(inArray):
    '''
    input: 
        the array from which the erroraneous values are corrected and updated
        the index of the row which needs to be updated would be: (len(inArray)-1/2)+1
    output:
        the array with the updated values calculated from the neighbouring cells
        
    method: the erroraneous value is calculated using the average of neighbouring cells
            more sophisticated methods such as heuristics/machine learning could also applied
    '''
    errorIndex = (len(inArray)-1)/2
    tempCoords = np.delete(inArray, (errorIndex), axis=0)

    return np.average(tempCoords, axis = 0)


def errorcorrect(lAngles, tolerance, maxDis, nLegs, updateWin):
    '''
    input:  
        lAngles:  numpy array containing all leg coordinates, their angles and
                    other body segment coordinates and angle
        tolerance:  the pixel diameter which is okay for jitter in legtip coordinates
        maxDis:    maximum distance moved by a legtip in two consecutive frames
        nLegs:      total number of legtip data present in the lAngles file
        updateWin:  the window of neighbouring cells from which the erroraneous value would be corrected 
    output:
        correctedLegAngles: numpy array containing corrected values for leg tips
        errorList:          list of frames where the values were corrected
        
    '''
    for i in xrange(updateWin, len(lAngles)-updateWin):
        for l in xrange(nLegs):
            x = lAngles[i, l*3]
            y = lAngles[i, (l*3)+1]
            if (i>updateWin or i<(len(lAngles)-updateWin)):
                preX = lAngles[i-1, l*3]
                preY = lAngles[i-1, (l*3)+1]
                dis = np.sqrt(np.square(x-preX)+np.square(y-preY))
                if dis>maxDis:    
                    print i, l, dis, '========================================='
                    lAngles[i,:] = neighbourUpdate(lAngles[i-updateWin:(i+updateWin)+1,:])
    return lAngles
            
imDir = getFolder(imDir)
rawDir = getFolder(rawDir)

imList = natural_sort(os.listdir(imDir))
rawList = natural_sort(os.listdir(rawDir))
os.chdir(imDir)



# parameters for automated error correction
tolerance = 3
maxDis = 20
updateWin = 2



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

legAngles = np.zeros((len(imList), (nLegs+3)*3), dtype='float32')

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
    legBlobs = getBlob(image, legDilate, colors[3], nLegs)
    headBlobs = getBlob(image, othersDilate, colors[0],1)
    tailBlobs = getBlob(image, othersDilate, colors[1],1)
    bodyBlobs = getBlob(image, othersDilate, colors[2],1)
    #after getting all the blobs and their coordiantes, now find out their respective angles from the center of the image
    if len(legBlobs)!= nLegs:
        print im, len(legBlobs), len(headBlobs), len(tailBlobs), len(bodyBlobs)
    angles = np.zeros((nLegs,3), dtype = 'float')
    for i in xrange(len(legBlobs)):
        angles[i,:] = getLegtipAngle(legTipCoords = legBlobs[i, 1:3], headCoords= headBlobs[0], tailCoords = tailBlobs[0], bodyCoords = bodyBlobs[0])
    angles = angles[angles[:,2].argsort()]#sort the legAngles according to the angle values
    
    for i in xrange(len(legBlobs)):
        legAngles[im, i*3:(i*3)+3] = angles[i] # insert the original blob x-coordinate, y-coordinate
        
    legAngles[im, 18:20] = headBlobs[0][1:] # insert the original blob x-coordinate, y-coordinate
    legAngles[im, 20] = degrees(atan2(headBlobs[0][1]-imgCenter, headBlobs[0][2]-imgCenter)) # insert the original blob x-coordinate, y-coordinate

    legAngles[im, 21:23,] = tailBlobs[0][1:] # insert the original blob x-coordinate, y-coordinate
    legAngles[im, 23] = degrees(atan2(tailBlobs[0][1]-imgCenter, tailBlobs[0][2]-imgCenter)) # insert the original blob x-coordinate, y-coordinate

    legAngles[im, 24:26] = bodyBlobs[0][1:] # insert the original blob x-coordinate, y-coordinate
    legAngles[im, 26] = degrees(atan2(bodyBlobs[0][1]-imgCenter, bodyBlobs[0][2]-imgCenter)) # insert the original blob x-coordinate, y-coordinate


#legAngles = errorcorrect(legAngles, tolerance, maxDis, nLegs, updateWin)


cv2.namedWindow('main1', cv2.WINDOW_GUI_EXPANDED)
#cv2.namedWindow('main2', cv2.WINDOW_AUTOSIZE)
cv2.setMouseCallback('main1',draw_circle)
cv2.createTrackbar("trackbar1", "main1", 0, len(imList)-1, getRaw)

#legLabels = ['L3','R3','R2','R1','L1','L2']

##cv2.createButton('L1', button1, L1, cv2.QT_RADIOBOX,1 )
##cv2.createButton('L2', button1, L2, cv2.QT_RADIOBOX,0 )
##cv2.createButton('L3', button1, L3, cv2.QT_RADIOBOX,0)
##cv2.createButton('R1', button1, R1, cv2.QT_RADIOBOX,0)
##cv2.createButton('R2', button1, R2, cv2.QT_RADIOBOX,0)
##cv2.createButton('R3', button1, R3, cv2.QT_RADIOBOX,0)
##cv2.createButton('Update', update, 6, cv2.QT_NEW_BUTTONBAR,0 )
##cv2.createButton('Save', save, 7, cv2.QT_NEW_BUTTONBAR,0 )

im = 0
img = cv2.imread(rawDir+rawList[0])
for i in xrange(nLegs):
    cv2.circle(img,(int(legAngles[im, i*3]),int(legAngles[im, (i*3)+1])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
    cv2.putText(img,'H'+str(legAngles[im, 20]), (int(legAngles[im, 18]),int(legAngles[im, 19])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    cv2.putText(img,'T'+str(legAngles[im, 23]), (int(legAngles[im, 21]),int(legAngles[im, 22])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))

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

kernel = np.ones((legDilate, legDilate), np.uint8)# to dilate

print 'done tracking, now displaying'
blk = np.ones(image.shape, dtype='uint8')*bgValue#create an empty iamge with gray background 
for im in xrange(len(imList)):
    image = cv2.imread(imList[im])
    image = cv2.dilate(image, kernel, iterations=1)
    img = getIm(im, legAngles)
    for i in xrange(nLegs):
        cv2.circle(blk,(int(legAngles[im, i*3]),int(legAngles[im, (i*3)+1,])), 2, tuple(patches[i]), thickness=1)#draw a circle on the detected leg tip blobs        
        cv2.circle(image,(int(legAngles[im, i*3]),int(legAngles[im, (i*3)+1,])), 2, tuple(patches[i]), thickness=1)#draw a circle on the detected leg tip blobs        

    cv2.imshow('123',np.hstack((img, image, blk)))
    cv2.waitKey(300)

cv2.destroyAllWindows()


