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
import copy

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


imDir = "/home/aman/Downloads/opencv/CS_1_20170512_225019_tracked_8_6Classes/"
rawDir = "/home/aman/Downloads/opencv/CS_1_20170512_225019_tracked_8/"

baseDir = "D:\\flyWalk_data\\LegPainting\\unzip\\processed\\20170513_001948\\"
rawDir = baseDir+"raw\\"
imDir = baseDir+"classified\\"

baseDir = '/media/aman/data/flyWalk_data/LegPainting/test/'
baseDir = '/media/aman/data/flyWalk_data/LegPainting/unzip/processed/20170513_001948/'

rawDir = baseDir+"raw/";
imDir = baseDir+"classified/";

#imDir = "/home/aman/Downloads/opencv/CS_1_20170512_225019_tracked_8_6Classes/"
#rawDir = "/home/aman/Downloads/opencv/CS_1_20170512_225019_tracked_8/"


imlist = os.listdir(rawDir)

csvName = rawDir.rstrip('/')+'.csv'

legLabels = ['L3','L2','L1','R1','R2','R3']

#legLabels = ['L1','R1','R2','R3','L3','L2']

legLabels = ['L3','L2','L1','R1','R2','R3']

#set Leg Id for labeling legs, sorted the basis of angle from the tail
L1 = 2
L2 = 1
L3 = 0
R3 = 5
R2 = 4
R1 = 3



headColor = [10, 225, 255]
tailColor = [255, 226, 84]
bodyColor = [0, 0, 255]
LegColor  = [130, 255, 79]
legTipcolor = [193, 121, 255]
bgColor = [255, 118, 198]


##headColor = [153,153,153]
##tailColor = [204,204,204]
##bodyColor = [0, 0, 0]
##LegColor  = [51,51,51]
##legTipcolor = [255, 255, 255]
##bgColor = [102,102,102]
##


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
    im = np.array(((image[:,:,0] == color[0]) &\
                    (image[:,:,1] == color[1]) &\
                    (image[:,:,2] == color[2]))*255, dtype = 'uint8')
    kernel = np.ones((dilateKernel, dilateKernel), np.uint8)# to dilate
    im = cv2.dilate(im, kernel, iterations=1)
    keypoints = detector.detect(im)
    blobs = []
    for i in xrange(len(keypoints)):
        blobs.append([keypoints[i].size, keypoints[i].pt[0],keypoints[i].pt[1]]) #(size, x-coordiante, y-coordinate)
#    print '==========================='
    if blobs !=[]:
        blobs = np.array(blobs)
        blobs = (blobs[blobs[:,0].argsort()])#sort by size of the blob
        blobs = blobs[-nblob:]# select only the highest size blobs for nLegs
    else:
        print 'no blobs detected'
        blobs = [[10,100,100]]
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
    cv2.circle(img,(int(lAngles[im, headCol]),int(lAngles[im, headCol+1])), 5, (0,0,255), thickness=4)#draw a circle on the detected head blobs
    cv2.circle(img,(int(lAngles[im, tailCol]),int(lAngles[im, tailCol+1])), 2, (100,255,0), thickness=2)#draw a circle on the detected tail blobs
    cv2.circle(img,(int(lAngles[im, bodyCol]),int(lAngles[im, bodyCol+1])), 2, (100,255,0), thickness=2)#draw a circle on the detected body  blobs
    cv2.line(img, (int(lAngles[im, headCol]),int(lAngles[im, headCol+1])),\
                    (int(lAngles[im, tailCol]),int(lAngles[im, tailCol+1])),(0,0,200), thickness=3)# draw a line from head to tail
    for i in xrange(nLegs):
        cv2.circle(img,(int(lAngles[im, i*nParams]),int(lAngles[im, (i*nParams)+1])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        cv2.line(img, (int(lAngles[im, bodyCol]),int(lAngles[im, bodyCol+1])),(int(lAngles[im, i*nParams]),int(lAngles[im, (i*nParams)+1])), tuple(patches[i]))
        cv2.putText(img,legLabels[i], (int(lAngles[im, i*nParams]),int(lAngles[im, (i*nParams)+1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    return img

fontAngle = 0.4

def getRaw(im):
    '''
    returns the legtip labeled raw image for index 'im'
    '''
    global img, ix, iy
    img = cv2.imread(rawDir+rawList[im])
    cv2.circle(img,(int(legAngles[im, headCol]),int(legAngles[im, headCol+1])), 5, (0,0,255), thickness=4)#draw a circle on the detected head blobs
    cv2.circle(img,(int(legAngles[im, tailCol]),int(legAngles[im, tailCol+1])), 2, (100,255,0), thickness=2)#draw a circle on the detected tail blobs
    cv2.circle(img,(int(legAngles[im, bodyCol]),int(legAngles[im, bodyCol+1])), 2, (100,255,0), thickness=2)#draw a circle on the detected body  blobs
    imOrig = img.copy()
    for i in xrange(nLegs):
        cv2.circle(img,(int(legAngles[im, i*nParams]),int(legAngles[im, (i*nParams)+1])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        #cv2.putText(img,legLabels[i]+str(legAngles[im, i*(nParams+1)-1]), (int(legAngles[im, i*nParams]),int(legAngles[im, (i*nParams)+1])), cv2.FONT_HERSHEY_COMPLEX, fontAngle, tuple(patches[i]))
    #cv2.putText(img,'H'+str(legAngles[im, headCol+nParams-1]), (int(legAngles[im, headCol]),int(legAngles[im, headCol+1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    #cv2.putText(img,'T'+str(legAngles[im, tailCol+nParams-1]), (int(legAngles[im, tailCol]),int(legAngles[im, tailCol+1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    #cv2.line(img, (int(legAngles[im, headCol]),int(legAngles[im, headCol+1])),\
     #               (int(legAngles[im, tailCol]),int(legAngles[im, tailCol+1])),(0,0,200), thickness=3)# draw a line from head to tail

    for i in xrange(nLegs):
        cv2.circle(imOrig,(int(lA[im, i*nParams]),int(lA[im, (i*nParams)+1])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
        #cv2.putText(imOrig,legLabels[i]+str(lA[im, i*(nParams+1)-1]), (int(lA[im, i*nParams]),int(lA[im, (i*nParams)+1])), cv2.FONT_HERSHEY_COMPLEX, fontAngle, tuple(patches[i]))
    #cv2.putText(imOrig,'H'+str(lA[im, headCol+nParams-1]), (int(lA[im, headCol]),int(lA[im, headCol+1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
    #cv2.putText(imOrig,'T'+str(lA[im, tailCol+nParams-1]), (int(lA[im, tailCol]),int(lA[im, tailCol+1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))

    img = np.hstack((img, imOrig))
    ix = 0
    iy = 0
    return img

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
        legAngles[im, leg*nParams] = ix
        legAngles[im, (leg*nParams)+1] = iy
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

def getOtherParams(otherCoords, refCoords):
    '''
    input: 
        otherCoords:   blob centroid coordinates (x,y)
        refCoords:     tail centroid coordinates   (x,y)
        origin:         coordinates of the origin, i.e. center of the image (x,y)
    output:
        calcParams:    a numpy array containing coordinates, Eucledian Distance and angle (x, y, euDis, theta)
    '''
    angle = (degrees(atan2(otherCoords[2]-refCoords[2], otherCoords[1]-refCoords[1])))#+360)%360# insert leg angle w.r.t the origin
    euDis = np.sqrt((np.square(otherCoords[2]-refCoords[2])+np.square(otherCoords[1]-refCoords[1])))
    return np.array([otherCoords[1], otherCoords[2], euDis, angle], dtype='float32')

def getLegTipParams(legTipCoords, headCoords, bodyCoords):
    '''
    input: 
        legTipCoords:   legtip centroid coordinates (x,y)
        headCoords:     head centroid coordinates   (x,y)
        refCoords:      coordinates of the origin, i.e. center of the image (x,y)
    output:
        legTipAngle:    a numpy array containing coordinates, Eucledian Distance and angle (x, y, euDis, theta)
    '''
    refCoords = bodyCoords
    legTipAngle = (degrees(atan2(legTipCoords[2]-refCoords[2], legTipCoords[1]-refCoords[1])))#+360)%360# insert leg angle w.r.t the origin
    headAngle = (degrees(atan2(headCoords[2]-refCoords[2], headCoords[1]-refCoords[1])))#+360)%360# insert leg angle w.r.t the origin
    euDis = np.sqrt((np.square(legTipCoords[2]-refCoords[2])+np.square(legTipCoords[1]-refCoords[1])))
    ang = legTipAngle-headAngle
    if ang>180:
        ang=ang-360
    elif ang<-180:
        ang=ang+360
    return np.array([legTipCoords[1], legTipCoords[2], euDis, ang], dtype='float32')

            
def getlegData(leg, legDataArray, nParams):
    '''
    returns the legData from legAngles array for given leg
    '''
    return legDataArray[:, leg*nParams:(leg+1)*nParams]


def getConsectAngles(legData, anglesIndex):
    '''
    returns the array of angle difference between same leg blobs in consecutive frames
    '''
    angleArray = np.zeros((len(legData)-1))
    for i in range(len(angleArray)):
        ang = legData[i+1,anglesIndex]-legData[i,anglesIndex]
        if ang>180:
            ang=ang-360
        elif ang<-180:
            ang=ang+360
        angleArray[i] = ang
    avConsAngle = np.average(angleArray)
    stdevConsAngle = np.std(angleArray)

    return angleArray, avConsAngle, stdevConsAngle

def getOutliers(angleArray, avConsAngle, stdevConsAngle):
    '''
    returns the array with outliers removed (on the basis of average angleDiff and STDEV)
            and a list of outliers indices
    '''
    low = avConsAngle-(stdevConsAngle)
    high = avConsAngle+(stdevConsAngle)
    outliers = np.where(np.logical_or(angleArray<=low, angleArray>=high))
    print [x for x in outliers]
    if len(outliers[0])>0:
        outliers = outliers[0]
    else: 
        outliers = []
    return outliers

def getAvgAngle(legData, outliers):
    '''
    returns the average and STEDV of angles of a leg afer removing the outliers
    '''
    array = legData[:,len(legData[0])-1]
    print array.shape, np.average(legData, axis=0)
    arr = np.delete(array, outliers, axis=0)
    print arr.shape
    return np.average(arr), np.std(arr)

def errorCorrect(legData, im, avAngle, stdAng, tolerance):#, avEuDis, stdEuDis):
    '''
    returns the error corrected values of legtip based on the average and STDEV of distance and angle of that legtip 
            Recalculates all the blobs, finds out the closest blob for the given legtip using average angle and distances
    '''
    angleCol = len(legData[0])-1
    image = cv2.imread(imList[im])
    legBlobs = getBlob(image, legDilate, colors[4],nLegs)
    headBlobs = getBlob(image, othersDilate, colors[0],1)
    bodyBlobs = getBlob(image, othersDilate, colors[2],1)
    blobs = np.zeros((len(legBlobs), len(legData[0])))
    for i in xrange(len(legBlobs)):
        blobs[i, :] = getLegTipParams(legTipCoords = legBlobs[i], headCoords = headBlobs[0], bodyCoords = bodyBlobs[0])
    blobs[:,angleCol] = np.abs(blobs[:,angleCol]-avAngle)
    blobs = blobs[blobs[:,angleCol].argsort()]
    if blobs[0,angleCol]<=tolerance*stdAng:
        return blobs[0]
    else:
        return legData[im-1]



def legTipCorrect(legData, nLegs, nParams, tolerance):
    '''
    returns the error corrected legAngle data for all legs
    '''
    for leg in xrange(nLegs):
        lData = getlegData(leg, legData, nParams)
        anglesIndex = nParams-1
        angleArray, avConsAngle, stdConsAng = getConsectAngles(lData, anglesIndex)
        outliers = getOutliers(angleArray, avConsAngle, stdConsAng)
        avAngle, stdAngle = getAvgAngle(lData, outliers)
        for i in xrange(len(outliers)):
            lData[outliers[i]+1] = errorCorrect(lData, outliers[i]+1, avAngle, stdAngle, tolerance)            
#        n=0
#        while (outliers!=[] and n<len(lData)):
#            print outliers[0], n
#            n+=1
#            lData[outliers[0]+1] = errorCorrect(lData, outliers[0]+1, avConsAngle, stdConsAng)
#            #lData[outliers[0]+1] = lData[outliers[0]]
#            angleArray, _, _ = getConsectAngles(lData, anglesIndex)
#            outliers= getOutliers(angleArray, avConsAngle, stdConsAng)
        legData[:, leg*nParams:(leg+1)*nParams] = lData
    return legData



##imDir = getFolder(imDir)
##rawDir = getFolder(rawDir)

imList = natural_sort(os.listdir(imDir))
rawList = natural_sort(os.listdir(rawDir))
os.chdir(imDir)



# parameters for automated error correction
tolerance = 1
maxDis = 30
delAngle = 40
updateWin = 2



restLabels = ['head','tail','body']
image = cv2.imread(imList[1])

imgCenter = (image.shape[0]/2+image.shape[1]/2)/2
origin = [[0, imgCenter, imgCenter]]
la = []

#create an empty array which will store all the values of 
#            a,b) leg blob coordinates for each leg
#            c) leg angle for each leg w.r.t center of the image, columns = 0:nLegs*3 = 6*3 = 0:18
#            d) head blob coordinates, head blob angle w.r.t center of the image columns: 19:21
#            e) tail blob coordinates, tail blob angle w.r.t center of the image, columns: 22:24
#            f) body blob coordinates, body blob angle w.r.t center of the image, columns: 25:27

# array size would be: (4*3, nLegs, len(imList)) (number of values per leg, number of legs, total frames)

image = cv2.imread(imList[0])
legBlobs = getBlob(image, legDilate, colors[4],nLegs)
headBlobs = getBlob(image, othersDilate, colors[0],1)
tailBlobs = getBlob(image, othersDilate, colors[1],1)
bodyBlobs = getBlob(image, othersDilate, colors[2],1)
#nParams = total parameters calculated for each blob, such as X,y, euclediandistance, angle etc.
nParams = len((getLegTipParams(legTipCoords = legBlobs[0], headCoords = headBlobs[0], bodyCoords = bodyBlobs[0])))
print 'Total parameters to be calculated: '+str(nParams)#after getting all the blobs and their coordiantes, now find out their respective angles from the center of the image

otherBlobs = 3 # count of other blobs to be tracked, viz head, tail , body
legAngles = np.zeros((len(imList), (nLegs+otherBlobs)*nParams), dtype='float32')

headCol = nLegs*nParams
tailCol = (nLegs+1)*nParams
bodyCol = (nLegs+2)*nParams

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
    if len(legBlobs)!= nLegs:
        print im, len(legBlobs), len(headBlobs), len(tailBlobs), len(bodyBlobs)
    angles = np.zeros((nLegs,nParams), dtype = 'float')
    for i in xrange(len(legBlobs)):
        angles[i,:] = getLegTipParams(legTipCoords = legBlobs[i], headCoords = headBlobs[0], bodyCoords = bodyBlobs[0])
    angles = angles[angles[:,nParams-1].argsort()]#sort the legAngles according to the angle values
    
    for i in xrange(len(legBlobs)):
        legAngles[im, i*nParams:(i*nParams)+nParams] = angles[i] # insert the original blob x-coordinate, y-coordinate
    legAngles[im, headCol:tailCol] = getOtherParams(headBlobs[0], origin[0])#insert headBlob parameters
    legAngles[im, tailCol:bodyCol] = getOtherParams(tailBlobs[0], origin[0])#insert tailBlob parameters
    legAngles[im, bodyCol:bodyCol+nParams] = getOtherParams(bodyBlobs[0], origin[0])#insert bodyBlob parameters


delAngle = [np.std(legAngles[:,i*2]) for i in xrange(nLegs)]
lA = legAngles.copy()
#legAngles = errorcorrect(legAngles, tolerance, delAngle, nLegs, updateWin)
legAngles = legTipCorrect(legAngles, nLegs, nParams, tolerance)

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
print img.shape
getRaw(im)
print img.shape
#for i in xrange(nLegs):
#    cv2.circle(img,(int(legAngles[im, i*nParams]),int(legAngles[im, (i*nParams)+1])), 3, tuple(patches[i]), thickness=2)#draw a circle on the detected leg tip blobs        
#cv2.putText(img,'H'+str(legAngles[im, headCol+nParams-1]), (int(legAngles[im, headCol]),int(legAngles[im, headCol+1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))
#cv2.putText(img,'T'+str(legAngles[im, tailCol+nParams-1]), (int(legAngles[im, tailCol]),int(legAngles[im, tailCol+1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, tuple(patches[i]))

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
blk = np.ones(image.shape, dtype='uint8')*bgValue#create an empty iamge with gray background 
for im in xrange(len(imList)):
    image = cv2.imread(imList[im])
    img = getIm(im, legAngles)
    for i in xrange(nLegs):
        cv2.circle(blk,(int(legAngles[im, i*nParams]),int(legAngles[im, (i*nParams)+1,])), 2, tuple(patches[i]), thickness=1)#draw a circle on the detected leg tip blobs        

    cv2.imshow('123',np.hstack((img, image, blk)))
    cv2.waitKey(30)

cv2.destroyAllWindows()



cv2.imshow('123',blk)
cv2.waitKey()

cv2.destroyAllWindows()


dis = np.zeros((len(imList),nLegs))

for l in xrange(nLegs):
    for i in xrange(1,len(imList)):
        dis[i,l] = np.sqrt(np.square(legAngles[i-1, l*nParams]-legAngles[i,l*nParams])+np.square(legAngles[i-1, (l*nParams)+1]-legAngles[i,(l*nParams)+1]))


l=0
plt.plot(dis[:, L1]+(l*30), label = 'L1')
l+=1
plt.plot(dis[:, L2]+(l*30), label = 'L2')
l+=1
plt.plot(dis[:, L3]+(l*30), label = 'L3')
l+=1
plt.plot(dis[:, R1]+(l*30), label = 'R1')
l+=1
plt.plot(dis[:, R2]+(l*30), label = 'R2')
l+=1
plt.plot(dis[:, R3]+(l*30), label = 'R3')
plt.yticks([0, 30, 60, 90, 120, 150],["L1", 'L2', 'L3', 'R1', 'R2', 'R3'])

#plt.legend(loc='right', bbox_to_anchor=(1.08,0.5),
#          fancybox=True, shadow=True)
#plt.legend(loc = 'upper center')
plt.show()

plt.close()



l=0
plt.plot(legAngles[:, (L1*3)+2], label = 'L1')
l+=1
plt.plot(legAngles[:, (L2*3)+2], label = 'L2')
l+=1
plt.plot(legAngles[:, (L3*3)+2], label = 'L3')
l+=1
plt.plot(legAngles[:, (R1*3)+2], label = 'R1')
l+=1
plt.plot(legAngles[:, (R2*3)+2], label = 'R2')
l+=1
plt.plot(legAngles[:, (R3*3)+2], label = 'R3')
#plt.yticks([0, 30, 60, 90, 120, 150],["L1", 'L2', 'L3', 'R1', 'R2', 'R3'])
plt.legend(loc = 'lower left', ncol=nLegs,bbox_to_anchor=(-0.05,-0.15),
          fancybox=True, shadow=True).draggable()
plt.show()



plt.close()





'''
def errorcorrection:
    for each leg, get x,y, euDis, angle
        find angle change between two consecutive steps for all frames
        find average and STDEV angle diff between two consecutive steps
        remove outliers from whole stack, save their frame number and leg number
        find blobs for the frame again and check if there is any blob centroid between angle+-STDEV
            If yes, then use that blob as replacement for the current blob,
                To check if the newly found blob can be used as replacement, use angle and euDis as measures of closeness
            Otherwise, use previous points as the coordinates for the current blob

    

'''


leg = 0


def showLeg(leg, legData):
    lData = legData[:, leg*nParams:(leg+1)*nParams]
    blk = np.ones(image.shape, dtype='uint8')*bgValue#create an empty iamge with gray background 
    for im in xrange(len(imList)):
        cv2.circle(blk,(int(lData[im, 0]),int(lData[im, 1])), 2, tuple(patches[1]), thickness=-1)#draw a circle on the detected leg tip blobs        
        cv2.imshow('123',blk)
        cv2.waitKey(30)
    
    cv2.destroyAllWindows()
n=0
showLeg(n, legAngles)

#cv2.imshow('123',blk)
#cv2.waitKey()
#
#cv2.destroyAllWindows()




















