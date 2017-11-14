#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 07:39:29 2016

@author: aman
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 12:31:48 2016

@author: aman
"""
import cv2
import os
import numpy as np
import re
import sys
from datetime import datetime
from thread import start_new_thread as startNT
import Tkinter as tk
import tkFileDialog as tkd
import zipfile
import matplotlib.pyplot as plt
import time


dirname = '/home/aman/Desktop/testWalk/20161017_200525'
dirname = '/media/aman/data/testWalk/20161017_200525'

initialDir = '/media/flywalk/data/'
initialDir = '/media/aman/data/flyWalk_data/LegPainting/glassChamber'

params = cv2.SimpleBlobDetector_Params()
params.blobColor = 0
params.minThreshold = 5
params.maxThreshold = 120
params.filterByArea = True
params.filterByCircularity = True
params.minCircularity = 0.2
params.filterByConvexity = False
params.filterByInertia = False
params.minArea = 1000
params.maxArea = 5000
hcropBox = 100
vcropBox = 100


def present_time():
        now = datetime.now()
        return now.strftime('%Y%m%d_%H%M%S')

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def getFolder(initialDir):
    '''
    GUI funciton for browsing and selecting the folder
    '''    
    root = tk.Tk()
    initialDir = tkd.askdirectory(parent=root,
                initialdir = initialDir, title='Please select a directory')
    root.destroy()
    return initialDir+'/'

def createTrack(trackData, img):
    '''
    input:
        create an image of shape 'imgShape' with the x,y coordiates of the track from the array 'trackData
    returns:
        an np.array with the cv2 image array, which can be saved or viewed independently of this function
    '''
    #img = np.ones((imgShape[0], imgShape[1], 3), dtype = 'uint8')
    blue = np.hstack((np.linspace(0, 255, num = len(trackData[0])/2),np.linspace(255, 0, num = (len(trackData[0])/2)+1)))
    green = np.linspace(255, 0, num = len(trackData[0]))
    red = np.linspace(0, 255, num = len(trackData[0]))
    cv2.putText(img,'Total frames: '+str(len(trackData[0])), (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    for i in xrange(1,len(trackData[0])):
        cv2.circle(img,(int(trackData[0,i]), int(trackData[1,i])), 2, (blue[i], green[i], red[i]), thickness=2)#draw a circle on the detected body blobs
    for i in xrange(1,len(trackData[0])):
        if i%100==0:
            cv2.putText(img,'^'+str(i), (int(trackData[0,i]), int(trackData[1,i])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
    #cv2.imshow('track', img); cv2.waitKey(); cv2.destroyAllWindows()
    return img


def getMotion(imData, avg, weight):
    if avg is None:
        print "[INFO] starting background model..."
        avg = imData.copy().astype("float")
    cv2.accumulateWeighted(imData, avg, weight)
    return cv2.absdiff(imData, cv2.convertScaleAbs(avg)), avg


def getBgIm(dirname, imList, imgs):
    '''
    returns a background Image for subtraction from all the images using weighted average
    
    '''
#    weight = 1.0/len(imList)
#    img = imgs[:,:,0]
#    avg = np.zeros(img.shape, dtype='float')
#    for i in xrange(len(imList)):
#        imData = imgs[:,:,i]
#        cv2.accumulateWeighted(imData, avg, weight)
    #cv2.imshow('avg', avg); cv2.waitKey(); cv2.destroyAllWindows()
    avg = np.array((np.median(imgs, axis=2)), dtype = 'uint8')
    return cv2.convertScaleAbs(avg)#cv2.imread(dirname+'/'+imList[0],cv2.IMREAD_GRAYSCALE)

    
def tracknCrop_display(dirname):
    print dirname
    flist = natural_sort(os.listdir(dirname))[:256]
    print len(flist)    
    img = cv2.imread(dirname+'/'+flist[0],cv2.IMREAD_GRAYSCALE)
    imgs = np.zeros((img.shape[0], img.shape[1], len(flist)), dtype = 'uint8')
    for i in xrange(len(flist)):
        imgs[:,:,i] = cv2.imread(dirname+'/'+flist[i],cv2.IMREAD_GRAYSCALE)

    im = []
    bgIm = getBgIm(dirname, flist, imgs)
    for f in range(0, len(flist)):
        im = imgs[:,:,f]
        img = cv2.bitwise_not(cv2.absdiff(im, bgIm))
        eq = (img-img.min())*(255/img.min())
        img = np.vstack((eq, img))
        cv2.imshow('123',img)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    return im


def tracknCrop(dirname):
    trackDir = dirname+"_tracked/"
    try:
        os.mkdir(trackDir)
    except:
        pass
    
    flist = natural_sort(os.listdir(dirname))
    print '\nprocessing %i frames in\n==> %s'%(len(flist), dirname)
    detector = cv2.SimpleBlobDetector_create(params)
    
#    print "Started at "+present_time()
    y = True
    trackData = np.zeros((2,len(flist)))
    saveDir = trackDir+"temp/"
    saveDir_cropped = trackDir+"temp_cropped/"
    try:
        os.mkdir(saveDir)
        os.mkdir(saveDir_cropped)
    except:
        pass
    if len(flist)==0:
        pass
    else:
        img = cv2.imread(dirname+'/'+flist[0],cv2.IMREAD_GRAYSCALE)
        imgs = np.zeros((img.shape[0], img.shape[1], len(flist)), dtype = 'uint8')
        startTime1 = time.time()
        for i in xrange(len(flist)):
            imgs[:,:,i] = cv2.imread(dirname+'/'+flist[i],cv2.IMREAD_GRAYSCALE)
        bgIm = getBgIm(dirname, flist, imgs)
        print('Read Images in: %0.3f seconds'%(time.time()-startTime1))
        startTime2 = time.time()
        im = []
        threshold = 20
        dilateKernel = 2
        kernel = np.ones((dilateKernel, dilateKernel), np.uint8)# to dilate
        for f in range(0, len(flist)):
    #        if f%1000==0:
    #            sys.stdout.write("\rAt %s Processing File: %d"%(present_time(),f))
    #            sys.stdout.flush()
            im = imgs[:,:,f]
            keypoints = detector.detect(im)
            kp = None
            try:
                for kp in keypoints:
                    pts = [int(kp.pt[1])-hcropBox, int(kp.pt[1])+hcropBox, int(kp.pt[0])-hcropBox,int(kp.pt[0])+hcropBox]
                    #im_cropped = im[pts[0]:pts[1], pts[2]:pts[3]]
                    
                    im_cropped = im[pts[0]:pts[1], pts[2]:pts[3]]
#                    im_cropped = cv2.dilate(im[pts[0]:pts[1], pts[2]:pts[3]], kernel, iterations=1 )
#                    mask = (cv2.dilate(cv2.absdiff(im[pts[0]:pts[1], pts[2]:pts[3]], bgIm[pts[0]:pts[1], pts[2]:pts[3]]), kernel, iterations=1 )>threshold)#bg subtraction
#                    imcropped = (mask*im_cropped)/255.0
#                    cv2.imshow('img', im_cropped); cv2.waitKey(1)
#                    cv2.imshow('imgMask', imcropped); cv2.waitKey(1)
                    trackData[:,f] = (kp.pt[0],kp.pt[1])
                    y=True
                if im_cropped.shape >= (vcropBox*2):
                    cv2.imwrite(saveDir+flist[f], im)
#                    imcropped = cv2.bitwise_not(im_cropped)
#                    im_cropped = (imcropped-imcropped.min())*(255.0/(imcropped.max()-imcropped.min()))#fix the contrast
                    #cv2.imshow('img', im_cropped); cv2.waitKey(1)
                    cv2.imwrite(saveDir_cropped+flist[f].rstrip('.jpeg')+'.png', im_cropped)
                else:
                    raise ValueError()
            except:
                pass
#                if (y == True and kp == None):
#                    saveDir = trackDir+"temp_"+str(nDir)+'/'
#                    saveDir_cropped = trackDir+"temp_cropped_"+str(nDir)+'/'
#                    try:
#                        os.mkdir(saveDir)
#                        os.mkdir(saveDir_cropped)
#                    except:
#                        pass
#                    nDir+=1
#                    y=False
        cv2.destroyAllWindows()
        print('Processed %i Images in %0.3f seconds\nAverage total processing speed: %05f FPS'\
                %(len(flist), time.time()-startTime2, (len(flist)/(time.time()-startTime1)))) 
        fname = dirname+"_trackData_"+rawDir
        if im!=[]:
            trackImg = createTrack(trackData, cv2.imread(dirname+'/'+flist[0]))
            cv2.imwrite(fname+'.jpeg', trackImg)
        np.savetxt(fname+".csv",np.transpose(trackData), fmt='%.3f', delimiter = ',', header = 'X-Coordinate, Y-Coordinate')
        print "Finished processing directory at "+present_time()
        dirs = natural_sort([ name for name in os.listdir(trackDir) if os.path.isdir(os.path.join(trackDir, name)) ])
        os.chdir(trackDir)
        for d in dirs:
            if len(os.listdir(d))<100:
                for f in os.listdir(d):
                    os.remove(d+"/"+f)
                os.rmdir(d)
            else:
    #            print d, len(os.listdir(d))
                zf = zipfile.ZipFile(d+".zip", "w")
                for dirnames, subdirs, files in os.walk(d):
                    zf.write(dirnames)
                    for filenames in files:
                        zf.write(os.path.join(dirnames, filenames))
                zf.close()
                for f in os.listdir(d):
                    os.remove(d+"/"+f)
                os.rmdir(d)
    
        return trackData

'''

zf = zipfile.ZipFile("myzipfile.zip", "w")
for dirname, subdirs, files in os.walk("mydirectory"):
    zf.write(dirname)
    for filename in files:
        zf.write(os.path.join(dirname, filename))
zf.close()
'''
baseDir = getFolder(initialDir)
rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])

print "Started processing directories at "+present_time()
for rawDir in rawdirs:
#    print rawDir
#    print "----------Processing directoy: "+os.path.join(baseDir,rawDir)+'--------'
    d = os.path.join(baseDir,rawDir,'imageData')
    imdirs = natural_sort([ name for name in os.listdir(d) if os.path.isdir(os.path.join(d, name)) ])
    for imdir in imdirs:
        if 'tracked' in imdir:
            pass
        else:
            t = tracknCrop(os.path.join(d,imdir))
#            t = tracknCrop_display(os.path.join(d,imdir))


#import math
#dirname = initialDir+'/'+rawDir+'/imageData/'+imdirs[1]
#imList = natural_sort(os.listdir(dirname))
#aa = getBgIm(dirname, imList)
#
#for i in xrange(len(imList)):
#    im = cv2.imread(dirname+'/'+imList[i], cv2.IMREAD_GRAYSCALE)
#    #cv2.imshow('np', im); cv2.waitKey(100)
#    cv2.imshow('np', cv2.absdiff(im, cv2.convertScaleAbs(aa))); cv2.waitKey(10)
#cv2.destroyAllWindows()
#







