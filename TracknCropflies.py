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



dirname = '/home/aman/Desktop/testWalk/20161017_200525'
dirname = '/media/aman/data/testWalk/20161017_200525'

initialDir = '/media/flywalk/data/'
initialDir = '/media/aman/data/flyWalk_data/LegPainting/media (2)/flywalk/data/delete/glassChamber'
#initialDir = '/media/aman/data/flyWalk_data/LegPainting/media/flywalk/data/delete/glassChamber'


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
cropBox =100


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

def tracknCrop(dirname):
    print dirname
    trackDir = dirname+"_tracked/"
    try:
        os.mkdir(trackDir)
    except:
        pass
    
    flist = natural_sort(os.listdir(dirname))
    detector = cv2.SimpleBlobDetector_create(params)
    
#    print "Started at "+present_time()
    y = True
    nDir = 1
    trackData = np.zeros((2,len(flist)))
    saveDir = trackDir+"temp/"
    saveDir_cropped = trackDir+"temp_cropped/"
    try:
        os.mkdir(saveDir)
        os.mkdir(saveDir_cropped)
    except:
        pass
    im = []
    for f in range(0, len(flist)):
#        if f%1000==0:
#            sys.stdout.write("\rAt %s Processing File: %d"%(present_time(),f))
#            sys.stdout.flush()
        im = cv2.imread(dirname+'/'+flist[f],cv2.IMREAD_GRAYSCALE)
        keypoints = detector.detect(im)
        try:
            for kp in keypoints:
                im_cropped = im[int(kp.pt[1])-cropBox:int(kp.pt[1])+cropBox,\
                                int(kp.pt[0])-cropBox:int(kp.pt[0])+cropBox]
                trackData[:,f] = (kp.pt[0],kp.pt[1])
                cv2.imwrite(saveDir+flist[f], im)
            y=True
            if im_cropped.size == cropBox*cropBox*4:
                cv2.imwrite(saveDir_cropped+flist[f], im_cropped)
                im_cropped=0
            else:
                raise ValueError()
        except:
            if y==True:
                saveDir = trackDir+"temp_"+str(nDir)+'/'
                saveDir_cropped = trackDir+"temp_cropped_"+str(nDir)+'/'
                try:
                    os.mkdir(saveDir)
                    os.mkdir(saveDir_cropped)
                except:
                    pass
                nDir+=1
                y=False
    fname = dirname+"_trackData_"+rawDir
    if im!=[]:
        trackImg = createTrack(trackData, cv2.imread(dirname+'/'+flist[0]))
        cv2.imwrite(fname+'.jpeg', trackImg)
    np.savetxt(fname+".csv",np.transpose(trackData), fmt='%.3f', delimiter = ',', header = 'X-Coordinate, Y-Coordinate')
    print "\ndone "+dirname+" at "+present_time()
    dirs = natural_sort([ name for name in os.listdir(trackDir) if os.path.isdir(os.path.join(trackDir, name)) ])
    os.chdir(trackDir)
#    print os.getcwd()
#    print len(dirs)
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

for rawDir in rawdirs:
#    print rawDir
#    print "----------Processing directoy: "+os.path.join(baseDir,rawDir)+'--------'
    d = os.path.join(baseDir,rawDir,'imageData')
    imdirs = natural_sort([ name for name in os.listdir(d) if os.path.isdir(os.path.join(d, name)) ])
    for imdir in imdirs:
        t = tracknCrop(os.path.join(d,imdir))



