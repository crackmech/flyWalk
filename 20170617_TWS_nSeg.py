#import ij;
from ij import IJ
import ij.process;
import trainableSegmentation as ts;
import os;
import re
from datetime import datetime

def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


baseDir = '/home/vijay/amanaggarwal/flyWalk/WEKA/'
inputDir = baseDir+"/CS_1_20170512_225508_tracked_0/";
outputDir = baseDir+"tmp/sc/";
classifier = baseDir+"20170615_6Classes.model";

step = 3
#classifier1 = "/home/aman/Desktop/classifier.model";
flist = natural_sort(os.listdir(inputDir));
print len(flist);

def segment_(fname,inputDir, classifier, outputDir, step):
	#ts.WekaSegmentation().loadClassifier(classifier);
	inps = []
	for i in xrange(step):
		inp = IJ.openImage(inputDir+fname[i])
#		inp = IJ.openImage(inputDir+fname);
		segmentator = ts.WekaSegmentation(inp);
		segmentator.loadClassifier(classifier);
		segmentator.applyClassifier(False);
		result = segmentator.getClassifiedImage();
		IJ.save(segmentator.getClassifiedImage(), outputDir+fname+".png");

#segment(fname, inputDir, classifier, outputDir);
def segment(fname,inputDir, classifier, outputDir, step):
	for i in xrange(step):
		print fname[i]
		inp = IJ.openImage(inputDir+fname[i]);
		segmentator = ts.WekaSegmentation(inp);
		if i==0:
			segmentator.loadClassifier(classifier);
		segmentator.applyClassifier(False);
		result = segmentator.getClassifiedImage();
		print 'saving image'
		IJ.save(segmentator.getClassifiedImage(), outputDir+fname[i]+".png");

segment(flist[0:3], inputDir, classifier, outputDir, step);
#
#for i in xrange(0, 100,10):#)len(flist), step):
#	print i+1
#	segment(flist[i], inputDir, classifier, outputDir, step);
print 'done'