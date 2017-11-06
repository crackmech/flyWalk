import ij;
from ij import IJ
IJ.run("Trainable Weka Segmentation", "open=./images/icon.png");
from ij import ImageStack
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

baseDir = '/media/aman/data/flyWalk_data/LegPainting/test/'
inputDir = baseDir+"temp_cropped/";
outputDir = baseDir+"sc/";



#baseDir = 'D:\\flyWalk_data\\LegPainting\\test\\'
#inputDir = baseDir+"\\temp_cropped\\";
#outputDir = baseDir+"sc\\";

classifier = baseDir+"20170615_6Classes.model";


classifier = baseDir+"20170615_6Classes_updated20171106.model"
#classifier = baseDir+"testClassifier.model";

try:
    os.mkdir(outputDir)
except:
    pass

flist = natural_sort(os.listdir(inputDir));
print len(flist);


print 'Loading Classifier at: '+present_time()
segs = ts.WekaSegmentation()
segs.loadClassifier(classifier)
print 'Loaded Classifier at:  '+present_time()

images = []
for i in xrange(len(flist)):
	imp = IJ.openImage(inputDir+flist[i])
	if imp:
		images.append(imp)

for i in xrange(len(flist)):
	imp = images[i]
	if imp:
		segIm = segs.applyClassifier(imp, 6, False)
		IJ.save(segIm, outputDir+flist[i]);

print present_time() 

print 'Finished Segmentation'

IJ.run("Quit")


