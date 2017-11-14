import ij;
from ij import IJ, ImagePlus, ImageStack
IJ.run("Trainable Weka Segmentation", "open=./images/icon.png");
from ij import ImageStack
import ij.process;
import trainableSegmentation as ts;
import os;
import re
from datetime import datetime
from ij.io import DirectoryChooser
import glob

def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


baseDir = '/media/aman/data/flyWalk_data/LegPainting/unzip/200Frames/'

classifier = os.path.join(baseDir,"20170615_6Classes_updated20171106_3.model")

#baseDir = '/media/aman/data/flyWalk_data/LegPainting/unzip/test/'
#classifier = os.path.join(baseDir,"20170615_6Classes_updated20171106_3.model")


os.chdir(baseDir)
flist = sorted(glob.glob('*.tiff'))
print os.getcwd()

print 'Loading Classifier at: '+present_time()
segs = ts.WekaSegmentation()
segs.loadClassifier(classifier)
print 'Loaded Classifier at:  '+present_time()

for i in xrange(len(flist)):
	print i, flist[i]

for i in xrange(len(flist)):
	fname = flist[i]
	rawDir = fname.rstrip('.tiff')
	classDir = rawDir+'_classified'
	os.mkdir(rawDir)
	os.mkdir(classDir)
	imp = IJ.openImage(baseDir+fname)
	nIm = imp.getNSlices()
	print i, present_time(), imp.width, imp.height, nIm
	if imp:
		segIm = segs.applyClassifier(imp, 0, False)
		#segIm.show()
		segImStack = segIm.getStack()
		impStack = imp.getStack()
		for k in xrange(nIm):
			IJ.save(ImagePlus('a',segImStack.getProcessor(k+1)), os.path.join(baseDir, classDir, classDir+str(k)+'.png'));
			IJ.save(ImagePlus('a',impStack.getProcessor(k+1)), os.path.join(baseDir, rawDir, rawDir+str(k)+'.png'));

'''