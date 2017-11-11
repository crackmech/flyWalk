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

nIm = 32


baseDir = '/media/aman/data/flyWalk_data/LegPainting/unzip/200Frames/'
baseDir = '/media/aman/data/flyWalk_data/LegPainting/unzip/test/'

classifier = os.path.join(baseDir,"20170615_6Classes_updated20171106.model")
#classifier = os.path.join(baseDir,"testClassifier.model")
os.chdir(baseDir)
flist = sorted(glob.glob('*.tiff'))
print os.getcwd()
'''
for i in xrange(len(flist)):
	fname = flist[i]
	rawDir = fname.rstrip('.tiff')
	classDir = rawDir+'_classified'
	os.mkdir(rawDir)
	os.mkdir(classDir)
	imp = IJ.openImage(baseDir+fname)
	#imp.show()
	print i, flist[i], '\n'
	impStack = imp.getStack()
	for k in xrange(nIm):
		IJ.save(ImagePlus('a',impStack.getProcessor(k+1)), os.path.join(baseDir,rawDir, rawDir+str(k)+'.png'));
'''


print 'Loading Classifier at: '+present_time()
segs = ts.WekaSegmentation()
segs.loadClassifier(classifier)
print 'Loaded Classifier at:  '+present_time()

print flist
for i in xrange(len(flist)):
	fname = flist[i]
	rawDir = fname.rstrip('.tiff')
	classDir = rawDir+'_classified'
	os.mkdir(rawDir)
	os.mkdir(classDir)
	imp = IJ.openImage(baseDir+fname)
	nIm = imp.getNSlices()
	#print imp.width, imp.height, imp.getNSlices()
	if imp:
		segIm = segs.applyClassifier(imp, 0, False)
		#segIm.show()
		segImStack = segIm.getStack()
		impStack = imp.getStack()
		for k in xrange(nIm):
			IJ.save(ImagePlus('a',segImStack.getProcessor(k+1)), os.path.join(baseDir, classDir, classDir+str(k)+'.png'));
			IJ.save(ImagePlus('a',impStack.getProcessor(k+1)), os.path.join(baseDir, rawDir, rawDir+str(k)+'.png'));

'''