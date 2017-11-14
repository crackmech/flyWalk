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


def present_time():
        return datetime.now().strftime('%Y%m%d_%H%M%S')


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

baseDir = '/media/aman/data/flyWalk_data/LegPainting/test/'
baseDir = '/media/aman/data/flyWalk_data/LegPainting/unzip/frames/test/'
#srcDir = DirectoryChooser("Choose!").getDirectory()
#if srcDir:
#	baseDir = srcDir
#else:
#	pass
classifier = os.path.join(baseDir,"20170615_6Classes_updated20171106_3.model")
nIm = 8 # number of images to be processed per batch

rawdirs = natural_sort([ name for name in os.listdir(baseDir) if os.path.isdir(os.path.join(baseDir, name)) ])

print 'Loading Classifier at: '+present_time()
segs = ts.WekaSegmentation()
segs.loadClassifier(classifier)
print 'Loaded Classifier at:  '+present_time()

for r in rawdirs:
	inputDir = os.path.join(baseDir, r);
	outputDir = inputDir+'_classified'
	try:
	    os.mkdir(outputDir)
	except:
	    pass
	flist = natural_sort(os.listdir(inputDir))[:64]
	print len(flist);
	
	images = []
	for i in xrange(len(flist)):
		imp = IJ.openImage(os.path.join(inputDir, flist[i]))
		if imp:
			images.append(imp)
	
	for i in xrange(0, len(flist)-(len(flist)%nIm), nIm):
		if i%(10*nIm)==0:
			print i, present_time()
		imStack = ImageStack(imp.width, imp.height)
		for j in xrange(nIm):
			imStack.addSlice(str(i+j),images[i+j].getProcessor())
		imp = ImagePlus('images',imStack)
		imp.show()
		if imp:
			segIm = segs.applyClassifier(imp, 0, False)
			segImStack = segIm.getStack()
			segIm.show()
			for k in xrange(nIm):
				IJ.save(ImagePlus('a',segImStack.getProcessor(k+1)), os.path.join(outputDir,flist[i+k]));

		#IJ.save(segIm, outputDir+flist[i]);

print 'Finished Segmentation at: '+present_time() 

#IJ.run("Quit")


