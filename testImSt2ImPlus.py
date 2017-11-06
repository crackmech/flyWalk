#run("Image Sequence...", "open=/media/aman/data/flyWalk_data/LegPainting/unzip/20170512_235710_tracked/7083.png number=20 sort");
#saveAs("Tiff", "/media/aman/data/flyWalk_data/LegPainting/unzip/20170512_235710_tracked_20.tif");


# Walk recursively through an user-selected directory
# and add all found filenames that end with ".tif"
# to a VirtualStack, which is then shown.
#
# It is assumed that all images are of the same type
# and have the same dimensions.
 
import os
from ij.io import DirectoryChooser
from ij import IJ, ImagePlus, ImageStack, VirtualStack
import re

nIms = [32, 100, 200]


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def saveTiff(srcDir, nIms):
	flist = natural_sort(os.listdir(srcDir))
	imp = IJ.openImage(os.path.join(srcDir, flist[0]))
	vs = ImageStack(imp.width, imp.height)
	for i in xrange(nIms):
		vs.addSlice(str(i),IJ.openImage(os.path.join(srcDir, flist[i])).getProcessor())
	IJ.save(ImagePlus("Stack from subdirectories", vs), srcDir.rstrip('/')+'_'+str(nIms)+'.tiff');


srcDir = DirectoryChooser("Choose!").getDirectory()
rawdirs = natural_sort([ name for name in os.listdir(srcDir) if os.path.isdir(os.path.join(srcDir, name)) ])

for raws in rawdirs:
	for i in nIms:
		saveTiff(os.path.join(srcDir, raws), i)
'''
outputDir = DirectoryChooser("Choose!").getDirectory()
flist = natural_sort(os.listdir(srcDir))
path = os.path.join(srcDir, flist[0])
imp = IJ.openImage(os.path.join(srcDir, flist[0]))
#imp.show()
vs = ImageStack(imp.width, imp.height)
nIms = 20

for i in xrange(nIms):
	print srcDir, flist[i]
	vs.addSlice(str(i),IJ.openImage(os.path.join(srcDir, flist[i])).getProcessor())
	#vs.addSlice(str(i),IJ.openImage(os.path.join(srcDir, flist[i])))

ims = ImagePlus("Stack from subdirectories", vs)
ims.show()
for i in xrange(nIms):
	IJ.save(ImagePlus('a',vs.getProcessor(i+1)), outputDir+flist[i]);
'''



'''
def run():
  srcDir = DirectoryChooser("Choose!").getDirectory()
  if not srcDir:
    # user canceled dialog
    return
  # Assumes all files have the same size
  vs = None
  for root, directories, filenames in os.walk(srcDir):
    for filename in filenames:
      # Skip non-TIFF files
      if not filename.endswith(".tif"):
        continue
      path = os.path.join(root, filename)
      # Upon finding the first image, initialize the VirtualStack
      if vs is None:
        imp = IJ.openImage(path)
        vs = VirtualStack(imp.width, imp.height, None, srcDir)
      # Add a slice, relative to the srcDir
      vs.addSlice(path[len(srcDir):])
  #
  ImagePlus("Stack from subdirectories", vs).show()
 
run()
'''