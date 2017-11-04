
baseDir = '/home/vijay/amanaggarwal/flyWalk/WEKA/';
inputDir = baseDir+"/CS_1_20170512_225508_tracked_0/";
outputDir = baseDir+"tmp/sc/";

baseDir = 'D:\\flyWalk_data\\LegPainting\\test\\';
inputDir = baseDir+"\\temp_cropped\\";
outputDir = baseDir+"sc\\";



classifier = baseDir+"20170615_6Classes.model";
classifier = baseDir+"testClassifier.model";

flist = getFileList(inputDir);


processFile(inputDir, flist, 0, outputDir,flist.length);

n=10;

function processFile(inputDir, list, fileNumber, outputDir, n){
	call("trainableSegmentation.Weka_Segmentation.loadClassifier", classifier);
	j=n;
	for (i=0;i<j;i++){
		call("trainableSegmentation.Weka_Segmentation.applyClassifier", inputDir, list[fileNumber+i], "showResults=false", "storeResults=true", "probabilityMaps=true", outputDir);
}
}

function processFolder(inputDir,n) {
	list = getFileList(inputDir);
	for (i = 0; i < list.length; i=i+n) {
		if(File.isDirectory(inputDir + list[i]))
			processFolder("" + inputDir + list[i]);
		if(endsWith(list[i], suffix))
//			print([list[i]+"list[i+1],list[i+2],list[i+3],list[i+4]);
			processFile(inputDir, list, i, outputDir,n);
	}
}




//processFolder(inputDir,n);









