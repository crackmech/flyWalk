//import ij.*;
//import ij.process.*;
//import trainableSegmentation.*;
run("Trainable Weka Segmentation", "open=/home/vijay/amanaggarwal/flyWalk/WEKA/tmp/tmp/1467.jpeg.jpeg.png");
selectWindow("Trainable Weka Segmentation v3.2.13");
call("trainableSegmentation.Weka_Segmentation.loadClassifier", "/home/vijay/amanaggarwal/flyWalk/WEKA/20170615_6Classes.model");
call("trainableSegmentation.Weka_Segmentation.applyClassifier", "/home/vijay/amanaggarwal/flyWalk/WEKA/greys_11", "greys_11_000.png", "showResults=true", "storeResults=false", "probabilityMaps=false", "");
/*
n=1;

baseDir = "/home/vijay/amanaggarwal/flyWalk/WEKA/"
inputDir = baseDir+"/CS_1_20170512_225508_tracked_0/";
outputDir = baseDir+"tmp/sc/";
classifier = baseDir+"20170615_6Classes.model";

flist = getFileList(inputDir);
//print (flist);

//run("Trainable Weka Segmentation", "open=/home/aman/Desktop/OK371_20150518_1200_06-18_1_0-1.tif");

//selectWindow("Trainable Weka Segmentation v3.1.2");
//call("trainableSegmentation.Weka_Segmentation.loadClassifier", "/home/aman/Desktop/classifier.model");
processFile(inputDir, flist, 0, outputDir,n);//flist.length)


function processFile(inputDir, list, fileNumber, outputDir, n){
	call("trainableSegmentation.Weka_Segmentation.loadClassifier", classifier);
	j=n;
	i=0;
	print(inputDir);
	print(list[fileNumber+i]);
	for (i=0;i<j;i++){
		call("trainableSegmentation.Weka_Segmentation.applyClassifier", inputDir, list[fileNumber+i], "showResults=true", "storeResults=true", "probabilityMaps=true", outputDir);
}
}

//call("trainableSegmentation.Weka_Segmentation.applyClassifier", "/home/aman/Desktop/OK371_20150518_1200_06-18_1_2", "_2015-05-24-184352-0407.raw.jpeg", "showResults=false", "storeResults=true", "probabilityMaps=true", "/home/aman/Desktop/tmp_20160924");

//suffix = ".jpeg";

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


/*processFolder(inputDir,n);

*/







*?