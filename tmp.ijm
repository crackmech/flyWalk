run("Trainable Weka Segmentation", "open=/home/vijay/amanaggarwal/flyWalk/WEKA/tmp/tmp/1467.jpeg.jpeg.png");
selectWindow("Trainable Weka Segmentation v3.2.13");
call("trainableSegmentation.Weka_Segmentation.loadClassifier", "/home/vijay/amanaggarwal/flyWalk/WEKA/20170615_6Classes.model");
call("trainableSegmentation.Weka_Segmentation.applyClassifier", "/home/vijay/amanaggarwal/flyWalk/WEKA/greys_11", "greys_11_000.png", "showResults=true", "storeResults=false", "probabilityMaps=false", "");
