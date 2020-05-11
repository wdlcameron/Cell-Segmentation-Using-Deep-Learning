var grid_size = newArray(4,4);
var results_filename;
var bounding_boxes = 5;
var num_classes = 1;
var currentlyIdentifying = 1;  //Change if you are identifying something else (e.g. 1 for cell, 2 for mitochondria);

macro "Batch Measure" {


	setBatchMode(false);

	dir = getDirectory("Choose a Directory ");
    list = getFileList(dir);

	truncationCount = 0;
	dirName = substring(dir, 0, lengthOf(dir)-1);
	while (!endsWith(dirName, "\\"))
	{
	truncationCount = truncationCount + 1;
	dirName = substring(dirName, 0, lengthOf(dirName)-1);
	}
	dirName = substring(dir, lengthOf(dir)-truncationCount-1, lengthOf(dir)-1);
	results_filename = dirName + ".labels";

    for (i=0; i<list.length; i++) {
        path = dir+list[i];
		sub_path = list[i];
		
        showProgress(i, list.length);
        if (endsWith(path,"/"))		// If path is a directory, recurse into the directory
            {
            	RecurseDirectory(dir, sub_path);
            }

  
	//waitForUser("Done!", "Done upper loop!");
 	while (nImages>0) { 
          selectImage(nImages); 
          close(); 
      } 
}

waitForUser("Done", "Done Processing Images");
}




function RecurseDirectory (dir, sub_path)
{
	dir2 = dir+sub_path;
	listR = getFileList(dir2);
    
    for (i=0; i<listR.length; i++) {
        path2 = dir2+listR[i];
        //waitForUser(path2);
        sub_path2 = sub_path + listR[i];
        showProgress(i, listR.length);
        
        if (endsWith(path2,"/"))		// If path is a directory, recurse into the directory
            {
            	RecurseDirectory(dir, sub_path2);
            }


        else if (endsWith(path2,"ome.tif"))
		{
			open(path2);
        	if (nImages>=1) {
				CreateLabels(dir, sub_path); }  
		}        

    }
    
    
	//waitForUser("Done!", "Done!");
 	while (nImages>0) { 
          selectImage(nImages); 
          close(); 
      } 
}



//note: Import the required arrays if necessary
function CreateLabels(dir, sub_path){

FileName=getTitle();
getDimensions(width, height, channels, slices, frames);


//Find the height and the width of an individual grid
grid_box_Height = height/grid_size[0];
grid_box_Width = width/grid_size[1];

labels_array = newArray((grid_size[0]*grid_size[1]*bounding_boxes)*(1+4+num_classes));
grid_counter_array = newArray(grid_size[0] * grid_size[1]);
	            
FileName=replace (FileName, ".ome.tif", "");



if (isOpen("ROI Manager")==0)
	run("ROI Manager...");

roiManager("reset");

roiManager("Show All");
pathROI = dir+sub_path+FileName;
roiFile = pathROI+"--ROI.zip";
labelsFile = pathROI + "--labels.txt";


//Prepare the label file
//delete the file if it already exists
if (File.exists(labelsFile))
	{
		File.delete(labelsFile);
	}
File.append("Class,X,Y,W,H", labelsFile);
if (File.exists(roiFile))
	roiManager("open",roiFile);

append_String = "";


if (roiManager("count")!=0)
{
	Array.fill(labels_array, 0);
	Array.fill(grid_counter_array, 0);
	

	
	for(j=0; j<roiManager("count"); j++)
	{
		append_String = FileName;
		roiManager("Select", j);
	
		Roi.getBounds(x,y,w,h);
		X_Centroid = (x+w/2);
		Y_Centroid = (y+h/2);

		array_length = 5;
		results_array = newArray(array_length);
		results_array [0] = currentlyIdentifying;
		results_array [1] = X_Centroid;
		results_array [2] = Y_Centroid;
		results_array [3] = w;
		results_array [4] = h;
	
		write_Array_To_File(labelsFile, results_array, array_length);
	}

}

close();
	
}


function write_Array_To_File(output_filename, results_array, array_length)
{
	output_string = "";
for (i=0; i<array_length; i++)
{
	output_string += d2s(results_array[i], 2);
	output_string += ",";
}

File.append(output_string, output_filename);
}
