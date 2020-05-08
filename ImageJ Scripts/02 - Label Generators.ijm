var grid_size = newArray(4,4);
var results_filename;
var bounding_boxes = 5;
var num_classes = 1;
var currentlyIdentifying = 1;  //Change if you are identifying something else (1 for cell, 2 for mitochondria);

macro "Batch Measure" {


	setBatchMode(false);

	dir = getDirectory("Choose a Directory ");
    list = getFileList(dir);

	truncationCount = 0;
	dirName = substring(dir, 0, lengthOf(dir)-1);
	while (!endsWith(dirName, "\\"))
	{
	truncationCount = truncationCount + 1;
	//upperDir = Array.slice(upperDir, 0, upperDir.length-1);
	dirName = substring(dirName, 0, lengthOf(dirName)-1);
	//waitForUser("Filename i uppers", upperDir);
	}
	dirName = substring(dir, lengthOf(dir)-truncationCount-1, lengthOf(dir)-1);
	//waitForUser("", dirName);
		//results_filename = "statistics.fritter";
	
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

//delete the file if it already exists
if (File.exists(labelsFile))
	{
		File.delete(labelsFile);
		//waitForUser("Prev file deleted");
	}
File.append("Class,X,Y,W,H", labelsFile);
	


if (File.exists(roiFile))
	roiManager("open",roiFile);
//waitForUser("", roiFile);

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

		/*
		X_Grid_Location = floor(X_Centroid/grid_box_Width);
		Y_Grid_Location = floor(Y_Centroid/grid_box_Height);
		
		box_width_percentage = w/grid_box_Width;
		box_height_percentage = h/grid_box_Height;


			//Need to change
		x_centroid_percentage = X_Centroid;
		y_centroid_percentage = Y_Centroid;
		
			//waitForUser(X_Grid_Location, box_width_percentage +" , "+ box_height_percentage); 


			//Add to the appropriate sectiono f the vector;
			//the offset refers to the start of the location in the array that you want to write to
		preoffset = (X_Grid_Location * grid_size[1]) + Y_Grid_Location;
		offset = preoffset + grid_counter_array[preoffset]*(1+4+num_classes);


		
		waitForUser(preoffset, offset);
		
		if (grid_counter_array[preoffset]<bounding_boxes)
		{
			
			//Step 1 - Identify the 
			labels_array[offset] = 1; //found something at this location
			labels_array[offset+1] = y_centroid_percentage;
			labels_array[offset+2] = x_centroid_percentage;
			labels_array[offset+3] = box_height_percentage;
			labels_array[offset+4] = box_width_percentage;
			labels_array[offset+4+currentlyIdentifying] = 1; //Indicate the class
	
	
			//Increment the counter
			grid_counter_array[preoffset] = grid_counter_array[preoffset]+1;
		}
	write_Array_To_File(results_filename, labels_array, (1+4+num_classes));
	}

	

	Array.print(labels_array);
	Array.print(grid_counter_array);

	*/

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


//waitForUser ("Results are", output_string);
File.append(output_string, output_filename);
}
