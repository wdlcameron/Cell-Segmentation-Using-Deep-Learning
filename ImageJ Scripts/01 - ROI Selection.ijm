var recdir = 1;
var results_filename;
var SLICE = 2 //Choose the slice you want to default to for boxing
var suffix = "ome.tif"


macro "Fritter Wrap" {
	// Select a directory and recurse through it, working on .lsm images
	dir = getDirectory("Choose a Directory ");
	// Recurse through selected directory
	openFiles(dir, 0);
	// Done
	waitForUser("Finished analyzing images.");
}



//Recurse through the open directories
function openFiles(dir, recursion_count) 
{
	firstFileCheck = 0;
	list = getFileList(dir);
	for (i=0; i<list.length; i++) {
		path = dir+list[i];
		showProgress(i, list.length);
		showStatus(path);
		if (endsWith(path,"/"))	{	// If path is a directory, recurse into the directory
			recdir = 1;
			openFiles(path, (recursion_count+1));

		}
		// If path is a processed image, open and work on the file
		file2 = "";
		if (endsWith(path,suffix))
		{
			open(path);
			if(firstFileCheck == 0)
			{
				File.append("New Window\n" + getTitle() + "\n\n",results_filename);
				firstFileCheck = 1;  //Don't add again for the rest of the list
			}

			if (nImages>=1) {
				if (recdir == 1) {
					recdir = 0;	
				}
				selectWindow(list[i]);				
				selectROIs(path);
				}
		}
	}
}


function selectROIs(path)
{
	WindowName = getTitle();
	getDimensions(width, height, channels, slices, frames);
	run("Hyperstack to Stack");
	setSlice(SLICE);
	
	//Configure the ROI manager
	if (isOpen("ROI Manager")==0)
		run("ROI Manager...");
	roiManager("reset");
	roiManager("Show All");

	//open a previous ROI file if it exists
	pathROI = replace(path, ".ome.tif", "");
	roiFile = pathROI+"--ROI.zip";
	if (File.exists(roiFile))
		roiManager("open",roiFile);

	waitForUser("Select cells and add the selections to the ROI manager (ctrl+T)."
	+ "\n\nPress OK when done.");


	// save & process user input
	if ( roiManager("count")!=0 )
		roiManager("save",roiFile);
	selectWindow(WindowName);

	if (isOpen(WindowName))
		{
			selectWindow(WindowName);
			close();
		}

}
