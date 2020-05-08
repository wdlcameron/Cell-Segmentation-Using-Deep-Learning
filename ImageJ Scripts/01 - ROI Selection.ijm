var recdir = 1;
var results_filename;

var manualMode = 1 // select 1 for manual selection (Romario)
//var SLICE = 12
var SLICE = 2

macro "Fritter Wrap" {
	// Select a directory and recurse through it, working on .lsm images
	if (manualMode == 0)
	{
		setBatchMode(true);		// Batch mode: don't show any image windows
	}
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
			//waitForUser ("Recursing into the directory", "Recursion Count: " + recursion_count + " firstColourChecker = " + first_colour_checker);
			openFiles(path, (recursion_count+1));

		}
		// If path is a processed image, open and work on the file
		file2 = "";
		if (endsWith(path,"ome.tif"))
		{
			open(path);
			if(firstFileCheck == 0)
			{
				File.append("New Window\n" + getTitle() + "\n\n",results_filename);
				firstFileCheck = 1;  //Don't add again for the rest of the list
			}

			
						// process image
			if (nImages>=1) {
				if (recdir == 1) {
					//File.append("\n\n" + dir,results_filename);
					recdir = 0;	
				}
				selectWindow(list[i]);
				// prepare image
				//setSlice(2);

				//run("Hi Lo Indicator");
				
				selectROIs(path);
				}
			
		}

	}

}



function selectROIs(path)
{

	//upperThreshold = 55000; // was 40000
	//lowerThreshold = 500;

	//lowerThreshold = 400; // was 3000

	//run("Add Slice");
		WindowName = getTitle();
		getDimensions(width, height, channels, slices, frames);
		
		
		//Optional
		run("Hyperstack to Stack");

		
		

			//run("Add Slice");
			setSlice(SLICE);

			// prepare ROI manager
			if (isOpen("ROI Manager")==0)
				run("ROI Manager...");

			roiManager("reset");

			roiManager("Show All");
			pathROI = replace(path, ".ome.tif", "");
			roiFile = pathROI+"--ROI.zip";
			//waitForUser("", roiFile);
			if (File.exists(roiFile))
				roiManager("open",roiFile);
			// wait for user

			if (manualMode == 1)
			{
				waitForUser("Select cells and add the selections to the ROI manager (ctrl+T)."
				+ "\n\nPress OK when done.");
			}
			

			
			
			// save & process user input
			if ( roiManager("count")!=0 )
				roiManager("save",roiFile);
			selectWindow(WindowName);
			//print(otsu);


/*			File.append("Anisotropy\tPerpendicular\tParallel",results_filename);
			setSlice(5);
			run("Median...", "radius=2 slice");
			setSlice(6);
			run("Median...", "radius=2 slice");
*/

		/*	
			if ( roiManager("count")!=0 )
				{
				roiManager("save",roiFile);
				for(j=0; j<roiManager("count"); j++)
					{
						
						roiManager("Select", j);
						setSlice(5);
					
						List.setMeasurements;
						Anis = List.getValue ("Mean");
						

						roiManager("Select", j);
						setSlice(2);
					
						List.setMeasurements;
						Weight = List.getValue ("Mean");

						roiManager("Select", j);
						setSlice(3);
					
						List.setMeasurements;
						Perp = List.getValue ("Mean");
						
					//waitForUser("Perp", Perp);
						roiManager("Select", j);
						setSlice(4);
					
						List.setMeasurements;
						Para = List.getValue ("Mean");
						AreaROI = List.getValue ("Area");
						X_Centroid = List.getValue("X");
						Y_Centroid = List.getValue("Y");

						G_Factor = 1;
						r = (Para-G_Factor*Perp)/(Para+2*G_Factor*Perp);
						
						
						roiManager("Select", j);
						setSlice(6);
					
						List.setMeasurements;
						preWeight = List.getValue ("Mean");
						weightedAnisotropy = preWeight/(Weight);
						
					//waitForUser("Perp", Para);
					if ( upperThreshold > Para && Para > lowerThreshold)
					File.append(Anis + "\t" + Perp + "\t" + Para + "\t" + AreaROI + "\t" + X_Centroid + "\t" + Y_Centroid + "\t" + preWeight + "\t" + weightedAnisotropy + "\t" + r,results_filename);
					//else
					//File.append(" " + "\t" + "\t" +  Para + "\t" + "\t" + " " + "\t" + "\t" + "\t" + Anis + "\t" + r,results_filename);

					
					}

				
				}


				*/


			if (isOpen(WindowName))
				{
					selectWindow(WindowName);
					close();
				}



}
