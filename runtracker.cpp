//#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "kcftracker.hpp"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]){

	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = false;
	bool LAB = false;

	// Create KCFTracker object
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	// Frame readed
	Mat frame;

	// Tracker results
	Rect result;
	
	// Using min and max of X and Y for groundtruth rectangle
	float xMin = 306;
	float yMin = 5;
	float width = 95;
	float height = 65;

	string path = "D:\\study\\KCF\\code\\tracker_release2\\data\\Benchmark\\Deer\\img\\";
	char imgName[10];
	string frameName;
	
	for (int nFrames = 1; nFrames < 71; nFrames++)
	{
		sprintf(imgName, "%04d.jpg", nFrames);
		frameName = path + imgName;

		// Read each frame from the list
		frame = imread(frameName, CV_LOAD_IMAGE_COLOR);

		// First frame, give the groundtruth to the tracker
		if (nFrames == 1) {
			tracker.init( Rect(xMin, yMin, width, height), frame );
			rectangle( frame, Point( xMin, yMin ), Point( xMin+width, yMin+height), Scalar( 0, 255, 255 ), 1, 8 );			
		}
		// Update
		else{
			result = tracker.update(frame);
			rectangle( frame, Point( result.x, result.y ), Point( result.x+result.width, result.y+result.height), Scalar( 0, 255, 255 ), 1, 8 );			
		}

		if (!SILENT){
			imshow("Image", frame);
			waitKey(10);
		}
	}

}
