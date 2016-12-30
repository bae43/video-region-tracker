#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	VideoCapture cap("C:\\Users\\bae43\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\video1.mp4");
	if (!cap.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	double count = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
	cap.set(CV_CAP_PROP_POS_FRAMES, count - 1); //Set index to last frame
	namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);

	while (1)
	{
		Mat frame;
		bool success = cap.read(frame);
		if (!success) {
			cout << "Cannot read  frame " << endl;
			break;
		}
		imshow("MyVideo", frame);
		if (waitKey(0) == 27) break;
	}
}