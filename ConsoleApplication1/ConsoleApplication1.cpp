#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include<list>

using namespace cv;
using namespace std;

void drawBox(Mat& img, int x, int y, int h, int w) {
	Scalar color(255, 255, 0); // BGR
	int thickness = 2;

	line(img, Point2f(x, y), Point2f(x + w, y), color, thickness);
	line(img, Point2f(x + w, y), Point2f(x + w, y + h), color, thickness);
	line(img, Point2f(x + w, y + h), Point2f(x, y + h), color, thickness);
	line(img, Point2f(x, y + h), Point2f(x, y), color, thickness);

}

void renderTrack(Mat3b& img, const list<Point2f>& track, const Scalar& color) {

	Point2f prev;
	Point2f cur = track.front();

	// circle(img, cur, 3, color, 3);
	for (auto iterator = track.begin(); iterator != track.end(); iterator++) {
		prev = cur;
		cur = *iterator;
		line(img, prev, cur, color, 2);
	}
}

void renderPoints(Mat3b& img, const vector<Point2f>& points, const Scalar& color) {
	for (int i = 0; i < points.size(); i++) {
		circle(img, points[i], 3, color, 3);
	}
}

int main(int argc, char** argv) {

	VideoCapture cap("C:\\Users\\bae43\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\video1.mp4");
	if (!cap.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	long lastFrame = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
	//cap.set(CV_CAP_PROP_POS_FRAMES, count - 1); //Set index to last frame
	//namedWindow("MyVideo", CV_WINDOW_AUTOSIZE);
	

	Mat3b frameBgr;
	Mat1b frameGray;
	Mat1b prevFrameGray;
	int maxTracks = 50;
	vector<Point2f> prevPoints(maxTracks);
	vector<Point2f> curPoints(maxTracks);
	vector<uchar> status(maxTracks);
	Mat err;

	int trackLen = 40;
	vector<list<Point2f>> tracks(maxTracks);
	vector<Scalar> trackColors(maxTracks);

	RNG rng;
	for (int i = 0; i < maxTracks; i++) {
		int icolor = (unsigned)rng;
		Scalar color(icolor & 255, (icolor >> 8) & 255, (icolor >> 16) & 255);
		trackColors[i] = color;
	}

	bool success = cap.read(frameBgr);
	if (!success) {
		cout << "Cannot read frame " << endl;
		return -1;
	}

	// initialize first frame features
	cvtColor(frameBgr, prevFrameGray, CV_BGR2GRAY);
	goodFeaturesToTrack(prevFrameGray, prevPoints, 50, 0.3, 20);
	for (int i = 0; i < prevPoints.size(); i++) {
		tracks[i].push_back(Point2f(prevPoints[i].x, prevPoints[i].y));
	}

	/// XXX
	//lastFrame = 20;
	for (long curFrame = 0; curFrame < lastFrame; curFrame++) {
	
		bool success = cap.read(frameBgr);
		if (!success) {
			cout << "Cannot read frame " << endl;
			break;
		}

		cvtColor(frameBgr, frameGray, CV_BGR2GRAY);
		
		//renderPoints(frameBgr, prevPoints, Scalar(255, 255, 255));

        calcOpticalFlowPyrLK(prevFrameGray, frameGray, prevPoints, curPoints, status, err);

		for (int i = 0; i < curPoints.size(); i++) {
			if (status[i] > 0) {
				tracks[i].push_back(Point2f(curPoints[i].x, curPoints[i].y));
			}
			// fade track if lost or too long
			if (status[i] == 0 && tracks[i].size() > 1 || tracks[i].size() > trackLen) {
				tracks[i].pop_front();
			}
		
		}

		for (int i = 0; i < tracks.size(); i++) {
			if (tracks[i].size() > 1) {
				renderTrack(frameBgr, tracks[i], trackColors[i]);
			}
		}

		char buf[1024];
		snprintf(buf, 1024, "C:\\Users\\bae43\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\Debug\\frame-%04d.png", curFrame);
		imwrite(buf, frameBgr);

		//drawBox(frameBgr, 100, 100, 50, 50);
		//imshow("MyVideo", frame);
		if (waitKey(0) == 27) break;
		
		prevFrameGray = frameGray.clone();
		prevPoints = curPoints;
	}
}