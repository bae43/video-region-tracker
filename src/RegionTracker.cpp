#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"

#include <iostream>
#include <list>
#include <queue>

using namespace cv;
using namespace std;

void drawBox(Mat& img, const Point2f& p1, const Point2f& p2, const Point2f& p3, const Point2f& p4) {
	Scalar color(255, 255, 0); // BGR
	int thickness = 2;

	line(img, p1, p2, color, thickness);
	line(img, p2, p3, color, thickness);
	line(img, p3, p4, color, thickness);
	line(img, p4, p1, color, thickness);
}

void renderTrack(Mat3b& img, const list<Point2f>& track, const Scalar& color) {

	Point2f prev;
	Point2f cur = track.front();

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

// fills in best -- a list of indices of closest points in sorted order by distance
void getBestAnchors(const Point2f& center, const vector<Point2f>& possible, vector<int>& best, int maxCount) {

	if (possible.size() < maxCount) {
		maxCount = possible.size();
	}

	// use 3 values to pack in index
	vector<Point3f> recentered(possible.size());
	for (int i = 0; i < possible.size(); i++) {
		recentered[i] = Point3f(possible[i].x - center.x, possible[i].y - center.y, i);
	}

	auto cmp = [](Point3f p1, Point3f p2) { return (p1.x * p1.x + p1.y * p1.y) < (p2.x * p2.x + p2.y * p2.y); };
	priority_queue<int, std::vector<Point3f>, decltype(cmp)> q(cmp);

	for (int i = 0; i < recentered.size(); i++) {
		q.push(recentered[i]);
		if (q.size() > maxCount) {
			// remove biggest, i.e. farthest away
			q.pop();
		}
	}

	best.clear();
	int count = q.size();
	for (int i = 0; i < count; i++) {
		// add the center offset back for image space coords
		best.push_back(q.top().z);
		q.pop();
	}
}

int main(int argc, char** argv) {

	const string input = "C:\\Users\\bae43\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\video1.mp4";
	const float x = 17;
	const float y = 57;
	const int w = 558;
	const int h = 303;

	// CW from top left
	Point2f p1(x, y);
	Point2f p2(x + w, y);
	Point2f p3(x + w, y + h);
	Point2f p4(x, y + h);
	vector<Point2f> points = { p1, p2, p3, p4 };

	VideoCapture cap(input);
	if (!cap.isOpened()) {
		cout << "Cannot open the video file" << endl;
		return -1;
	}

	long lastFrame = cap.get(CV_CAP_PROP_FRAME_COUNT); //get the frame count
	
	Mat3b frameBgr;
	Mat1b frameGray;
	Mat1b prevFrameGray;
	int maxTracks = 32;
	vector<Point2f> prevPoints(maxTracks);
	vector<Point2f> curPoints(maxTracks);
	vector<uchar> status(maxTracks);
	Mat err;

	const int trackLen = 40;
	vector<list<Point2f>> tracks(maxTracks);
	vector<Scalar> trackColors(maxTracks);
	list<Point2f> centerTrack;
	centerTrack.push_back(Point2f(x + w/2, y + h / 2));

	const int anchorCount = 5;
	// a Point2f with an additional int index at the end for where it is in curPoints
	vector<int> anchors(anchorCount);

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
	goodFeaturesToTrack(prevFrameGray, prevPoints, 50, 0.15, 20);
	for (int i = 0; i < prevPoints.size(); i++) {
		tracks[i].push_back(Point2f(prevPoints[i].x, prevPoints[i].y));
	}

	char buf[1024];
	snprintf(buf, 1024, "C:\\Users\\bae43\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\Debug\\frame-%04d.png", 0);
	imwrite(buf, frameBgr);

	for (long curFrame = 1; curFrame < lastFrame; curFrame++) {
	
		bool success = cap.read(frameBgr);
		if (!success) {
			cout << "Cannot read frame " << endl;
			break;
		}

		cvtColor(frameBgr, frameGray, CV_BGR2GRAY);
		
		//renderPoints(frameBgr, prevPoints, Scalar(255, 255, 255));
		curPoints.clear();
        calcOpticalFlowPyrLK(prevFrameGray, frameGray, prevPoints, curPoints, status, err);

		vector<Point2f> prevGoodPoints;
		vector<Point2f> curGoodPoints;

		// filter to only ones that are tracked
		for (int i = 0; i < curPoints.size(); i++) {
			if (status[i] > 0) {
				tracks[i].push_back(Point2f(curPoints[i].x, curPoints[i].y));
				prevGoodPoints.push_back(prevPoints[i]);
				curGoodPoints.push_back(curPoints[i]);
			}
			// fade track if lost or too long
			if (status[i] == 0 && tracks[i].size() > 1 || tracks[i].size() > trackLen) {
				tracks[i].pop_front();
			}
		
		}

		Mat h = findHomography(prevGoodPoints, curGoodPoints);
		vector<Point2f> newPoints(4);
		perspectiveTransform(points, newPoints, h);

		points = newPoints;
		
		centerTrack.push_back(Point2f((points[0] + points[1] + points[2] + points[3]) / 4));
		renderTrack(frameBgr, centerTrack, Scalar(255, 255, 255));

		drawBox(frameBgr, points[0], points[1], points[2], points[3]);
		for (int i = 0; i < tracks.size(); i++) {
			if (tracks[i].size() > 1) {
				renderTrack(frameBgr, tracks[i], trackColors[i]);
			}
		}

		char buf[1024];
		snprintf(buf, 1024, "C:\\Users\\bae43\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\Debug\\frame-%04d.png", curFrame);
		imwrite(buf, frameBgr);

		if (waitKey(0) == 27) break;
		
		prevFrameGray = frameGray.clone();
		prevPoints = curPoints;

		// reset if too many tracks lost
		if (curGoodPoints.size() < anchorCount) {
			goodFeaturesToTrack(prevFrameGray, prevPoints, 50, 0.15, 20);
			for (int i = 0; i < prevPoints.size(); i++) {
				tracks[i].clear();
				tracks[i].push_back(Point2f(prevPoints[i].x, prevPoints[i].y));
			}
		}
		
	}
}