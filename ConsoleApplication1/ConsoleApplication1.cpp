#include "stdafx.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include "opencv2/imgproc.hpp"
#include <opencv2/video/tracking.hpp>

#include <iostream>
#include <list>
#include <queue>

using namespace cv;
using namespace std;

void drawBox(Mat& img, float x, float y, int h, int w) {
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
	float x = 17;
	float y = 57;
	const int w = 558;
	const int h = 303;

	VideoCapture cap(input);
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
	goodFeaturesToTrack(prevFrameGray, prevPoints, 50, 0.2, 20);
	for (int i = 0; i < prevPoints.size(); i++) {
		tracks[i].push_back(Point2f(prevPoints[i].x, prevPoints[i].y));
	}

	char buf[1024];
	snprintf(buf, 1024, "C:\\Users\\bae43\\Documents\\Visual Studio 2015\\Projects\\ConsoleApplication1\\Debug\\frame-%04d.png", 0);
	imwrite(buf, frameBgr);

	/// XXX
	//lastFrame = 20;
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

		// ensures anchors is also sorted by distance
		Point2f center(x + w / 2, y + h / 2);
		getBestAnchors(center, curGoodPoints, anchors, anchorCount);

		Point2f update;
		int offsetIdx;

		// take median
		if (anchors.size() >= 3) {
			auto cmp = [](Point2f p1, Point2f p2) { return (p1.x * p1.x + p1.y * p1.y) < (p2.x * p2.x + p2.y * p2.y); };
			priority_queue<int, std::vector<Point2f>, decltype(cmp)> q(cmp);
			for (int i = anchors.size() / 2 - 1; i <= anchors.size() / 2 + 1; i++) {
				offsetIdx = anchors[i];
				q.push(Point2f(curGoodPoints[offsetIdx] - prevGoodPoints[offsetIdx]));
			}
			q.pop();
			update = q.top();
		}
		else {
			offsetIdx = anchors[anchors.size() / 2];
			update = curGoodPoints[offsetIdx] - prevGoodPoints[offsetIdx];
		}
		
		x += update.x;
		y += update.y;
		for (int i = 0; i < anchors.size(); i++) {
			line(frameBgr, center, curGoodPoints[anchors[i]], Scalar(128, 128, 128));
		}
		centerTrack.push_back(Point2f(x + w / 2, y + h / 2));
		renderTrack(frameBgr, centerTrack, Scalar(255, 255, 255));

		// highlight best tracking point
		line(frameBgr, center, curGoodPoints[offsetIdx], Scalar(0, 255, 0));

		drawBox(frameBgr, x, y, h, w);
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

		// reset if too many tracks lost
		if (curGoodPoints.size() < anchorCount) {
			goodFeaturesToTrack(prevFrameGray, prevPoints, 50, 0.2, 20);
			for (int i = 0; i < prevPoints.size(); i++) {
				tracks[i].clear();
				tracks[i].push_back(Point2f(prevPoints[i].x, prevPoints[i].y));
			}
		}
		
	}
}