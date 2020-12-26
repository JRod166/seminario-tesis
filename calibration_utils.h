#ifndef CAMERA_UTILS
#define CAMERA_UTILS

#include <iostream>
#include <iomanip>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/core/opengl.hpp"

using namespace cv;
using namespace std;

float distance(Point2f p1, Point2f p2) {
    return pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2);
}
bool intersection(Point2f o1, Point2f p1, Point2f o2, Point2f p2, Point2f &r) {
	Point2f x = o2 - o1;
	Point2f d1 = p1 - o1;
	Point2f d2 = p2 - o2;

	float cross = d1.x * d2.y - d1.y * d2.x;
	if (abs(cross) < 1e-8) {
		return false;
	}

	double t1 = (x.x * d2.y - x.y * d2.x) / cross;
	r = o1 + d1 * t1;
	return true;
}

float distance_to_rect(Point2f p1, Point2f p2, Point2f x) {
    float l2 = distance(p1, p2);
    if (l2 == 0.0) return sqrt(distance(p1, x));
    //float result = abs((p2.y - p1.y) * x.x - (p2.x - p1.x) * x.y + p2.x * p1.y - p2.y * p1.x) / sqrt(pow(p2.y - p1.y, 2) + pow(p2.x - p1.x, 2));
    //return result;
    float t = ((x.x - p1.x) * (p2.x - p1.x) + (x.y - p1.y) * (p2.y - p1.y)) / l2;
    t = max(0.0f, min(1.0f, t));
    float result = distance(x, Point2f(p1.x + t * (p2.x - p1.x),
                                       p1.y + t * (p2.y - p1.y)));
    return sqrt(result);
}
void getEulerAngles(Mat &rotCamerMatrix, Vec3d &eulerAngles) {

    Mat cameraMatrix, rotMatrix, transVect, rotMatrixX, rotMatrixY, rotMatrixZ;
    double* _r = rotCamerMatrix.ptr<double>();
    double projMatrix[12] = {_r[0], _r[1], _r[2], 0,
                             _r[3], _r[4], _r[5], 0,
                             _r[6], _r[7], _r[8], 0
                            };

    decomposeProjectionMatrix( Mat(3, 4, CV_64FC1, projMatrix),
                               cameraMatrix,
                               rotMatrix,
                               transVect,
                               rotMatrixX,
                               rotMatrixY,
                               rotMatrixZ,
                               eulerAngles);
    cameraMatrix.release();
    rotMatrix.release();
    transVect.release();
    rotMatrixX.release();
    rotMatrixY.release();
    rotMatrixZ.release();
}
void getEulerAngles(vector<Point3f>objectPoints, vector<Point2f> points, Mat camera_matrix, Mat distortion_coeffs, Vec3d &eulerAngles) {
    Mat rvec(3, 1, DataType<double>::type);
    Mat tvec(3, 1, DataType<double>::type);
    Mat rotation;

    solvePnP(Mat(objectPoints), Mat(points), camera_matrix, distortion_coeffs, rvec, tvec);
    Rodrigues(rvec, rotation);
    getEulerAngles(rotation, eulerAngles);
}
#endif