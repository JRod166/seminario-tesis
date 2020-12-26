#include <iostream>
#include <iomanip>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "camera_calibration.h"
#include "ring/image_processing.h"
#include "ring/pattern_search.h"

class CameraCalibrationRing: public CameraCalibration {
public:
	CameraCalibrationRing(VideoCapture &cap, int pattern_cols, int pattern_rows, float square_size): CameraCalibration(cap,pattern_cols,pattern_rows,square_size) {
		load_object_points();
	}
	CameraCalibrationRing(string file_path, int pattern_cols, int pattern_rows, float square_size): CameraCalibration(file_path,pattern_cols,pattern_rows,square_size) {
		load_object_points();
	}

	Point2f calculate_pattern_center(vector<Point2f> pattern_points) {
		int p1,p2;
		if(pattern_cols == 4 && pattern_rows == 3){
			p1 = 5;
			p2 = 6;
		}
		else {
			p1 = 7;
			p2 = 12;
		}
		return Point2f( (pattern_points[p1].x + pattern_points[p2].x) / 2.0,
		                (pattern_points[p1].y + pattern_points[p2].y) / 2.0);
	}

	bool find_points_in_frame(Mat frame, vector<Point2f> &points) {
		Mat frame_gray, thresh, frame_mask, frame_original;

		int n_points = find_pattern_points(frame, w, h, points, pattern_cols, pattern_rows);
		bool result = n_points == board_points;
		frame_gray.release();
		thresh.release();
		return result;
	}


	void load_object_points() {
		Size board_size(pattern_cols, pattern_rows);
		float square_size_2 = w / pattern_cols;
		float desp_w = h * 0.2;
		float desp_h = w * 0.2;
		object_points.clear();
		object_points_image.clear();
		for ( int i = 0; i < board_size.height; i++ ) {
			for ( int j = 0; j < board_size.width; j++ ) {
				object_points.push_back(Point3f(  float(j * square_size), float(i * square_size), 0));
				//cout<<"object: "<<float(j * square_size)<<", "<<float(i * square_size)<<endl;
				object_points_image.push_back(Point3f(  float(j * square_size_2 + desp_w) , float( w - (i * square_size_2 + desp_h)), 0));
				//cout<<"image: "<<float(j * square_size_2+desp_w)<<", "<<float(w- (i * square_size_2+desp_h))<<endl;
			}
		}
	}
};