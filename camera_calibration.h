#include <iostream>
#include <bits/stdc++.h> 
#include <sys/stat.h> 
#include <sys/types.h> 
#include <iomanip>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include "calibration_utils.h"
#include "ring/projective_invariants.h"

#include <fstream>

#include <vector>
using namespace cv;
using namespace std;

#define REFINE_AVG       		0
#define REFINE_BLEND      		1
#define REFINE_VARICENTER 		2
#define REFINE_NONE 			3
#define REFINE_FP 				4
#define REFINE_FP_IDEAL 		5
#define REFINE_FP_INTERSECTION 	6

class CameraCalibration {
public:
	vector<vector<Point2f>> set_points;
	vector<Mat> frames;
	vector<Point3f> object_points;
	vector<Point3f> object_points_image;
	vector<string> frame_paths;
	VideoCapture cap;
	Mat camera_matrix;
	Mat dist_coeffs;
	Mat actual_frame;
	int w;
	int h;
	int iteration = 0;
	float square_size;
	int pattern_cols;
	int pattern_rows;
	int board_points;
	Size image_size;
	int current_frame;
	std::ofstream results;
	double prev_rms;
	float average_collinear_error = 0.0;
	float average_angular_error = 0.0;
	string actualDoc;

	CameraCalibration(VideoCapture &cap, int cols, int rows, float square_size) {
		this->pattern_cols = cols;
		this->pattern_rows = rows;
		this->board_points = cols * rows;
		this->square_size = square_size;

		this -> cap = cap;
		if ( !cap.isOpened() ) {
			cout << "Cannot open the video file. \n";
		}
		Mat frame;
		cap >> frame;
		w = frame.rows;
		h = frame.cols;
		image_size = frame.size();
		frame.release();
		//object_points.push_back(vector<Point3f>());
	}

	CameraCalibration(string frames_path, int cols, int rows, float square_size) {
		this->pattern_cols = cols;
		this->pattern_rows = rows;
		this->board_points = cols * rows;
		this->square_size = square_size;
		ifstream f(frames_path);
		cout << "Reading " << frames_path << endl;
		string path;
		while (f >> path) {
			frame_paths.push_back(path);
		}
		f.close();
		if (frame_paths.size() > 0) {
			Mat frame = imread(frame_paths[0]);
			w = frame.rows;
			h = frame.cols;
			image_size = frame.size();
			frame.release();
		}
	}
	virtual Point2f calculate_pattern_center(vector<Point2f> pattern_points) = 0;
	//virtual void calibrate_camera() = 0;
	virtual void load_object_points() = 0;
	virtual bool find_points_in_frame(Mat frame, vector<Point2f> &points) = 0;

	double calibrate_camera();

	void resetFrames();

	Mat getNextFrame();

	
	void collect_points_fronto_parallel(int refine_type, int refine_fronto_parallel_type);
	Point2f getIntersectionPoint(int x, int y, vector<Point2f>& new_points);
	void refine_points(vector<Point2f> &old_points, vector<Point2f> &new_points, int type);
	void refine_points_avg(vector<Point2f> &old_points, vector<Point2f>&new_points);
	void refine_points_blend(vector<Point2f> &old_points, vector<Point2f>&new_points);
	void refine_points_varicenter(vector<Point2f> &old_points, vector<Point2f>&new_points);
	void refine_points_intersection(vector<Point2f>& new_points);

	/**
	* @brief Skip f frames using a simple for,
	*
	* @param cap Videocapture reference
	* @param f number of frames to be skiped
	*/
	void skip_frames(int f) {
		current_frame += f;
		if ( cap.isOpened() ) {
			Mat frame;
			for (int i = 0; i < f; i++) {
				cap >> frame;
			}
			frame.release();
		}
	}

	/**
	* @brief Remove the distortion of the image, the result its only for visualization pourposes
	*
	* @param frame         Video frame
	* @param w             Width of the frame
	* @param h             Height of the frame
	* @param camera_matrix Camera matrix
	* @param dist_coeffs   Distortion coefficients
	*/
	void undistort_image(Mat & frame) {
		Mat rview, map1, map2;
		Size imageSize(h, w);
		initUndistortRectifyMap(camera_matrix,
		                        dist_coeffs,
		                        Mat(),
		                        getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, imageSize, 1, imageSize, 0),
		                        imageSize,
		                        CV_16SC2,
		                        map1,
		                        map2);
		remap(frame, rview, map1, map2, INTER_LINEAR);
		rview.copyTo(frame);
		rview.release();
		map1.release();
		map2.release();
	}

	bool check_grid(int** quadBins, int n_columns, int n_rows, int max_points){
		bool its_ok = true;
		double total_points = n_columns * n_rows * max_points;
		double min_percent = 0.8;
		int current_points = 0;

		for (int y_block = 0; y_block < n_rows-1; ++y_block) {
			for (int x_block = 0; x_block < n_columns-1; ++x_block) {
				current_points += quadBins[y_block][x_block];
				/*if(quadBins[y_block][x_block]<max_points){
					its_ok = false;
					// FINISH PROCESS
					y_block = n_rows;
					x_block = n_columns;
				}*/
			}
		}
		if(current_points < total_points * min_percent){
			its_ok = false;
		}
		return its_ok;
	}

	/**
	 * @brief Choose some frame to cover the screen area
	 *
	 * @param cap       Videocapture reference
	 * @param w         Width of the frame
	 * @param h         Height of the frame
	 * @param n_frames  Number of frames to be selected
	 * @param frames    Vector of frame positions selected
	 * @param n_rows    Number of rows to define quads areas
	 * @param n_columns Number of cols to define quads areas
	 * @return          True if was posible to found the required number of frames
	 */
	bool select_frames_process(const int n_frames, const int n_rows, const int n_columns, Mat &m_calibration, Mat &m_centroids) {
		resetFrames();

		Mat frame;
		Mat colinearity;
		Mat angular_cr;
		int width  = h;
		int height = w;
		int blockSize_y = height / n_rows;
		int blockSize_x = width  / n_columns;
		
		Vec3d eulerAngles;

		int on_success_skip = 9;
		int on_overflow_skip = 4;

		int num_color_palette = 100;
		vector<Scalar> color_palette(num_color_palette);
		RNG rng(12345);
		for (int i = 0; i < num_color_palette; i++)
			color_palette[i] = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));

		/*Size boardSize(5, 4);
		int squareSize = 45;
		int points = 20;
		vector<Point3f> objectPoints;
		for ( int i = 0; i < boardSize.height; i++ ) {
			for ( int j = 0; j < boardSize.width; j++ ) {
				objectPoints.push_back(Point3f(  float(j * squareSize),
				                                 float(i * squareSize), 0));
			}
		}*/
		vector<Point2f> temp(board_points);

		int** quadBins = new int*[n_rows];
		for (int y_block = 0; y_block < n_rows; ++y_block) {
			quadBins[y_block] = new int[n_columns];
			for (int x_block = 0; x_block < n_columns; ++x_block) {
				quadBins[y_block][x_block] = 0;
			}
		}

		int selected_frames = 0;
		//int max_points = round((n_frames + 4.0) / ((n_rows) * (n_columns)));
		int max_points = round((n_frames) / ((n_rows) * (n_columns)));
		if (max_points * n_rows * n_columns <= n_frames && (n_rows == 1 || n_columns == 1)) {
			max_points++;
		}
		if (max_points == 0)
		{
			cout<<"Cannot allocate "<<n_frames<<"frames on a "<<n_rows<<" x "<<n_columns<<" grid "<<endl;
		}
		while (selected_frames < n_frames && max_points * n_columns * n_rows > selected_frames) {
			bool rejected = false;
			bool accepted = false;
			vector<Point2f> pattern_points;
			frame = getNextFrame();
			if(frame.empty()){
				break;
			} 
			colinearity = frame.clone();
			angular_cr = frame.clone();
			if (find_points_in_frame(frame.clone(),pattern_points)) {

				Point2f pattern_center = calculate_pattern_center(pattern_points);
				int x_block = floor(pattern_center.x / blockSize_x);
				int y_block = floor(pattern_center.y / blockSize_y);
				int c_x = pattern_center.x ;
				int c_y = pattern_center.y ;
				bool near_center = true;
				int block_radio = (blockSize_x + blockSize_y) / (2 * 3.0);
				Point2f block_center((x_block + 0.5)*blockSize_x, (y_block + 0.5)*blockSize_y);

				projective_invariants invariants(pattern_rows,pattern_cols,pattern_points);
				
				for( int rects = 0 ; rects< invariants.rects.size(); rects++)
				{
					line(colinearity,invariants.rects[rects][1]-(invariants.rects[rects][0]*1000),invariants.rects[rects][1]+(invariants.rects[rects][0]*1000),Scalar(255,0,0),1);
				}
				for ( int point_i = 0 ; point_i < invariants.angles.size() ; point_i++ )
				{
					line(angular_cr,pattern_points[invariants.angles[point_i][0]],pattern_points[invariants.angles[point_i][1]],Scalar(0,0,255),2);
					line(angular_cr,pattern_points[invariants.angles[point_i][0]],pattern_points[invariants.angles[point_i][2]],Scalar(255,0,255),2);
					line(angular_cr,pattern_points[invariants.angles[point_i][0]],pattern_points[invariants.angles[point_i][3]],Scalar(0,255,0),2);
					line(angular_cr,pattern_points[invariants.angles[point_i][0]],pattern_points[invariants.angles[point_i][4]],Scalar(255,0,0),2);
				} 
				if (norm(Mat(Point2f(c_x, c_y)),Mat(block_center)) > block_radio ) {
					near_center = false;
				}
				
				if(x_block == 0 || y_block == 0 || x_block == n_columns-1 || y_block == n_rows-1){
					near_center = true;
				}
				circle(m_centroids, block_center, block_radio, Scalar(255, 0, 0));
				circle(m_centroids, Point2f(c_x, c_y), 1, Scalar(0, 0, 255));
				if (near_center&& invariants.colinear && invariants.angular_related) {
					if ( quadBins[y_block][x_block] < max_points) {
						for (int i = 0; i < board_points; i++) {
							temp[i] = pattern_points[i];
						}
						if (!camera_matrix.empty() && !dist_coeffs.empty()) {
							//getEulerAngles(object_points, temp, camera_matrix, dist_coeffs, eulerAngles);
							//float yaw = eulerAngles[1];
							//float pitch = eulerAngles[0];
							//float roll = eulerAngles[2];
							//if (!(yaw > -20 && yaw < 20 && roll > -30 && roll < 30 && (pitch > 150 || pitch < -150))) {
							//	rejected = true;
							//}
						}
						if (!rejected) {
							average_angular_error += invariants.accumulated_angular_error;
							average_collinear_error += invariants.accumulated_colinear_error;
							circle(m_centroids, Point2f(c_x, c_y), 6, Scalar(0, 255, 0), -1);
							frames.push_back(frame);
							for ( int f = 0 ; f < pattern_points.size() ; f++)
							{
								circle(frame, pattern_points[f], 1, Scalar(0,255,255), -1);
							}
							string size_frames = to_string(frames.size());
							//cout<<size_frames<<endl;

							imwrite("images/"+actualDoc+"/"+to_string(iteration)+'-'+size_frames+".jpg", frame);
							selected_frames++;
							accepted = true;
							skip_frames(on_success_skip);
							//f += on_success_skip;

							quadBins[y_block][x_block] ++;
							for (int j = 0; j < board_points; j++) {
								circle(m_calibration, temp[j], 10, color_palette[selected_frames]);
							}

						}
					}
					else {
						skip_frames(on_overflow_skip);
						//f += on_overflow_skip;
					}
				}
				if(check_grid(quadBins, n_columns, n_rows, max_points)){
					cout << "Grid is ok" << endl;
					break;
				}

			}
			//draw rectangle
			int yOffsetText = 25;
			for (int y_block = 0; y_block < n_rows; ++y_block) {
				for (int x_block = 0; x_block < n_columns; ++x_block) {
					int local_n_y = height / n_rows;  //n of blocks
					int local_a_y = y_block * local_n_y;                 //lower bound
					int local_b_y = local_a_y + local_n_y;               //upper bound

					int local_n_x = width / n_columns;  //n of blocks
					int local_a_x = x_block * local_n_x;                 //lower bound
					int local_b_x = local_a_x + local_n_x;               //upper bound

					std::ostringstream quads_str;
					//quads_str << std::fixed << std::setprecision(2) << quadBins[y_block][x_block] / max_points * 100.0 << "%";
					quads_str << quadBins[y_block][x_block] << "/" << max_points ;
					rectangle(m_centroids, Point(local_a_x, local_a_y), Point(local_a_x + width * 0.1, local_a_y + yOffsetText), Scalar(0, 0, 0), -1);
					putText(m_centroids, quads_str.str(), Point(local_a_x, local_a_y + yOffsetText), FONT_HERSHEY_PLAIN, 2, Scalar(0, 255, 255), 2);
					rectangle(m_centroids, Point(local_a_x, local_a_y), Point(local_b_x, local_b_y), Scalar(255, 255, 255));
					//rectangle(frame, Point(local_a_x, local_a_y), Point(local_b_x, local_b_y), Scalar(255, 255, 255));
				}
			}
			rectangle(m_centroids, Point(width * 0.15, height - 35), Point(width * 0.85, height), Scalar(0, 0, 0), -1);
			//total_str << "total: " << std::fixed << std::setprecision(2) << 100 * sumPercentage << "%   " << "s.d.: " << standarDeviation(percentage_values);
			std::ostringstream total_str;
			total_str << "Selected: " << selected_frames << "/" << n_frames << " Total:" << frames.size();
			putText(m_centroids, total_str.str(), Point(width * 0.15, height - 10), FONT_HERSHEY_PLAIN, 2, Scalar(255, 255, 0), 2);
			imshow("CentersDistribution", m_centroids);
			imshow("CalibrationFrames", m_calibration);
			imshow("Undistort", frame);
			imshow("Colinearity", colinearity);
			imshow("Angles Cross Relation", angular_cr);
			//f++;
			//skip_frames(cap,1);
			waitKey(1);
			if (!accepted){
				frame.release();
			}
		}
		cout << "Finish selection with " << selected_frames << " of " << n_frames << endl;
		return selected_frames == n_frames;
	}

	/**
	* @brief Run the selection frames process
	*
	* @param cap       Videocapture reference
	* @param w         Width of the frame
	* @param h         Height of the frame
	* @param n_frames  Number of frames to be selected
	* @param frames    Vector of frame positions selected
	* @param n_rows    Number of rows to define quads areas
	* @param n_columns Number of cols to define quads areas
	* @return          True if was posible to found the required number of frames
	*/
	bool select_frames(const int n_frames, const int n_rows, const int n_columns) {
		int rows = n_rows;
		int cols = n_columns;
		int select_frames = n_frames;
		Mat m_calibration = Mat::zeros(Size(h, w), CV_8UC3);
		Mat m_centroids = Mat::zeros(Size(h, w), CV_8UC3);
		
		while (!select_frames_process(select_frames, rows, cols, m_calibration, m_centroids)) {
			cout << rows << " " << cols << endl;
			rows --;
			cols --;
			select_frames = n_frames - frames.size();
			if (rows < 1 || cols < 1) {
				break;
			}
			m_centroids = Mat::zeros(Size(h, w), CV_8UC3);
		}

	}

	/**
	* @brief Search pattern points of choosen frames and save the points in the set_points array
	*
	* @param cap           VideoCapture reference
	* @param w             Width of the frame
	* @param h             Height of the frame
	* @param set_points    Set of points to use in the camera calibration
	*/
	void collect_points() {
		set_points.clear();
		for (int f = 0; f < frames.size(); f++) {
			vector<Point2f> pattern_points;
			if (find_points_in_frame(frames[f].clone(), pattern_points)) {
				set_points.push_back(pattern_points);
			}
			imshow("FrontoParallel",frames[f]);
			waitKey(1);
		}
	}

	void calibrate_camera_iterative(int n_iterations, int n_frames, int grid_rows, int grid_cols, string name) {
		actualDoc = name;
		string folder = "images/"+actualDoc;
		mkdir(folder.c_str(),0777);
		results.open ("results"+name+".csv");
		results.flush();
		select_frames(n_frames, grid_rows, grid_cols);
		collect_points();
		average_collinear_error = average_collinear_error/ frames.size();
		average_angular_error = average_angular_error/frames.size();
		results<<"colinear,"<<average_collinear_error<<endl;
		results<<"angular,"<<average_angular_error<<endl;
		results<<"Metodo,Fx,Fy,Uo,V0,rms\n";
		results<<"OpenCV,";
		prev_rms = calibrate_camera();
		double this_rms = 0;
		double threshold = 0.00005;
		//cout<<"RMS: "<<prev_rms<<endl;
		//cout<<prev_rms<<" -->  "<<this_rms<<" ==> "<<abs(this_rms-prev_rms)<<endl;
		for (int i = 0; abs(this_rms-prev_rms) > threshold ; i++) {
			prev_rms = this_rms;
			iteration++;
			results<<"Iteration "<<i+1<<',';
			results.flush();
			//collect_points_fronto_parallel(REFINE_VARICENTER, REFINE_FP_INTERSECTION);
			//collect_points_fronto_parallel(REFINE_NONE, REFINE_FP_IDEAL);
			collect_points_fronto_parallel(REFINE_VARICENTER, REFINE_FP_IDEAL);
			//collect_points_fronto_parallel(REFINE_AVG,REFINE_FP_IDEAL);
			this_rms = calibrate_camera();
			//cout<<prev_rms<<" -->  "<<this_rms<<" ==> "<<this_rms-prev_rms<<endl;
		}
		cout << " ### Finishing iterations ### "<<endl;
		results.close();
	}

	/**
	* @brief Add the distortion to the points
	*
	* @param xy            Undistorted points
	* @param uv            Distorted points
	* @param camera_matrix Camera matrix
	* @param dist_coeffs   Distortion coefficients
	*/
	void distort_points(const vector<Point2f> &xy, vector<Point2f> &uv) {
		double fx = camera_matrix.at<double>(0, 0);
		double fy = camera_matrix.at<double>(1, 1);
		double cx = camera_matrix.at<double>(0, 2);
		double cy = camera_matrix.at<double>(1, 2);
		double k1 = dist_coeffs.at<double>(0, 0);
		double k2 = dist_coeffs.at<double>(0, 1);
		double p1 = dist_coeffs.at<double>(0, 2);
		double p2 = dist_coeffs.at<double>(0, 3);
		double k3 = dist_coeffs.at<double>(0, 4);

		double x;
		double y;
		double r2;
		double xDistort;
		double yDistort;
		for (int p = 0; p < xy.size(); p++) {
			x = (xy[p].x - cx) / fx;
			y = (xy[p].y - cy) / fy;
			r2 = x * x + y * y;

			// Radial distorsion
			xDistort = x * (1 + k1 * r2 + k2 * pow(r2, 2) + k3 * pow(r2, 3));
			yDistort = y * (1 + k1 * r2 + k2 * pow(r2, 2) + k3 * pow(r2, 3));

			// Tangential distorsion
			xDistort = xDistort + (2 * p1 * x * y + p2 * (r2 + 2 * x * x));
			yDistort = yDistort + (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y);

			// Back to absolute coordinates.
			xDistort = xDistort * fx + cx;
			yDistort = yDistort * fy + cy;
			uv[p] = Point2f(xDistort, yDistort);
		}
	}
};

/**
 * @brief Search pattern points in the undistorted image, find a homography
 * to get a cannonical view, find patter points in the cannonical view and
 * refine points and save the points in the set_points array
 * @details [long description]
 *
 * @param cap           VideoCapture reference
 * @param w             Width of the frame
 * @param h             Height of the frame
 * @param set_points    Set of points to use in the camera calibration
 * @param camera_matrix Camera matrix
 * @param dist_coeffs   Distortion coefficients
 * @param refine_type   Tipe of refinement in every iteration
 */
void CameraCalibration::collect_points_fronto_parallel(int refine_type, int refine_fronto_parallel_type) {
	//Size imageSize(h, w);
	//Size boardSize(8, 5);
	//int n_points = 42;
	Mat frame;
	Mat map1, map2;
	Mat input_undistorted;
	//vector<Point3f> points_real;
	vector<Point2f> temp(board_points);
	vector<Point2f> temp2(board_points);
	vector<vector<Point2f>> start_set_points;
	vector<vector<Point2f>> new_set_points;
	vector<Point2f> points_undistorted;
	vector<Point2f> points_fronto_parallel;

	set_points.clear();
  int	frame_number = 0;
	for (int f = 0; f < frames.size(); f++) {
		frame_number++;
		points_undistorted.clear();
		points_fronto_parallel.clear();
		frames[f].copyTo(frame);
		frame.copyTo(input_undistorted);
		undistort_image(input_undistorted);
		imshow("Undistort", input_undistorted);

		undistort(frame, input_undistorted, camera_matrix, dist_coeffs);
		
		//imshow("UndistortInput", input_undistorted);
		if (!find_points_in_frame(input_undistorted.clone(), points_undistorted)) {
			continue;
		}

		projective_invariants undistorted_invariant = projective_invariants(pattern_rows,pattern_cols,points_undistorted);

		Mat homography = cv::findHomography(points_undistorted, object_points_image);
		Mat inv_homography = cv::findHomography(object_points_image, points_undistorted);
		Mat img_in  = input_undistorted.clone();
		Mat img_out = input_undistorted.clone();
		
		cv::warpPerspective(img_in, img_out, homography, image_size);

		Mat colinearity = img_out.clone();
		Mat angular_cr = img_out.clone();
		

		for (int p = 0; p < board_points; p++){
			circle(input_undistorted, points_undistorted[p], 2, Scalar(0, 255, 0));
			imwrite("images/"+actualDoc+"/"+to_string(iteration)+'-'+to_string(p)+"-reproject.jpg", input_undistorted);
		}

		if (find_points_in_frame(img_out, points_fronto_parallel)) {
			for ( int f = 0 ; f < points_fronto_parallel.size() ; f++)
			{
				circle(img_out, points_fronto_parallel[f], 1, Scalar(0,255,255), -1);
			}
			string size_frames = to_string(frame_number);
			imwrite("images/"+actualDoc+"/"+to_string(iteration)+'-'+size_frames+"-fronto.jpg", img_out);
			projective_invariants fronto_parallel_invariant = projective_invariants(pattern_rows,pattern_cols,points_fronto_parallel);
			for( int rects = 0 ; rects< fronto_parallel_invariant.rects.size(); rects++)
			{
				line(colinearity,fronto_parallel_invariant.rects[rects][1]-(fronto_parallel_invariant.rects[rects][0]*1000),fronto_parallel_invariant.rects[rects][1]+(fronto_parallel_invariant.rects[rects][0]*1000),Scalar(255,0,0),1);
			}
			for ( int point_i = 0 ; point_i < fronto_parallel_invariant.angles.size() ; point_i++ )
			{
				line(angular_cr,points_fronto_parallel[fronto_parallel_invariant.angles[point_i][0]],points_fronto_parallel[fronto_parallel_invariant.angles[point_i][1]],Scalar(0,0,255),2);
				line(angular_cr,points_fronto_parallel[fronto_parallel_invariant.angles[point_i][0]],points_fronto_parallel[fronto_parallel_invariant.angles[point_i][2]],Scalar(255,0,255),2);
				line(angular_cr,points_fronto_parallel[fronto_parallel_invariant.angles[point_i][0]],points_fronto_parallel[fronto_parallel_invariant.angles[point_i][3]],Scalar(0,255,0),2);
				line(angular_cr,points_fronto_parallel[fronto_parallel_invariant.angles[point_i][0]],points_fronto_parallel[fronto_parallel_invariant.angles[point_i][4]],Scalar(255,0,0),2);
			} 
			if((fronto_parallel_invariant.colinear && undistorted_invariant.colinear) || set_points.size() <= 15)
			{	
				if ( (fronto_parallel_invariant.angular_related && undistorted_invariant.angular_related) || set_points.size() <= 15)
				{	
					for (int p = 0; p < board_points; p++) {
						circle(img_out, points_fronto_parallel[p], 0, Scalar(0, 255, 0));
					}

					vector<Point2f> object_p_canonical;
					if (refine_fronto_parallel_type == REFINE_FP_IDEAL) {
						for (int p = 0; p < board_points; p++) {
							object_p_canonical.push_back(Point2f((points_fronto_parallel[p].x + object_points_image[p].x) / 2.0 ,
																									(points_fronto_parallel[p].y + object_points_image[p].y) / 2.0 ));
						}
					} else if (refine_fronto_parallel_type == REFINE_FP_INTERSECTION) {
						for (int p = 0; p < board_points; p++) {
							object_p_canonical.push_back(points_fronto_parallel[p]);
						}
						refine_points_intersection(object_p_canonical);
					} else {
						for (int p = 0; p < board_points; p++) {
							object_p_canonical.push_back(points_fronto_parallel[p]);
						}
					}

					vector<Point2f> new_points2D(board_points);
					perspectiveTransform(object_p_canonical, new_points2D, inv_homography);
					//for (int p = 0; p < n_points; p++) {
					//	circle(input_undistorted, new_points2D[p], 2, Scalar(0, 0, 255));
					//	circle(frame, new_points2D[p], 2, Scalar(0, 255, 0));
					//}
					refine_points(points_undistorted, new_points2D, refine_type);

					vector<Point2f> new_points2D_distort(board_points);
					distort_points(new_points2D, new_points2D_distort);

					set_points.push_back(new_points2D_distort);
					//start_set_points.push_back(points_undistorted);
					//new_set_points.push_back(new_points2D);
					}
					else 
					{
						//cout << "Reprojection is not correctly correlated"<<endl;
					}
			} else {
				//cout<< "Reprojection is not collinear"<<endl;
			}
		} else {
			//cout << "Not found in FP" << endl;
		}
		imshow("Reproject", input_undistorted);
		imshow("FrontoParallel", img_out);
		imshow("Distort", frame);
		imshow("Colinearity", colinearity);
		imshow("Angles Cross Relation", angular_cr);
		
		img_in.release();
		img_out.release();
		homography.release();
		inv_homography.release();
		waitKey(1);
	}
	
	//cout << "\t" << avgColinearDistance(start_set_points);
	//cout << "\t" << avgColinearDistance(new_set_points) << endl;
}

double CameraCalibration::calibrate_camera() {
	vector<Mat> tvecs;
	vector<Mat> rvecs;
	vector<vector<Point3f>> calibration_obj_points(1);
	calibration_obj_points[0].resize(0);
	calibration_obj_points[0] = object_points;
	calibration_obj_points.resize(set_points.size(), calibration_obj_points[0]);
	//cout<<set_points.size()<<endl;
	double rms = calibrateCamera(calibration_obj_points,
	                             set_points,
	                             image_size,
	                             camera_matrix,
	                             dist_coeffs,
	                             rvecs,
	                             rvecs,CALIB_ZERO_TANGENT_DIST | 
																 CALIB_FIX_K1 /*| CALIB_FIX_K2 | CALIB_FIX_K3 |
																 CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 |
																 CALIB_FIX_S1_S2_S3_S4 | CALIB_FIX_TAUX_TAUY*/);
	/*cout << endl;
	cout << "Camera_matrix" << endl;
	cout << camera_matrix << endl;
	cout << "Dist_coeffs" << endl;
	cout << dist_coeffs   << endl;
	cout << "rms: "<<rms<<endl;*/
	const double* pos_i = camera_matrix.ptr<double>(0);
	const double* pos_j = camera_matrix.ptr<double>(1);
	results<<pos_i[0]<<","; //Fx
	results.flush();
	results<<pos_j[1]<<","; //Fy
	results.flush();
	results<<pos_i[2]<<","; //Uo
	results.flush();
	results<<pos_j[2]<<","; //V0
	results.flush();
	results<<rms<<"\n";
	results.flush();
	return rms;
}

void CameraCalibration::resetFrames(){
	current_frame = 0;
	if ( cap.isOpened() ) {
		current_frame = 1;
		cap.set(CAP_PROP_POS_FRAMES, current_frame);
	}
}

Mat CameraCalibration::getNextFrame(){
	Mat frame;
	if ( cap.isOpened() ) {
		cap >> frame;
		current_frame++;
	}
	else {
		if(current_frame<frame_paths.size()){
			frame = imread(frame_paths[current_frame++]);
		}
	}
	return frame;
}

void CameraCalibration::refine_points(vector<Point2f> &old_points, vector<Point2f> &new_points, int type) {
	switch (type) {
	case REFINE_AVG:
		//TODO
		refine_points_avg(old_points, new_points);
		break;
	case REFINE_BLEND:
		//TODO
		refine_points_blend(old_points, new_points);
		break;
	case REFINE_VARICENTER:
		//TODO
		refine_points_varicenter(old_points, new_points);
		break;
	}
}
/**
 * @brief Refine points using the average of old and new point (o+n)/2
 * 
 * @param old_points Initial points
 * @param new_points Actual points
 */
void CameraCalibration::refine_points_avg(vector<Point2f> &old_points, vector<Point2f> &new_points) {
	for (int p = 0; p < new_points.size(); p++) {
		new_points[p].x = old_points[p].x * 0.5 + new_points[p].x * 0.5;
		new_points[p].y = old_points[p].y * 0.5 + new_points[p].y * 0.5;
	}
}
/**
 * @brief Refine points using a blend factor inversely proportional to the distance to the fit line
 * 
 * @param old_points Initial points
 * @param new_points Actual points
 */
void CameraCalibration::refine_points_blend(vector<Point2f> &old_points, vector<Point2f>&new_points) {
	vector<Point2f> points(5);
	Vec4f line;
	float factor;
	float d_old;
	float d_new;
	for (int row = 0; row < 4; row++) {
		for (int c = 0; c < 5; c++) {
			points[c] = new_points[row * 5 + c];
		}
		fitLine(points, line, cv::DIST_L2,  0, 0.01, 0.01);
		Point2f p1;
		Point2f p2;
		p1.x = line[2];
		p1.y = line[3];

		p2.x = p1.x + new_points[row * 5 + 4].x * line[0];
		p2.y = p1.y + new_points[row * 5 + 4].x * line[1];

		for (int c = 0; c < 5; c++) {
			d_old = distance_to_rect(p1, p2, old_points[row * 5 + c]);
			d_new = distance_to_rect(p1, p2, new_points[row * 5 + c]);
			factor = d_old / (d_old + d_new);
			new_points[row * 5 + c].x = old_points[row * 5 + c].x * (1 - factor) + new_points[row * 5 + c].x * factor;
			new_points[row * 5 + c].y = old_points[row * 5 + c].y * (1 - factor) + new_points[row * 5 + c].y * factor;
		}
	}
}
/**
 * @brief Refine points using the (old + new + intersection point of fit lines (horizontal and vertical) )/3
 * 
 * @param old_points Initial points
 * @param new_points Actual points
 */
void CameraCalibration::refine_points_varicenter(vector<Point2f> &old_points, vector<Point2f>& new_points) {
	for (int y = 0; y < pattern_rows; ++y) {
		for (int x = 0; x < pattern_cols; ++x) {
			int offset = y * pattern_cols + x;
			Point2f insectionPoint = getIntersectionPoint(x, y, new_points );
			new_points[offset].x = (old_points[offset].x + new_points[offset].x  + insectionPoint.x) / 3;
			new_points[offset].y = (old_points[offset].y + new_points[offset].y  + insectionPoint.y) / 3;

		}
	}
}
/**
 * @brief Refine points using only the intersection point of fit lines (horizontal and vertical)
 * 
 * @param new_points Intersection point for each position
 */
void CameraCalibration::refine_points_intersection(vector<Point2f>& new_points) {
	for (int y = 0; y < 4; ++y) {
		for (int x = 0; x < 5; ++x) {
			int offset = y * 5 + x;
			Point2f insectionPoint = getIntersectionPoint(x, y, new_points );
			new_points[offset].x = insectionPoint.x;
			new_points[offset].y = insectionPoint.y;
		}
	}
}

Point2f CameraCalibration::getIntersectionPoint(int x, int y, vector<Point2f>& new_points) {
	int offset = pattern_cols * y + x;

	std::vector<Point2f> pointsINLINE1, pointsINLINE2;
	Vec4f _line;

	int x_a = offset - x;
	int x_b = x_a + pattern_cols;

	for (int i = x_a; i < x_b; ++i) {
		pointsINLINE1.push_back(new_points[i]);
	}

	int y_a = offset - y * pattern_cols;

	for (int i = 0; i < pattern_rows; ++i) {
		int offset = y_a + i * pattern_cols;
		pointsINLINE2.push_back(new_points[offset]);
	}


	fitLine(pointsINLINE1, _line, cv::DIST_L2,  0, 0.01, 0.01);
	Point2f p1 (_line[2], _line[3]), p2 ;

	p2.x = p1.x + 205 * _line[0];
	p2.y = p1.y + 205 * _line[1];

	fitLine(pointsINLINE2, _line, cv::DIST_L2,  0, 0.01, 0.01);
	Point2f p1_2 (_line[2], _line[3]), p2_2 ;

	p2_2.x = p1_2.x + 205 * _line[0];
	p2_2.y = p1_2.y + 205 * _line[1];

	Point2f intersec ;
	intersection(p1, p2, p1_2, p2_2, intersec);
	return intersec;
}