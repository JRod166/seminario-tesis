#include <iostream>
#include <iomanip>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "camera_calibration_ring.h"

using namespace cv;
using namespace std;
vector<string> videos = {
    "hp.mkv",
    "genius.mkv",
    "s3eyecam_5_4.avi",
    "cerca.mkv",
    "medio.mkv",
    "lejos.mkv",
    "logitech.mkv"
};
string actualVid;
/**
 * @brief Initialize windows names, sizes and positions.
 */
void window_setup() {
    int window_w = 180 * 1.5;
    int window_h = 120 * 1.5;
    int second_screen_offset = 0;//1360;
    string window_name;
    window_name = "CentersDistribution";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, 0 + second_screen_offset, 0);

    window_name = "CalibrationFrames";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w + second_screen_offset, 0);

    window_name = "Undistort";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 2 + second_screen_offset, 0);

    window_name = "FrontoParallel";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, 0 + second_screen_offset, window_h + 60);

    window_name = "Reproject";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w + second_screen_offset, window_h + 60);

    window_name = "Distort";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 2 + second_screen_offset, window_h + 60);

    window_name = "Threshold";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 3 + second_screen_offset, 0);

    window_name = "Line Representation";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 3 + second_screen_offset, window_h + 60);

    window_name = "Ellipses";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 4 + second_screen_offset, 0);

    window_name = "Contours";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 4 + second_screen_offset, window_h + 60);

    window_name = "Colinearity";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 4 + second_screen_offset, window_h*2 + 60);

    window_name = "Angles Cross Relation";
    namedWindow(window_name, WINDOW_NORMAL);
    resizeWindow(window_name, window_w, window_h);
    moveWindow(window_name, window_w * 3 + second_screen_offset, window_h*2 + 60);
}

int main( int argc, char** argv ) {
    window_setup();

    int pattern_rows = 4;
    int pattern_cols = 5;
    float square_size = 44.3;
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/hp.mkv");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/genius.mkv");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/s3eyecam_5_4.avi");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/cerca.mkv");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/medio.mkv");
    //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/lejos.mkv");
    for(int i = 0; i < videos.size(); i++)
    {
        actualVid = videos[i];
        string vidFile = "/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/"+videos[i];
        VideoCapture cap(vidFile);
        if ( !cap.isOpened() ) {
            cout << "Cannot open the video file. \n";
            return -1;
        }
        
        /*
        int pattern_rows = 3;
        int pattern_cols = 4;
        //VideoCapture cap("/home/pokelover/Documentos/automatic_camera_calibration/ring_calibration_videos/lifecam_4_3out.mp4");
        float square_size = 55;
        if ( !cap.isOpened() ) {
            cout << "Cannot open the video file. \n";
            return -1;
        }
        */

        int n_frames  = 80;
        int grid_cols = 8;
        int grid_rows = 8;
        CameraCalibrationRing camera_calibration(cap, pattern_cols, pattern_rows, square_size);
        // read image pathts from frame_list.txt
        //CameraCalibrationRing camera_calibration("ring_calibration_frames_full/frame_list.txt", pattern_cols, pattern_rows, square_size);
        camera_calibration.calibrate_camera_iterative(10, n_frames, grid_rows, grid_cols,actualVid);

    }
        waitKey(0);
    return 0;
}