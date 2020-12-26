#include <iostream>
#include <iomanip>
#include <string>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d.hpp"
#include <vector>
using namespace cv;
using namespace std;

class projective_invariants{
  public:
    vector<Point2f> pattern_points;
    vector< vector <int>> colinears;
    vector< vector <int>> angles;
    vector<vector <Point2f>> rects;
    bool colinear, angular_related;
    float colinearity_error = 0.6;
    float angle_cr_error = 0.08;
    float accumulated_colinear_error = 0.0;
    float accumulated_angular_error = 0.0;


    float normal_on_point (Point2f point)
    {
      return sqrt(pow(point.x,2)+pow(point.y,2));
    }
    float angle_between_three_points(Point2f intersection, Point2f A, Point2f B)
    {
      Point2f v1 = A-intersection;
      Point2f v2 = B-intersection;

      float numerator = v1.x*v2.x + v1.y * v2.y;
      float divisor = normal_on_point(v1) * normal_on_point(v2);

      return acos(numerator/divisor);
    }
    
    projective_invariants(int rows, int cols, vector<Point2f> pattern_points)
    {
      this->pattern_points = pattern_points;
      if(cols == 4 && rows == 3){
       //TODO
      }
      else {
        colinears = {{0,1,2,3,4},{5,6,7,8,9},{10,11,12,13,14},{15,16,17,18,19}, //horizontal
        {0,5,10,15},{1,6,11,16},{2,7,12,17},{3,8,13,18},{4,9,14,19}, //vertical
        {5,11,17},{0,6,12,18},{1,7,13,19},{2,8,14}, //first diagonal
        {9,13,17},{4,8,12,16},{3,7,11,15},{2,6,10} //second diagonal
        };
        //concurrency, A, B, C ,D
        angles = {{6,11,10,5,0},
                  {7,11,6,1,2},
                  {8,3,4,9,14},
                  {11,17,16,15,10},
                  {12,13,18,17,16},
                  {13,9,14,19,18}};
      }
      colinear = check_colinearity();
      angular_related = check_angle_cr();
      //colinear = true;
      //angular_related = true;
    }

    bool check_colinearity()
    {
      vector<Point2f> line;
      int counter = 0;
      Vec4f lineEq;
      for (int i = 0; i < colinears.size(); i++)
      {
        line.clear();
        for (int j = 0; j < colinears[i].size(); j++)
        {
          line.push_back(pattern_points[colinears[i][j]]);
        }
        
        fitLine(line,lineEq,DIST_L2,0,0.5,0.5);
        Point2f v1 = Point2f(lineEq[0],lineEq[1]);
        Point2f p1 = Point2f(lineEq[2],lineEq[3]);
        
        rects.push_back({v1,p1});
        
        for (int k = 0; k < line.size(); k++)
        {
          double distance = distance_to_rect(p1+1000*v1,p1-1000*v1,line[k]);
          if(distance > colinearity_error)
          {
            //cout<<distance<<endl;
            return false;
          }
          else {
            accumulated_colinear_error+=distance;
            counter++;
          }
        }
      }
      accumulated_colinear_error = accumulated_colinear_error/counter;
      return true;
    }

    bool check_angle_cr()
    {
      int counter = 0;
      Point2f A = Point2f(1,0);
      Point2f B = Point2f(1,1);
      Point2f C = Point2f(0,1);
      Point2f D = Point2f(1,-1);
      Point2f intersection = Point2f (0,0);
      float aux_AC = angle_between_three_points(intersection, A,C);
      float aux_CD = angle_between_three_points(intersection, C,D);
      float aux_BD = angle_between_three_points(intersection, B,D);
      float aux_AD = angle_between_three_points(intersection, A,D);
      float correct_cross = aux_AC/aux_CD * aux_BD/aux_AD;
      for(int i = 0 ; i < angles.size() ; i++)
      {
        aux_AC = angle_between_three_points(pattern_points[angles[i][0]], pattern_points[angles[i][1]],pattern_points[angles[i][3]]);
        aux_CD = angle_between_three_points(pattern_points[angles[i][0]], pattern_points[angles[i][3]],pattern_points[angles[i][4]]);
        aux_BD = angle_between_three_points(pattern_points[angles[i][0]], pattern_points[angles[i][2]],pattern_points[angles[i][4]]);
        aux_AD = angle_between_three_points(pattern_points[angles[i][0]], pattern_points[angles[i][1]],pattern_points[angles[i][4]]);
        float aux_cross = aux_AC/aux_CD * aux_BD/aux_AD;
        if(aux_cross - correct_cross > angle_cr_error)
        {
          return false;
        } else {
          accumulated_angular_error += angle_cr_error;
          counter++;
        }
      }
      accumulated_angular_error = accumulated_angular_error/counter;
      return true;
    }
};