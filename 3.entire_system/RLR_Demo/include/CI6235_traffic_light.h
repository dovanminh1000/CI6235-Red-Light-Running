#ifndef TRAFFIC_LIGHT_H
#define TRAFFIC_LIGHT_H

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

struct traffic_light
{
    Scalar above_light;
    Scalar centre_light;
    Scalar below_light;
};

enum StateTrafficLight
{
    STATE_GREEN,
    STATE_YELLOW,
    STATE_RED

};

class TrafficLight
{
private:
    //CI6235: Traffic light color
    Scalar green_light = cv::Scalar(23, 187, 76);
    Scalar red_light = cv::Scalar(0, 0, 230);
    Scalar amber_light = cv::Scalar(0, 120, 255);
    Scalar black_light = cv::Scalar(0, 0, 0);

    traffic_light centre_traffic_light;

    //CI6235: Intialize parameters to check the time and traffic state
    int current_second = 0;
    int previous_second = 0;
    int state_time = 0;
    int current_state = 0;

    //CI6235: Determine traffic light position
    const int central_first_light_x = 350;
    const int central_first_light_y = 200;
    const int radius_light = 10;
    const int space_distance = 5;

    int central_second_light_x;
    int central_second_light_y;
    int central_third_light_x;
    int central_third_light_y;

    int central_traffic_light_boundary_x;
    int central_traffic_light_boundary_y;

    int boundary_width;
    int boundary_height;
    int second_light_distance;
    int third_light_distance;

public:
    //CI6235: Define function for traffic light simulation
    TrafficLight();
    void TrafficLightPosition();
    void DefaultState();
    int TrafficLightSimulation(Mat &img);
    void SimulateTrafficLight(int previous_state, int current_state, Mat &img, int state_time);
    bool EverySecond();
};

#endif // TRAFFIC_LIGHT_H
