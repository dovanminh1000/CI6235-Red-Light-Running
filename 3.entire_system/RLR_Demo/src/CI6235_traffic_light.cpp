#include "CI6235_traffic_light.h"

//CI6235: Check every second
bool TrafficLight::EverySecond()
{
    time_t theTime = time(NULL);
    struct tm *aTime = localtime(&theTime);

    current_second = aTime->tm_sec;
    if (current_second != previous_second)
    {
        previous_second = current_second;
        return true;
    }
    return false;
}

TrafficLight::TrafficLight()
{

    DefaultState(); //CI6235: Default state of traffic light
    TrafficLightPosition();
}

//CI6235: Determine traffic light position on frame
void TrafficLight::TrafficLightPosition()
{
    central_traffic_light_boundary_x = central_first_light_x - radius_light - space_distance;
    central_traffic_light_boundary_y = central_first_light_y - radius_light - space_distance;

    boundary_width = radius_light * 2 + space_distance * 2;
    boundary_height = radius_light * 2 * 3 + space_distance * 4;
    int const second_light_distance = 2 * radius_light + space_distance;
    int const third_light_distance = 2 * second_light_distance;

    central_second_light_x = central_first_light_x;
    central_second_light_y = central_first_light_y + second_light_distance;
    central_third_light_x = central_first_light_x;
    central_third_light_y = central_first_light_y + third_light_distance;
}

//CI6235: Intial state of traffic light
void TrafficLight::DefaultState()
{
    centre_traffic_light.above_light = black_light;
    centre_traffic_light.centre_light = black_light;
    centre_traffic_light.below_light = green_light;
}

//CI6235: Simulate traffic light operation
int TrafficLight::TrafficLightSimulation(Mat &img)
{

    if (EverySecond())
    {
        state_time++;
    }
    if (current_state == STATE_GREEN && state_time > 10) //CI6235: turn on green light for 10 second

    {
        current_state = STATE_YELLOW;
        state_time = 1;
    }
    else if (current_state == STATE_YELLOW && state_time > 2) //CI6235: turn on yellow light for 2 second
    {
        current_state = STATE_RED;
        state_time = 1;
    }
    else if (current_state == STATE_RED && state_time > 25) //CI6235: turn on red light for 2 second
    {
        current_state = STATE_GREEN;
        state_time = 1;
    }

    //CI6235: check status and switch traffic light state
    switch (current_state)
    {
    case STATE_GREEN:

        centre_traffic_light.above_light = black_light;
        centre_traffic_light.centre_light = black_light;
        centre_traffic_light.below_light = green_light;
        break;
    case STATE_YELLOW:

        centre_traffic_light.above_light = black_light;
        centre_traffic_light.centre_light = amber_light;
        centre_traffic_light.below_light = black_light;
        break;

    case STATE_RED:

        centre_traffic_light.above_light = red_light;
        centre_traffic_light.centre_light = black_light;
        centre_traffic_light.below_light = black_light;
        break;
    }

    resize(img, img, Size(640, 480));

    //CI6235: Draw traffic light
    cv::rectangle(img, cv::Rect(central_traffic_light_boundary_x, central_traffic_light_boundary_y, boundary_width, boundary_height), black_light, 2);
    cv::circle(img, cv::Point(central_first_light_x, central_first_light_y), radius_light, centre_traffic_light.above_light, -1);
    cv::circle(img, cv::Point(central_second_light_x, central_second_light_y), radius_light, centre_traffic_light.centre_light, -1);
    cv::circle(img, cv::Point(central_third_light_x, central_third_light_y), radius_light, centre_traffic_light.below_light, -1);
    float text_size = 0.2;

    //CI6235: Display time counter for each traffic light state
    switch (current_state)
    {
    case STATE_RED:
        cv::rectangle(img, cv::Rect(central_first_light_x + 17, central_first_light_y - 11, 50, 25), cv::Scalar(255, 255, 255), -1);
        putText(img, to_string(state_time) + "s",
                Point(central_first_light_x + 20, central_first_light_y + 10), FONT_HERSHEY_DUPLEX, 0.8, red_light, text_size);
        break;
    case STATE_YELLOW:
        cv::rectangle(img, cv::Rect(central_second_light_x + 17, central_second_light_y - 11, 50, 25), cv::Scalar(255, 255, 255), -1);

        putText(img, to_string(state_time) + "s",
                Point(central_second_light_x + 20, central_second_light_y + 10), FONT_HERSHEY_DUPLEX, 0.8, amber_light, text_size);
        break;
    case STATE_GREEN:
        cv::rectangle(img, cv::Rect(central_third_light_x + 17, central_third_light_y - 11, 50, 25), cv::Scalar(255, 255, 255), -1);

        putText(img, to_string(state_time) + "s",
                Point(central_third_light_x + 20, central_third_light_y + 10), FONT_HERSHEY_DUPLEX, 0.8, green_light, text_size);
        break;
    default:
        break;
    }

    return current_state;
}
