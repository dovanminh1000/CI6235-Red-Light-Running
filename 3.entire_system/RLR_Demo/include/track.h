#pragma once

#include <opencv2/core.hpp>
#include "kalman_filter.h"
#include <vector>

class Track
{
public:
    // Constructor
    Track();

    // Destructor
    ~Track() = default;

    void Init(const cv::Rect &bbox);
    void Predict();
    void Update(const cv::Rect &bbox);
    cv::Rect GetStateAsBbox() const;
    float GetNIS() const;

    int coast_cycles_ = 0, hit_streak_ = 0;

    //CI6235: Decleare variable for evidence collection
    std::vector<cv::Point> his_xy;
    bool get_license_plate = false;
    bool enable_license_plate = true;
    
    int testt = 111;
    cv::Rect detection;

    //std::vector<cv::Point> GetHisXY() const;

private:
    Eigen::VectorXd ConvertBboxToObservation(const cv::Rect &bbox) const;
    cv::Rect ConvertStateToBbox(const Eigen::VectorXd &state) const;

    KalmanFilter kf_;
};
