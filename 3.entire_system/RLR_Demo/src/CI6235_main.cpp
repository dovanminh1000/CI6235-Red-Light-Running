/**
 * SORT: A Simple, Online and Realtime Tracker
 */

#include <iostream>
#include <fstream>
#include <map>
#include <random>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>

#include "tracker.h"
#include "utils.h"

//CI6235: Include tensorflow library
#include <stdio.h>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include "CI6235_traffic_light.h"

#include <chrono>
#include <map>
#include <string>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

#include <iostream>
// OpenCV
#include "opencv2/opencv.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;
using namespace tflite;

#define USE_QUANTIZATION

//CI6235: Use to save ID and trajectory of each vehicle
struct TrackHistory
{
    int ID;
    vector<Point> xy;
};

vector<TrackHistory> track_his;

//CI6235: Preprocess frame when the model is quantized
vector<uint8_t> loadIntImage(Mat img, int input_size)
{
    std::vector<uint8_t> data;
    double fx, fy;

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    fx = double(input_size) / img.cols;
    fy = double(input_size) / img.rows;

    img.convertTo(img, CV_8UC1);
    resize(img, img, Size(), fx, fy, cv::INTER_LINEAR);
    data.assign((uint8_t *)img.data, (uint8_t *)img.data + (input_size * input_size * 3));
    return data;
}

//CI6235: Preprocess frame when the model is not quantized
vector<float> loadFloatImage(Mat img, int input_size)
{
    std::vector<float> data;
    double fx, fy;

    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    fx = double(input_size) / img.cols;
    fy = double(input_size) / img.rows;

    img.convertTo(img, CV_32F);
    resize(img, img, Size(), fx, fy, cv::INTER_LINEAR);
    img = (img * (2.0 / 255.0)) - 1.0;
    data.assign((float *)img.data, (float *)img.data + (input_size * input_size * 3));
    return data;
}

std::vector<std::vector<cv::Rect>> ProcessLabel(std::ifstream &label_file)
{
    // Process labels - group bounding boxes by frame index
    std::vector<std::vector<cv::Rect>> bbox;
    std::vector<cv::Rect> bbox_per_frame;
    // Label index starts from 1
    int current_frame_index = 1;
    std::string line;

    while (std::getline(label_file, line))
    {
        std::stringstream ss(line);
        // Label format <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        std::vector<float> label;
        std::string data;
        while (getline(ss, data, ','))
        {
            label.push_back(std::stof(data));
        }

        if (static_cast<int>(label[0]) != current_frame_index)
        {
            current_frame_index = static_cast<int>(label[0]);
            bbox.push_back(bbox_per_frame);
            bbox_per_frame.clear();
        }

        // Ignore low confidence detections
        if (label[6] > kMinConfidence)
        {
            bbox_per_frame.emplace_back(label[2], label[3], label[4], label[5]);
        }
    }
    // Add bounding boxes from last frame
    bbox.push_back(bbox_per_frame);
    return bbox;
}
int main(int argc, const char *argv[])
{
    //CI6235: Create traffic light object
    TrafficLight *trafficLight;
    trafficLight = new TrafficLight;

    //CI6235: Declare input data
    int input_size = 300; //Ensure this matches the .config
#ifdef USE_QUANTIZATION
    char *model_name = "../model/quantized_RLR.tflite";
#else
    char *model_name = "../model/RLR.tflite";
#endif
    string vid_filename = "../data/CI6235/CI6235_demo.mp4";

    float confidence_threshold = 0.4;

    //CI6235: Load model
    std::unique_ptr<tflite::FlatBufferModel> model =
        tflite::FlatBufferModel::BuildFromFile(model_name);
    if (!model)
    {
        printf("Failed to load model\n");
        exit(0);
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    builder(&interpreter);
    if (!interpreter)
    {
        printf("Failed to build interpreter\n");
        exit(0);
    }

    // Resize input tensors, if desired.
    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        printf("Failed to allocate tensors\n");
        exit(0);
    }
    //CI6235: Define number of threads to run model on multi-core CPU
    interpreter->SetNumThreads(4);

    //CI6235: Read input video
    VideoCapture cap(vid_filename);
    if (!cap.isOpened())
    {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }

    int frame_width = static_cast<int>(cap.get(CAP_PROP_FRAME_WIDTH));   //get the width of frames of the video
    int frame_height = static_cast<int>(cap.get(CAP_PROP_FRAME_HEIGHT)); //get the height of frames of the video

    float x_scale = frame_width / 640.0;
    float y_scale = frame_height / 480.0;
    int num_illegal_vehicle = 0;
    int frame_id = 0;

    /////
    // parse program input arguments
    boost::program_options::options_description desc{"Options"};
    desc.add_options()("help,h", "Help screen")("display,d", "Display online tracker output (slow) [False]");

    boost::program_options::variables_map vm;
    boost::program_options::store(parse_command_line(argc, argv, desc), vm);
    boost::program_options::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << '\n';
        return -1;
    }

    bool enable_display_flag = false;
    if (vm.count("display"))
    {
        enable_display_flag = true;
    }

    std::vector<cv::Scalar> colors;
    if (enable_display_flag)
    {

        // Generate random colors to visualize different bbox
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        constexpr int max_random_value = 20;
        std::uniform_int_distribution<> dis(0, max_random_value);
        constexpr int factor = 255 / max_random_value;

        for (int n = 0; n < kNumColors; ++n)
        {
            //Use dis to transform the random unsigned int generated by gen into an int in [0, 7]
            colors.emplace_back(cv::Scalar(dis(gen) * factor, dis(gen) * factor, dis(gen) * factor));
        }
    }

    // All training dataset in MOT15
    std::vector<std::string> dataset_names{"ADL-Rundle-6", "ADL-Rundle-8", "ETH-Bahnhof",
                                           "ETH-Pedcross2", "ETH-Sunnyday", "KITTI-13",
                                           "KITTI-17", "PETS09-S2L1", "TUD-Campus",
                                           "TUD-Stadtmitte", "Venice-2"};

    // create SORT tracker
    Tracker tracker;

    for (const auto &dataset_name : dataset_names)
    {
        // Open label file and load detections from MOT dataset
        // Note that it can also be replaced by detections from you own detector
        std::string label_path = "../data/" + dataset_name + "/det.txt";
        std::ifstream label_file(label_path);
        if (!label_file.is_open())
        {
            std::cerr << "Could not open or find the label!!!" << std::endl;
            return -1;
        }
        std::vector<std::vector<cv::Rect>> all_detections = ProcessLabel(label_file);
        // Close label file
        label_file.close();

        // Load image paths for visualization
        std::vector<cv::String> images;

        // Create output folder if it does not exist
        std::string output_folder = "../output/";
        boost::filesystem::path output_folder_path(output_folder);
        if (boost::filesystem::create_directory(output_folder_path))
        {
            std::cerr << "Directory Created: " << output_folder << std::endl;
        }

        std::string output_path = output_folder + dataset_name + ".txt";
        std::ofstream output_file(output_path);

        size_t total_frames = all_detections.size();

        for (size_t i = 0; i < total_frames; i++)
        {
            auto t1 = std::chrono::high_resolution_clock::now();

            Mat frame;
            Mat high_res_frame;
            cap >> frame;                 //CI6235: Read input frames
            frame.copyTo(high_res_frame); //CI6235: Keep high resolution image for evidence collection
            if (frame.empty())
                break;

            frame_id++;
            resize(frame, frame, Size(640, 480));
            cv::Mat img = frame.clone();
            auto t1_detection = std::chrono::high_resolution_clock::now();

            //CI6235: Preprocess image for the trained model inference
#ifdef USE_QUANTIZATION
            uint8_t *input = interpreter->typed_input_tensor<uint8_t>(0);
            vector<uint8_t> imgData = loadIntImage(frame, input_size);
            memcpy(input, imgData.data(), imgData.size() * sizeof(uint8_t));
#else
            float *input = interpreter->typed_input_tensor<float>(0);
            vector<float> imgData = loadFloatImage(frame, input_size);
            memcpy(input, imgData.data(), imgData.size() * sizeof(float));
#endif

            interpreter->Invoke();

            //CI6235: Extract output from prediction
            float *output_locations = interpreter->typed_output_tensor<float>(0);
            float *output_classes = interpreter->typed_output_tensor<float>(1);
            float *output_scores = interpreter->typed_output_tensor<float>(2);
            float *output_detections = interpreter->typed_output_tensor<float>(3);

            int H = img.rows;
            int W = img.cols;
            char txt[32];
            vector<Rect> detection_list;
            for (int d = 0; d < *output_detections; d++)
            {
                const int cls = int(output_classes[d]);
                const float score = output_scores[d];
                const int ymin = output_locations[4 * d] * H;
                const int xmin = output_locations[4 * d + 1] * W;
                const int ymax = output_locations[4 * d + 2] * H;
                const int xmax = output_locations[4 * d + 3] * W;
                int width = xmax - xmin;
                int height = ymax - ymin;
                if (score > confidence_threshold) //CI6235: Keep the bounding boxe list of good prediction to pass on tracking model
                {
                    Rect detection = Rect(xmin, ymin, width, height);
                    detection_list.push_back(detection);
                }
            }
            auto t2_detection = std::chrono::high_resolution_clock::now();

            auto frame_index = i + 1;
            /*** Run SORT tracker ***/
            const auto &detections = detection_list;
            tracker.Run(detections); //CI6235: Track vehicles resulted from vehicle prediction module
            const auto tracks = tracker.GetTracks();
            /*** Tracker update done ***/

            for (auto &trk : tracks)
            {
                const auto &bbox = trk.second.GetStateAsBbox();
            }

            // Visualize tracking result
            if (enable_display_flag)
            {
                // Make a copy for display
                cv::Mat img_tracking;
                img.copyTo(img_tracking);

                // Check for invalid input
                if (img.empty())
                {
                    std::cerr << "Could not open or find the image!!!" << std::endl;
                    return -1;
                }
                Mat illegal_vehicle_img;
                int traffic_light_state = trafficLight->TrafficLightSimulation(img_tracking); //CI6235: Simulate traffic light operation

                for (auto &trk : tracks)
                {
                    // only draw tracks which meet certain criteria
                    if (trk.second.coast_cycles_ < kMaxCoastCycles &&
                        (trk.second.hit_streak_ >= kMinHits || frame_index < kMinHits))
                    {
                        const auto &bbox = trk.second.detection;
                        cv::rectangle(img_tracking, bbox, colors[trk.first % kNumColors], 1); //CI6235: Draw bounding boxe of vehicles

                        int his_xy_size = trk.second.his_xy.size();
                        for (int i = 0; i < his_xy_size; i++)
                        {
                            cv::circle(img_tracking, trk.second.his_xy[i], 5, colors[trk.first % kNumColors], 1); //CI6235: Draw vehicle trajectory
                        }

                        if (trk.second.get_license_plate && traffic_light_state == STATE_RED) //CI6235: Take picture of violation vehicle when crossing the red light
                        {
                            num_illegal_vehicle++;

                            int org_x = bbox.x * x_scale;
                            int org_y = bbox.y * y_scale;
                            int org_width = bbox.width * x_scale;
                            int org_height = bbox.height * y_scale;

                            if (org_x < 0)
                                org_x = 0;
                            if (org_width > frame_width)
                                org_width = frame_width;
                            if (org_y < 0)
                                org_y = 0;
                            if (org_height > frame_height)
                                org_height = frame_height;

                            Rect org_bbox = Rect(org_x, org_y, org_width, org_height);
                            illegal_vehicle_img = high_res_frame(org_bbox); //CI6235: Crop violation vehicle from high resoluton image to see clear license plate

                            //CI6235: Get the time happing violation
                            time_t rawtime;
                            struct tm *timeinfo;
                            char buffer[80];
                            time(&rawtime);
                            timeinfo = localtime(&rawtime);
                            strftime(buffer, 80, "%FT%X%z.", timeinfo);
                            string time_stamp = buffer;

                            //CI6235: Draw violation information on evidence frame
                            int img_width = illegal_vehicle_img.cols;
                            cv::rectangle(illegal_vehicle_img, cv::Rect(0, 0, img_width, 50), cv::Scalar(240, 240, 240), CV_FILLED);
                            Scalar textColor = Scalar(0, 0, 255);
                            cv::putText(illegal_vehicle_img, "Violation: Red Light Running", Point(10, 15), FONT_HERSHEY_COMPLEX_SMALL, 0.6, textColor, 0.8);
                            cv::putText(illegal_vehicle_img, "Violation Time: " + time_stamp, Point(10, 30), FONT_HERSHEY_COMPLEX_SMALL, 0.6, textColor, 0.8);
                            cv::putText(illegal_vehicle_img, "Location: Pioneer", Point(10, 45), FONT_HERSHEY_COMPLEX_SMALL, 0.6, textColor, 0.8);

                            string output_path = "../output/evidence/violation_" + to_string(num_illegal_vehicle) + ".jpg"; //CI6235: Save evidence of violation vehicle
                            bool isSuccess = imwrite(output_path, illegal_vehicle_img);                                     //write the image to a file as JPEG
                            if (isSuccess == false)
                            {
                                cout << "Failed to save the image" << endl;
                            }
                        }
                    }
                }

                auto t2 = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
                std::chrono::duration<double> time_span_detection = std::chrono::duration_cast<std::chrono::duration<double>>(t2_detection - t1_detection);

                //std::cout << "********************************" << std::endl;
                //std::cout << "Detection time took: " << time_span_detection.count() * 1000 << "ms" << std::endl;
                //std::cout << "Total time took: " << time_span.count() * 1000 << "ms" << std::endl;
                //std::cout << "FPS = " << 1 / time_span.count() << std::endl;

                //CI6235: Display output image
                cv::rectangle(img_tracking, cv::Rect(0, 0, 640, 40), cv::Scalar(240, 240, 240), CV_FILLED);
                Scalar textColor = Scalar(0, 0, 255);
                String statusText = "Red Light Running Enforcement";
                putText(img_tracking, statusText,
                        Point(60, 30), FONT_HERSHEY_DUPLEX, 1, textColor, 0.6);
                cv::imshow("RLR", img_tracking);
                if (!illegal_vehicle_img.empty())
                {
                    String windowName = "Violation vehicle";
                    namedWindow(windowName);
                    cv::imshow(windowName, illegal_vehicle_img); //CI6235: Display image of violation vehicles
                }

                // Delay in ms
                auto key = cv::waitKey(0);

                // Exit if ESC pressed
                if (27 == key)
                {
                    return 0;
                }
                else if (32 == key)
                {
                    // Press Space to pause and press it again to resume
                    while (true)
                    {
                        key = cv::waitKey(0);
                        if (32 == key)
                        {
                            break;
                        }
                        else if (27 == key)
                        {
                            return 0;
                        }
                    }
                }
            } // end of enable_display_flag
        }     // end of iterating all frames

    } // end of iterating all dataset
    return 0;
}
