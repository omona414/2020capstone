/* g++ threadcam_realrealrealuart.cpp -o threadcam_realrealrealuart `pkg-config --cflags --libs opencv4` -pthread */

/* Includes */
#include "class_timer.hpp"
#include "class_detector.h"

#include <iostream>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <memory>
#include <unistd.h>    // Used for UART
#include <sys/fcntl.h> // Used for UART
#include <termios.h>   // Used for UART

// Define Constants
const char *uart_target = "/dev/ttyTHS1";
#define NSERIAL_CHAR 256
#define VMINX 1
#define BAUDRATE B115200

/* Flags */
// running flag
#define STOP 0
#define RUNNING 1
uint8_t isRunning;

// yellow detecting
#define YELLOW_DETECTING 0
#define YELLOW_SELECTED 1
uint8_t isYellowSelected;

// lane detecting
#define LANE_DETECTING_WAIT 0
#define LANE_DETECTING_START 1
uint8_t isLaneDetectingStart;

// edge detecting
#define EDGE_DETECTING_WAIT 0
#define EDGE_DETECTING_START 1
uint8_t isEdgeDetectingStart;

// object detecting
#define OBJECT_DETECTING_WAIT 0
#define OBJECT_DETECTING_START 1
uint8_t isObjectDetectingStart;

/* Namespaces */
using namespace cv;
using namespace std;
using std::thread;

/* UART */
int fid = -1;
struct termios port_options;
unsigned char rx_buffer[VMINX];

/* Global Variables */
// color detect mask
Vec3b Label[6];
int thres = 70;
int roiystart = 250;
Rect roi(0, roiystart, 480, (320- roiystart));

// camera frame
Mat lane_camera_src, obj_camera_src;
Mat lane_detect_img, obj_detect_img;
Mat img_color_detected_roi, img_edge;
Mat obj_camera_res;
std::stringstream stream;
int labelcnt[4];
unsigned int destinationnum = 0;
char uartcode;


// Mat img_src(480, 360, CV_8UC1);

/* Functions */
// UART Configuration
void UART_CONFIG()
{
    tcgetattr(fid, &port_options); // Get the current attributes of the Serial port

    fid = open(uart_target, O_RDWR | O_NOCTTY);

    tcflush(fid, TCIFLUSH);
    tcflush(fid, TCIOFLUSH);

    if (fid == -1)
    {
        printf("Error - Unable to open UART.  Ensure it is not in use by another application\n");
    }

    port_options.c_cflag &= ~PARENB;                         // Disables the Parity Enable bit(PARENB),So No Parity
    port_options.c_cflag &= ~CSTOPB;                         // CSTOPB = 2 Stop bits,here it is cleared so 1 Stop bit
    port_options.c_cflag &= ~CSIZE;                          // Clears the mask for setting the data size
    port_options.c_cflag |= CS8;                             // Set the data bits = 8
    port_options.c_cflag &= ~CRTSCTS;                        // No Hardware flow Control
    port_options.c_cflag |= CREAD | CLOCAL;                  // Enable receiver,Ignore Modem Control lines
    port_options.c_iflag &= ~(IXON | IXOFF | IXANY);         // Disable XON/XOFF flow control both input & output
    port_options.c_iflag &= ~(ICANON | ECHO | ECHOE | ISIG); // Non Cannonical mode
    port_options.c_oflag &= ~OPOST;                          // No Output Processing

    port_options.c_lflag = 0; //  enable raw input instead of canonical,

    port_options.c_cc[VMIN] = VMINX; // Read at least 1 character
    port_options.c_cc[VTIME] = 0;    // Wait indefinetly

    cfsetispeed(&port_options, BAUDRATE); // Set Read  Speed
    cfsetospeed(&port_options, BAUDRATE); // Set Write Speed

    // Set the attributes to the termios structure
    int att = tcsetattr(fid, TCSANOW, &port_options);

    if (att != 0)
    {
        printf("\nERROR in Setting port attributes");
    }
    else
    {
        printf("\nSERIAL Port Good to Go.\n");
    }

    // Flush Buffers
    tcflush(fid, TCIFLUSH);
    tcflush(fid, TCIOFLUSH);
}

// camera pipeline
string gstreamer_pipeline(int sensor_id, int sensor_mode, int capture_width, int capture_height, int display_width, int display_height, int framerate, int flip_method)
{
    return "nvarguscamerasrc sensor-id=" + to_string(sensor_id) + " sensor-mode=" + to_string(sensor_mode) +
           " ! video/x-raw(memory:NVMM), width=(int)" + to_string(capture_width) + ", height=(int)" +
           to_string(capture_height) + ", format=(string)NV12, framerate=(fraction)" + to_string(framerate) +
           "/1 ! nvvidconv flip-method=" + to_string(flip_method) + " ! video/x-raw, width=(int)" + to_string(display_width) + ", height=(int)" +
           to_string(display_height) + ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}

// mouse callback for detect color
void mouse_callback(int event, int x, int y, int flags, void *param)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        Vec3b color_pixel = lane_detect_img.at<Vec3b>(y, x);

        Mat bgr_color = Mat(1, 1, CV_8UC3, color_pixel);
        Mat hsv_color;

        cvtColor(bgr_color, hsv_color, COLOR_BGR2HSV);

        int hue = hsv_color.at<Vec3b>(0, 0)[0];
        int saturation = hsv_color.at<Vec3b>(0, 0)[1];
        int value = hsv_color.at<Vec3b>(0, 0)[2];

        cout << "hue = " << hue << endl;
        cout << "saturation = " << saturation << endl;
        cout << "value = " << value << endl;

        if (hue < 10)
        {
            cout << "hue < 10" << endl;
            Label[0] = Vec3b(hue - 10 + 180, thres, thres);
            Label[1] = Vec3b(180, 255, 255);
            Label[2] = Vec3b(0, thres, thres);
            Label[3] = Vec3b(hue, 255, 255);
            Label[4] = Vec3b(hue, thres, thres);
            Label[5] = Vec3b(hue + 10, 255, 255);
        }
        else if (hue > 170)
        {
            cout << "hue > 170" << endl;
            Label[0] = Vec3b(hue, thres, thres);
            Label[1] = Vec3b(180, 255, 255);
            Label[2] = Vec3b(0, thres, thres);
            Label[3] = Vec3b(hue + 10 - 180, 255, 255);
            Label[4] = Vec3b(hue - 10, thres, thres);
            Label[5] = Vec3b(hue, 255, 255);
        }
        else
        {
            cout << "case 3" << endl;
            Label[0] = Vec3b(hue, thres, thres);
            Label[1] = Vec3b(hue + 10, 255, 255);
            Label[2] = Vec3b(hue - 10, thres, thres);
            Label[3] = Vec3b(hue, 255, 255);
            Label[4] = Vec3b(hue - 10, thres, thres);
            Label[5] = Vec3b(hue, 255, 255);
        }

        cout << "hue = " << hue << endl;
        cout << "#1 = " << Label[0] << "~" << Label[1] << endl;
        cout << "#2 = " << Label[2] << "~" << Label[3] << endl;
        cout << "#3 = " << Label[4] << "~" << Label[5] << endl;
        setMouseCallback("lane_detect_camera", NULL);
        isYellowSelected = YELLOW_SELECTED;
    }
}

/* Threads */
//UART RX thread
void UART_RX_th()
{
//    UART_CONFIG();
    unsigned char serial_message[NSERIAL_CHAR];
    bool pickup = true;
    int rx_length;
    int nread = 0;

    while (isRunning)
    {
        if (fid != -1)
        {
            nread++;

            rx_length = read(fid, (void *)rx_buffer, VMINX); // Filestream, buffer to store in, number of bytes to read (max)
            destinationnum = (unsigned int)*rx_buffer;
            printf("Event %d, rx_length=%d, Read=%s\n", nread, rx_length, rx_buffer);
            printf("Event %d, rx_length=%d, Read=%s\n", nread, rx_length, rx_buffer);
            printf("Event %d, rx_length=%d, Read=%s\n", nread, rx_length, rx_buffer);
            printf("Event %d, rx_length=%d, Read=%s\n", nread, rx_length, rx_buffer);
            printf("Event %d, rx_length=%d, Read=%s\n", nread, rx_length, rx_buffer);
            printf("Event %d, rx_length=%d, Read=%s\n", nread, rx_length, rx_buffer);
            printf("Event %d, rx_length=%d, Read=%s\n", nread, rx_length, rx_buffer);
            printf("Event %d, rx_length=%d, Read=%s\n", nread, rx_length, rx_buffer);
            
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);

            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            printf("des: %d\n", destinationnum);
            
            usleep(500);
            

            if (rx_length >= 0)
            {
                if (nread <= NSERIAL_CHAR)
                    serial_message[nread - 1] = rx_buffer[0]; // Build message 1 character at a time

                if (rx_buffer[0] == '#')
                    pickup = false; // # symbol is terminator
            }
        }
    }
    close(fid);
}

// lane detect camera thread
void lane_Camera_th(VideoCapture cap)
{
    while (isRunning)
    {
        if (!cap.read(lane_camera_src))
        {
            cout << "Lane detect capture read error" << endl;
            break;
        }
    }
}

// object detect camera thread
void object_Camera_th(VideoCapture cap)
{
    while (isRunning)
    {
        if (!cap.read(obj_camera_src))
        {
            cout << "Object detect capture read error" << endl;
            break;
        }
    }
}

// lane detection thread
void lane_Detection_th()
{
//    UART_CONFIG();

    while (isRunning)
    {
        if (isLaneDetectingStart == LANE_DETECTING_START)
        {
            Mat img_roi = img_color_detected_roi.clone();
            Mat img_gray;

            int lowThreshold = 50;
            int highThreshold = 150;
            cvtColor(img_roi, img_gray, COLOR_BGR2GRAY);
            blur(img_gray, img_edge, Size(3, 3));
            Canny(img_edge, img_edge, lowThreshold, highThreshold, 3);

            vector<Vec2f> linesL;
            vector<Vec2f> linesR;
            HoughLines(img_edge, linesL, 1, CV_PI / 180, 40, 0, 0, 0, CV_PI / 2);
            HoughLines(img_edge, linesR, 1, CV_PI / 180, 40, 0, 0, CV_PI / 2, CV_PI);

            int j = 1;
            int h = 1;
            int o = 1;
            int mo = 0;
            int na = 0;
            Point pt1, pt2;
            Point avgptL1, avgptL2;
            Point avgptR1, avgptR2;
            Point yptL1, yptL2;
            Point yptR1, yptR2;
            Point center1, center2;
            double lean = 0, yzeroincpt = 0, ymaxincpt = 0, xincpt = 0;
            int intlean = 0;
            char uartlean = 0;

            for (size_t i = 0; i < linesL.size(); i++)
            {
                if (i == 0)
                {
                    j = 0;
                    o = 0;
                    mo = 1;
                }
                float rho = linesL[i][0], theta = linesL[i][1];
                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a) + roiystart);
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a) + roiystart);
                avgptL1.x += pt1.x;
                avgptL2.x += pt2.x;
                avgptL1.y += pt1.y;
                avgptL2.y += pt2.y;
                j++;
            }

            avgptL1.x = avgptL1.x / j;
            avgptL2.x = avgptL2.x / j;
            avgptL1.y = avgptL1.y / j;
            avgptL2.y = avgptL2.y / j;

            if (j >= 2)
            {
                lean = (avgptL1.y - avgptL2.y) / (double)(avgptL1.x - avgptL2.x);
                xincpt = lean * avgptL1.x - avgptL1.y;
                yzeroincpt = xincpt / lean;
                ymaxincpt = (xincpt + 320) / lean;
                yzeroincpt = cvRound((int)yzeroincpt);
                ymaxincpt = cvRound((int)ymaxincpt);
                yptL1.x = yzeroincpt;
                yptL1.y = 0;
                yptL2.x = ymaxincpt;
                yptL2.y = 320;
                line(lane_detect_img, yptL1, yptL2, Scalar(0, 0, 255), 1, 8);
            }

            for (size_t i = 0; i < linesR.size(); i++)
            {
                if (i == 0)
                {
                    h = 0;
                    o = 0;
                    na = 1;
                }
                float rho = linesR[i][0], theta = linesR[i][1];
                //Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a) + roiystart);
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a) + roiystart);
                avgptR1.x += pt1.x;
                avgptR2.x += pt2.x;
                avgptR1.y += pt1.y;
                avgptR2.y += pt2.y;
                h++;
            }
            avgptR1.x = avgptR1.x / h;
            avgptR2.x = avgptR2.x / h;
            avgptR1.y = avgptR1.y / h;
            avgptR2.y = avgptR2.y / h;
            if (h >= 2)
            {
                lean = (avgptR1.y - avgptR2.y) / (double)(avgptR1.x - avgptR2.x);
                xincpt = lean * avgptR1.x - avgptR1.y;
                yzeroincpt = xincpt / lean;
                ymaxincpt = (xincpt + 320) / lean;
                yzeroincpt = cvRound((int)yzeroincpt);
                ymaxincpt = cvRound((int)ymaxincpt);
                yptR1.x = yzeroincpt;
                yptR1.y = 0;
                yptR2.x = ymaxincpt;
                yptR2.y = 320;
                line(lane_detect_img, yptR1, yptR2, Scalar(255, 0, 0), 1, 8);
            }

            center1.x = (yptL1.x + yptR1.x) / (o + mo + na);
            center1.y = (yptL1.y + yptR1.y) / (o + mo + na);
            center2.x = (yptL2.x + yptR2.x) / (o + mo + na);
            center2.y = (yptL2.y + yptR2.y) / (o + mo + na);

            line(lane_detect_img, center1, center2, Scalar(0, 255, 0), 1, 8);

            if ((mo == 1) || (na == 1))
            {
                lean = (center1.y - center2.y) / (double)(center1.x - center2.x);
                //cout << "before lean : " << lean << endl;
                lean = atan2((center1.y - center2.y), (center1.x - center2.x));
                //cout << "before lean2 : " << lean << endl;
                intlean = cvRound((int)(lean * 180 / 3.1415 * (-1) / 180 * 180));
                cout << "int  lean : " << intlean << endl;
                //uart_lean((char)uartlean);
                uartlean = (char)intlean;
                write(fid, &uartlean, 1);
                cout << "char lean : " << uartlean << endl;
            }
            isEdgeDetectingStart = EDGE_DETECTING_START;
        }
    }
}

// object detection thread
void object_Detection_th()
{
	Config config_v4_tiny;
	config_v4_tiny.net_type = YOLOV4_TINY;
	config_v4_tiny.detect_thresh = 0.5;
	// config_v4_tiny.file_model_cfg = "../configs/yolov4-tiny.cfg";
	// config_v4_tiny.file_model_weights = "../configs/yolov4-tiny.weights";
	config_v4_tiny.file_model_cfg = "../configs/fifth/yolov4-tiny-obj.cfg";
	config_v4_tiny.file_model_weights = "../configs/fifth/yolov4-tiny-obj_best.weights";
	config_v4_tiny.calibration_image_list_file_txt = "../configs/calibration_images.txt";
	config_v4_tiny.inference_precison = FP32;

//    UART_CONFIG();

    std::unique_ptr<Detector> detector(new Detector());
    detector->init(config_v4_tiny);

    std::vector<BatchResult> batch_res;

    // isObjectDetectingStart = OBJECT_DETECTING_START;

    while(isRunning)
    {
        cv::Mat temp_img = obj_camera_src.clone();
 		std::vector<cv::Mat> batch_img;

        batch_img.push_back(temp_img);

//         //prepare batch data
        //  cv::Mat temp_img = obj_camera_src.clone();
 		// std::vector<cv::Mat> batch_img;

        //  batch_img.push_back(temp_img);

//         //detect
        detector->detect(batch_img, batch_res);

//         //disp
        for (int i = 0; i < batch_img.size(); ++i)
        {
            for (const auto &r : batch_res[i])
            {
                std::cout << "batch " << i << " id:" << r.id << " prob:" << r.prob << " rect:" << r.rect << std::endl;
                cv::rectangle(obj_detect_img, r.rect, cv::Scalar(255, 0, 0), 2);
                stream << std::fixed << std::setprecision(2) << "id:" << r.id << "  score:" << r.prob;
                cv::putText(obj_detect_img, stream.str(), cv::Point(r.rect.x, r.rect.y - 5), 0, 0.5, cv::Scalar(0, 0, 255), 2);
                if(destinationnum >0 && ((r.rect.x)*(r.rect.y)>10500))
                labelcnt[r.id]++;
                if(labelcnt[destinationnum]==2)
                {
                    uartcode = 255;
                    write(fid, &uartcode, 1);
                    uartcode = 0;
                    labelcnt[0] = 0;
                    labelcnt[1] = 0;
                    labelcnt[2] = 0;
                    labelcnt[3] = 0;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                    cout << "dest" << destinationnum << endl;
                }
                else if(labelcnt[0]==2)
                {
                    uartcode = 254;
                    write(fid, &uartcode, 1);
                    uartcode = 0;
                    labelcnt[0] = 0;
                    labelcnt[1] = 0;
                    labelcnt[2] = 0;
                    labelcnt[3] = 0;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                    cout << "fork" << endl;
                }
                else if(labelcnt[3]==2)
                {
                    uartcode = 255;
                    write(fid, &uartcode, 1);
                    uartcode = 0;
                    labelcnt[0] = 0;
                    labelcnt[1] = 0;
                    labelcnt[2] = 0;
                    labelcnt[3] = 0;
                    destinationnum = 0;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                    cout << "parking" << endl;
                }
            }
         }
     }
}

// main thread
void main_th()
{
    // clock var for calculate fps
    clock_t begin, end;

    // initialize flags
    isRunning = RUNNING;
    isYellowSelected = YELLOW_DETECTING;
    isLaneDetectingStart = LANE_DETECTING_WAIT;
    isObjectDetectingStart = OBJECT_DETECTING_WAIT;
    isEdgeDetectingStart = EDGE_DETECTING_WAIT;

    // camera options
    int lane_camera = 0;
    int obj_camera = 1;
    int sensor_mode = 3;
    int capture_width = 480;
    int capture_height = 320;
    int display_width = 480;
    int display_height = 320;
    int framerate = 15;
    int flip_method = 0;

    string lane_camera_pipeline = gstreamer_pipeline(lane_camera,
                                                     sensor_mode,
                                                     capture_width,
                                                     capture_height,
                                                     display_width,
                                                     display_height,
                                                     framerate,
                                                     flip_method);
    VideoCapture lane_cap(lane_camera_pipeline, CAP_GSTREAMER);
    if (!lane_cap.isOpened())
        cout << "Failed to open lane detect camera." << endl;
    thread lane_Camera_thread(lane_Camera_th, lane_cap);

    string obj_camera_pipeline = gstreamer_pipeline(obj_camera,
                                                    sensor_mode,
                                                    capture_width,
                                                    capture_height,
                                                    display_width,
                                                    display_height,
                                                    framerate,
                                                    flip_method);
    VideoCapture obj_cap(obj_camera_pipeline, CAP_GSTREAMER);
    if (!obj_cap.isOpened())
        cout << "Failed to open object detect camera." << endl;
    thread object_Camera_thread(object_Camera_th, obj_cap);

    UART_CONFIG();

    thread lane_Detection_thread(lane_Detection_th);
	thread object_Detection_thread(object_Detection_th);
    
    thread UART_RX_thread(UART_RX_th);

    Mat img_hsv;
    namedWindow("lane_detect_camera", WINDOW_AUTOSIZE);
    namedWindow("yellow_detect", WINDOW_AUTOSIZE);
    namedWindow("obj_detect_camera", WINDOW_AUTOSIZE);
    namedWindow("edge_detected_camera", WINDOW_AUTOSIZE);
    moveWindow("lane_detect_camera", 500, 500);
    
	//namedWindow("image", WINDOW_AUTOSIZE);

    setMouseCallback("lane_detect_camera", mouse_callback);

    while (isRunning)
    {
        begin = clock();

        lane_detect_img = lane_camera_src.clone();
        obj_detect_img = obj_camera_src.clone();

        // color detect
        Mat img_color_detected, edge_detected_img;
        Mat draw_lane;
        Mat img_mask, img_mask1, img_mask2, img_mask3;
        cvtColor(lane_detect_img, img_hsv, COLOR_BGR2HSV);
        inRange(img_hsv, Label[0], Label[1], img_mask1);
        inRange(img_hsv, Label[2], Label[3], img_mask2);
        inRange(img_hsv, Label[4], Label[5], img_mask3);
        img_mask = img_mask1 | img_mask2 | img_mask3;
        bitwise_and(lane_detect_img, lane_detect_img, img_color_detected, img_mask);
        img_color_detected_roi = img_color_detected(roi);

        

        if (isYellowSelected == YELLOW_DETECTING)
        {
 //           cout << "Choose yellow color." << endl;
        }
        else
        {
            isLaneDetectingStart = LANE_DETECTING_START;
                        
        }

        if (isEdgeDetectingStart == EDGE_DETECTING_START)
        {
            edge_detected_img = img_edge.clone();
            imshow("edge_detected_camera", edge_detected_img);
        }
        imshow("yellow_detect", img_color_detected);
        imshow("lane_detect_camera", lane_detect_img);
        imshow("obj_detect_camera", obj_detect_img);

        end = clock();
//        cout << "fps: " << 1 / (double(end - begin) / CLOCKS_PER_SEC) << endl;

        if ((waitKey(30) & 0xff) == 27)
        {
            isRunning = STOP;
            break;
        }
    }

    lane_Camera_thread.join();
    object_Camera_thread.join();
    lane_Detection_thread.join();
    object_Detection_thread.join();
    UART_RX_thread.join();
    destroyAllWindows();
    lane_cap.release();
    obj_cap.release();
}

/* Main Function */
int main()
{
    thread main_thread(main_th);
	
    main_thread.join();
}