#include <iostream>

#include <libBGS.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char *argv[])
{
    if(argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " <video file, output video>" << std::endl;
    }

    std::string vid = argv[1];
    std::cout << vid << std::endl;
    cv::VideoCapture capture(vid);
    if (!capture.isOpened())
    {
        std::cerr << "Failed to open video!\n" << std::endl;
        return 1;
    }

    int width = capture.get(CV_CAP_PROP_FRAME_WIDTH);
    int height = capture.get(CV_CAP_PROP_FRAME_HEIGHT);

    cv::Mat frame, fg_image;
    cv::Mat low_threshold_mask;
    cv::Mat high_threshold_mask;

    // Video Writer
    cv::VideoWriter writer;
    std::string out = argv[2];
    writer.open(out, CV_FOURCC('D','I','V','X'), 30, cv::Size(width,height));

    // AdaptiveMedian
    //bgs::AdaptiveMedianParams params;
    //params.SamplingRate() = 7;
    //params.LearningFrames() = 30;
    //params.LowThreshold() = 40;
    //params.HighThreshold() = 2*params.LowThreshold();
    //bgs::AdaptiveMedian bgs(params);

    // GMM
    bgs::GrimsonParams params;
    params.Alpha() = 0.001f;
    params.MaxModes() = 3;
    params.LowThreshold() = 9;
    params.HighThreshold() = 2*params.LowThreshold();
    bgs::GrimsonGMM bgs(params);

    cv::namedWindow("Video"); cvMoveWindow("Video", 500, 100);
    cv::namedWindow("Background"); cvMoveWindow("Background", 900, 100);
    cv::namedWindow("Foreground Mask"); cvMoveWindow("Foreground Mask", 500, 400);
    cv::namedWindow("Foreground Image"); cvMoveWindow("Foreground Image", 900, 400);

    // Processing
    int frmCnt = 0;

    // perform background subtraction of each frame
    for(;;)
    {
        std::cout << "Processing frame #" << frmCnt << std::endl;

        capture >> frame;
        if(frame.empty())
            break;

        // histogram equilization
        //std::vector<cv::Mat> channels;
        //cv::split(frame,channels);
        //cv::equalizeHist(channels[0], channels[0]);
        //cv::equalizeHist(channels[1], channels[1]);
        //cv::equalizeHist(channels[2], channels[2]);
        //cv::merge(channels,frame);

        // perform background subtraction
        bgs.Subtract(frame, low_threshold_mask, high_threshold_mask);

        // update background subtraction
        bgs.Update(frame, low_threshold_mask);

        // Create Foreground Image
        fg_image.setTo(0);
        frame.copyTo(fg_image, low_threshold_mask);
        
        cv::Mat background = bgs.Background();
        
        cv::imshow("Video", frame);
        cv::imshow("Background", background);
        cv::imshow("Foreground Mask", low_threshold_mask);
        cv::imshow("Foreground Image", fg_image);
        
        // Video Writer
        writer << fg_image;

        frmCnt++;

        char key = cv::waitKey(1000./30.);
        if(key == 'q' || key == 'Q' || key == 27)
            break;
    }

    writer.release();

    return 0;
}

