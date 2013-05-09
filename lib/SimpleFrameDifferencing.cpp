#include "SimpleFrameDifferencing.hpp"

using namespace bgs;

SimpleFrameDifferencing::SimpleFrameDifferencing()
{
    m_params = SimpleFrameDifferencingParams();
    m_frame_num = 0;
}

SimpleFrameDifferencing::SimpleFrameDifferencing(const BgsParams &p)
{
    m_params = (SimpleFrameDifferencingParams&)p;
    m_frame_num = 0;
}

SimpleFrameDifferencing::~SimpleFrameDifferencing()
{

}

void SimpleFrameDifferencing::Initalize(const cv::Mat& image)
{
    if(image.type() != CV_8UC3)
        CV_Error( CV_StsUnsupportedFormat, "Only 3-channel 8-bit images are supported in libBGS" );

    m_params.SetFrameSize(image.cols, image.rows);
    m_params.Channels() = image.channels();

    m_frameBuffer.empty();

    // fill the frame buffer
    for(int i = 0; i <= m_params.Offset(); i++)
    {
        m_frameBuffer.push(image.clone());
    }
}

void SimpleFrameDifferencing::Save(std::string file)
{

}

void SimpleFrameDifferencing::Load(std::string file)
{

}

void SimpleFrameDifferencing::Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask)
{
    if(m_frame_num == 0)
        Initalize((image));

    if(low_threshold_mask.empty())
        low_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    if(high_threshold_mask.empty())
        high_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    // maintain buffer
    m_frameBuffer.push(image.clone());
    m_frameBuffer.pop();

    unsigned char low_threshold, high_threshold;

    // update each pixel of the image
    for(unsigned int r = 0; r < m_params.Height(); ++r)
    {
        for(unsigned int c = 0; c < m_params.Width(); ++c)
        {
            // perform background subtraction
            if(m_params.Channels() == 3)
                SubtractPixel(r, c, image.at<cv::Vec3b>(r,c), low_threshold, high_threshold);
            else
                SubtractPixel(r, c, image.at<unsigned char>(r,c), low_threshold, high_threshold);

            // setup silhouette mask
            low_threshold_mask.at<unsigned char>(r,c) = low_threshold;
            high_threshold_mask.at<unsigned char>(r,c) = high_threshold;
        }
    }

    m_frame_num++;
}

void SimpleFrameDifferencing::Update(const cv::Mat& image,  const cv::Mat& update_mask)
{
    // it doesn't make sense to have conditional updates in this framework
}

void SimpleFrameDifferencing::SubtractPixel(int r, int c, const cv::Vec3b& pixel, unsigned char& low_threshold, unsigned char& high_threshold)
{
    // calculate distance to sample point
    float dist = 0;
    for(int ch = 0; ch < 3; ++ch)
    {
        dist += (pixel[ch] - m_frameBuffer.front().at<cv::Vec3b>(r,c)[ch]) * (pixel[ch] - m_frameBuffer.front().at<cv::Vec3b>(r,c)[ch]);
    }

    // determine if sample point is F/G or B/G pixel
    low_threshold = BACKGROUND;
    if(dist > m_params.LowThreshold())
    {
        low_threshold = FOREGROUND;
    }

    high_threshold = BACKGROUND;
    if(dist > m_params.HighThreshold())
    {
        high_threshold = FOREGROUND;
    }
}

void SimpleFrameDifferencing::SubtractPixel(int r, int c, const unsigned char pixel, unsigned char& low_threshold, unsigned char& high_threshold)
{
    // calculate distance to sample point
    float dist = (pixel - m_frameBuffer.front().at<unsigned char>(r,c)) * (pixel - m_frameBuffer.front().at<unsigned char>(r,c));

    // determine if sample point is F/G or B/G pixel
    low_threshold = BACKGROUND;
    if(dist > m_params.LowThreshold())
    {
        low_threshold = FOREGROUND;
    }

    high_threshold = BACKGROUND;
    if(dist > m_params.HighThreshold())
    {
        high_threshold = FOREGROUND;
    }
}


