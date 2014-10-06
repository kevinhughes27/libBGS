#include "Mean.hpp"

using namespace bgs;

Mean::Mean()
{
    m_params = MeanParams();
    m_frame_num = 0;
}

Mean::Mean(const BgsParams &p)
{
    m_params = (MeanParams&)p;
    m_frame_num = 0;
}

Mean::~Mean()
{

}

void Mean::Initalize(const cv::Mat& image)
{
    if(image.type() != CV_8UC1 && image.type() != CV_8UC3)
        CV_Error( CV_StsUnsupportedFormat, "Only 1-channel or 3-channel 8-bit images are supported in libBGS" );

    m_params.SetFrameSize(image.cols, image.rows);
    m_params.Channels() = image.channels();

    m_mean = image.clone();
    m_background = cv::Mat(m_params.Height(), m_params.Width(), image.type());
}

void Mean::Save(std::string file)
{

}

void Mean::Load(std::string file)
{

}

void Mean::Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask)
{
    if(m_frame_num == 0)
        Initalize((image));

    if(low_threshold_mask.empty())
        low_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    if(high_threshold_mask.empty())
        high_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    unsigned char low_threshold, high_threshold;

    // update each pixel of the image
    for(unsigned int r = 0; r < m_params.Height(); ++r)
    {
        for(unsigned int c = 0; c < m_params.Width(); ++c)
        {
            // perform background subtraction + update background model
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

void Mean::Update(const cv::Mat& image,  const cv::Mat& update_mask)
{
    // update background model
    for (unsigned int r = 0; r < m_params.Height(); ++r)
    {
        for(unsigned int c = 0; c < m_params.Width(); ++c)
        {
            // perform conditional updating only if we are passed the learning phase
            if(update_mask.at<unsigned char>(r,c) == BACKGROUND || m_frame_num < m_params.LearningFrames())
            {
                // update B/G model
                float mean;
                if(m_params.Channels() == 3)
                {
                    for(int ch = 0; ch < 3; ++ch)
                    {
                        mean = m_params.Alpha() * m_mean.at<cv::Vec3b>(r,c)[ch] + (1.0f-m_params.Alpha()) * image.at<cv::Vec3b>(r,c)[ch];
                        m_mean.at<cv::Vec3b>(r,c)[ch] = mean;
                        m_background.at<cv::Vec3b>(r,c)[ch] = (unsigned char)(mean + 0.5);
                    }
                }
                else
                {
                    mean = m_params.Alpha() * m_mean.at<unsigned char>(r,c) + (1.0f-m_params.Alpha()) * image.at<unsigned char>(r,c);
                    m_mean.at<unsigned char>(r,c) = mean;
                    m_background.at<unsigned char>(r,c) = (unsigned char)(mean + 0.5);
                }
            }
        }
    }
}

void Mean::SubtractPixel(int r, int c, const cv::Vec3b& pixel, unsigned char& low_threshold, unsigned char& high_threshold)
{
    // calculate distance to sample point
    float dist = 0;
    for(int ch = 0; ch < 3; ++ch)
    {
        dist += (pixel[ch]-m_mean.at<cv::Vec3b>(r,c)[ch])*(pixel[ch]-m_mean.at<cv::Vec3b>(r,c)[ch]);
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

void Mean::SubtractPixel(int r, int c, const unsigned char pixel, unsigned char& low_threshold, unsigned char& high_threshold)
{
    // calculate distance to sample point
    float dist = (pixel-m_mean.at<unsigned char>(r,c))*(pixel-m_mean.at<unsigned char>(r,c));

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


