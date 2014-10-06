#include "AdaptiveMedian.hpp"

using namespace bgs;

AdaptiveMedian::AdaptiveMedian()
{
    m_params = AdaptiveMedianParams();
    m_frame_num = 0;
}

AdaptiveMedian::AdaptiveMedian(const BgsParams &p)
{
    m_params = (AdaptiveMedianParams&)p;
    m_frame_num = 0;
}

AdaptiveMedian::~AdaptiveMedian()
{

}

void AdaptiveMedian::Initalize(const cv::Mat& image)
{
    if(image.type() != CV_8UC1 && image.type() != CV_8UC3)
        CV_Error( CV_StsUnsupportedFormat, "Only 1-channel or 3-channel 8-bit images are supported in libBGS" );

    m_params.SetFrameSize(image.cols, image.rows);
    m_params.Channels() = image.channels();
    m_median = image.clone();
}

void AdaptiveMedian::Save(std::string file)
{

}

void AdaptiveMedian::Load(std::string file)
{

}

void AdaptiveMedian::Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask)
{
    if(m_frame_num == 0)
        Initalize(image);

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

void AdaptiveMedian::Update(const cv::Mat& image,  const cv::Mat& update_mask)
{
    if(m_frame_num % m_params.SamplingRate() == 1)
    {
        // update background model
        for (unsigned int r = 0; r < m_params.Height(); ++r)
        {
            for(unsigned int c = 0; c < m_params.Width(); ++c)
            {
                // perform conditional updating only if we are passed the learning phase
                if(update_mask.at<unsigned char>(r,c) == BACKGROUND || m_frame_num < m_params.LearningFrames())
                {
                    if(m_params.Channels() == 3)
                    {
                        for(int ch = 0; ch < 3; ++ch)
                        {
                            if(image.at<cv::Vec3b>(r,c)[ch] > m_median.at<cv::Vec3b>(r,c)[ch])
                            {
                                m_median.at<cv::Vec3b>(r,c)[ch]++;
                            }
                            else if(image.at<cv::Vec3b>(r,c)[ch] < m_median.at<cv::Vec3b>(r,c)[ch])
                            {
                                m_median.at<cv::Vec3b>(r,c)[ch]--;
                            }
                        }
                    }
                    else
                    {
                        if(image.at<unsigned char>(r,c) > m_median.at<unsigned char>(r,c))
                        {
                            m_median.at<unsigned char>(r,c)++;
                        }
                        else if(image.at<unsigned char>(r,c) < m_median.at<unsigned char>(r,c))
                        {
                            m_median.at<unsigned char>(r,c)--;
                        }
                    }

                }
            }
        }
    }
}

void AdaptiveMedian::SubtractPixel(int r, int c, const cv::Vec3b pixel, unsigned char& low_threshold, unsigned char& high_threshold)
{
    // perform background subtraction
    low_threshold = high_threshold = FOREGROUND;

    int diffR = abs(pixel[0] - m_median.at<cv::Vec3b>(r,c)[0]);
    int diffG = abs(pixel[1] - m_median.at<cv::Vec3b>(r,c)[1]);
    int diffB = abs(pixel[2] - m_median.at<cv::Vec3b>(r,c)[2]);

    if(diffR <= m_params.LowThreshold() && diffG <= m_params.LowThreshold() &&  diffB <= m_params.LowThreshold())
    {
        low_threshold = BACKGROUND;
    }

    if(diffR <= m_params.HighThreshold() && diffG <= m_params.HighThreshold() &&  diffB <= m_params.HighThreshold())
    {
        high_threshold = BACKGROUND;
    }
}

void AdaptiveMedian::SubtractPixel(int r, int c, const unsigned char pixel, unsigned char& low_threshold, unsigned char& high_threshold)
{
    // perform background subtraction
    low_threshold = high_threshold = FOREGROUND;

    int diff = abs(pixel - m_median.at<unsigned char>(r,c));

    if(diff <= m_params.LowThreshold())
    {
        low_threshold = BACKGROUND;
    }

    if(diff <= m_params.HighThreshold())
    {
        high_threshold = BACKGROUND;
    }
}



