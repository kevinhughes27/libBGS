#include "WrenGA.hpp"

using namespace bgs;

WrenGA::WrenGA()
{
    m_params = WrenParams();

    m_variance = 36.0f;

    m_frame_num = 0;
}

WrenGA::WrenGA(const BgsParams &p)
{
    m_params = (WrenParams&)p;

    m_variance = 36.0f;

    m_frame_num = 0;
}

WrenGA::~WrenGA()
{

}

void WrenGA::Initalize(const cv::Mat& image)
{
    if(image.type() != CV_8UC3)
        CV_Error( CV_StsUnsupportedFormat, "Only 3-channel 8-bit images are supported in libBGS" );

    // GMM for each pixel
    m_gaussian.resize(m_params.Size());

    int pos = 0;
    for(unsigned int r = 0; r < m_params.Height(); ++r)
    {
        for(unsigned int c = 0; c < m_params.Width(); ++c)
        {
            for(int ch = 0; ch < 3; ++ch)
            {
                m_gaussian[pos].mu[ch] = image.at<cv::Vec3b>(r,c)[ch];
                m_gaussian[pos].var[ch] = m_variance;
            }

            pos++;
        }
    }

    // background
    m_background = cv::Mat(m_params.Height(), m_params.Width(), CV_8UC3);
}

void WrenGA::Save(std::string file)
{

}

void WrenGA::Load(std::string file)
{

}

void WrenGA::Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask)
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
            SubtractPixel(r, c, image.at<cv::Vec3b>(r,c), low_threshold, high_threshold);
            low_threshold_mask.at<unsigned char>(r,c) = low_threshold;
            high_threshold_mask.at<unsigned char>(r,c) = high_threshold;
        }
    }

    m_frame_num++;
}

void WrenGA::Update(const cv::Mat& image,  const cv::Mat& update_mask)
{
    int pos = 0;

    for(unsigned int r = 0; r < m_params.Height(); ++r)
    {
        for(unsigned int c = 0; c < m_params.Width(); ++c)
        {
            // perform conditional updating only if we are passed the learning phase
            if(update_mask.at<unsigned char>(r,c) == BACKGROUND || m_frame_num < m_params.LearningFrames())
            {
                float dR = m_gaussian[pos].mu[0] - image.at<cv::Vec3b>(r,c)[0];
                float dG = m_gaussian[pos].mu[1] - image.at<cv::Vec3b>(r,c)[1];
                float dB = m_gaussian[pos].mu[2] - image.at<cv::Vec3b>(r,c)[2];

                float dist = (dR*dR + dG*dG + dB*dB);

                m_gaussian[pos].mu[0] -= m_params.Alpha()*(dR);
                m_gaussian[pos].mu[1] -= m_params.Alpha()*(dG);
                m_gaussian[pos].mu[2] -= m_params.Alpha()*(dB);

                float sigmanew = m_gaussian[pos].var[0] + m_params.Alpha()*(dist-m_gaussian[pos].var[0]);
                m_gaussian[pos].var[0] = sigmanew < 4 ? 4 : sigmanew > 5*m_variance ? 5*m_variance : sigmanew;

                m_background.at<cv::Vec3b>(r,c)[0] = (unsigned char)(m_gaussian[pos].mu[0] + 0.5);
                m_background.at<cv::Vec3b>(r,c)[1] = (unsigned char)(m_gaussian[pos].mu[1] + 0.5);
                m_background.at<cv::Vec3b>(r,c)[2] = (unsigned char)(m_gaussian[pos].mu[2] + 0.5);
            }

            pos++;
        }
    }
}

void WrenGA::SubtractPixel(int r, int c, const cv::Vec3b& pixel, unsigned char& low_threshold, unsigned char& high_threshold)
{
    unsigned int pos = r*m_params.Width()+c;

    // calculate distance between model and pixel
    float mu[3];
    float var[1];
    float delta[3];
    float dist = 0;
    for(int ch = 0; ch < 3; ++ch)
    {
        mu[ch] = m_gaussian[pos].mu[ch];
        var[0] = m_gaussian[pos].var[0];
        delta[ch] = mu[ch] - pixel[ch];
        dist += delta[ch]*delta[ch];
    }

    // calculate the squared distance and see if pixel fits the B/G model
    low_threshold = BACKGROUND;
    high_threshold = BACKGROUND;

    if(dist > m_params.LowThreshold()*var[0])
        low_threshold = FOREGROUND;
    if(dist > m_params.HighThreshold()*var[0])
        high_threshold = FOREGROUND;
}


