#include "PratiMediod.hpp"

using namespace bgs;

PratiMediod::PratiMediod()
{
    m_params = PratiParams();
    m_frame_num = 0;
}

PratiMediod::PratiMediod(const BgsParams &p)
{
    m_params = (PratiParams&)p;
    m_frame_num = 0;
}

PratiMediod::~PratiMediod()
{

}

void PratiMediod::Initalize(const cv::Mat& image)
{
    if(image.type() != CV_8UC3)
        CV_Error( CV_StsUnsupportedFormat, "Only 3-channel 8-bit images are supported in libBGS" );

    m_params.SetFrameSize(image.cols, image.rows);
    m_params.Channels() = image.channels();

    m_mask_low_threshold = cv::Mat(m_params.Height(), m_params.Width(), CV_8U);
    m_mask_high_threshold = cv::Mat(m_params.Height(), m_params.Width(), CV_8U);

    m_background = cv::Mat(m_params.Height(), m_params.Width(), image.type());

    m_median_buffer.resize(m_params.Size());
}

void PratiMediod::Save(std::string file)
{

}

void PratiMediod::Load(std::string file)
{

}

void PratiMediod::Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask)
{
    if(m_frame_num == 0)
        Initalize(image);

    if(low_threshold_mask.empty())
        low_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    if(high_threshold_mask.empty())
        high_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    if(m_frame_num < m_params.HistorySize())
    {
        low_threshold_mask = cv::Mat::zeros(low_threshold_mask.size(), low_threshold_mask.type());
        high_threshold_mask = cv::Mat::zeros(high_threshold_mask.size(), high_threshold_mask.type());
        return;
    }

    // update each pixel of the image
    for(unsigned int r = 0; r < m_params.Height(); ++r)
    {
        for(unsigned int c = 0; c < m_params.Width(); ++c)
        {
            // need at least one frame of image before we can start calculating the masks
            CalculateMasks(r, c, image.at<cv::Vec3b>(r,c));
        }
    }

    // combine low and high threshold masks
    Combine(m_mask_low_threshold, m_mask_high_threshold, low_threshold_mask);
    Combine(m_mask_low_threshold, m_mask_high_threshold, high_threshold_mask);

    m_frame_num++;
}

void PratiMediod::Update(const cv::Mat& image,  const cv::Mat& update_mask)
{
    // update the image buffer with the new frame and calculate new median values
    if(m_frame_num % m_params.SamplingRate() == 0)
    {
        if((int)m_median_buffer[0].dist.size() == (int)m_params.HistorySize())
        {
            // subtract distance to sample being removed from all distances
            for(unsigned int r = 0; r < m_params.Height(); ++r)
            {
                for(unsigned int c = 0; c < m_params.Width(); ++c)
                {
                    int i = r*m_params.Width()+c;

                    if(update_mask.at<unsigned char>(r,c) == BACKGROUND)
                    {
                        int oldPos = m_median_buffer[i].pos;
                        for(unsigned int s = 0; s < m_median_buffer[i].pixels.size(); ++s)
                        {
                            int maxDist = 0;
                            for(int ch = 0; ch < 3; ++ch)
                            {
                                int tempDist = abs(m_median_buffer[i].pixels.at(oldPos)[ch] - m_median_buffer[i].pixels.at(s)[ch]);
                                if(tempDist > maxDist)
                                    maxDist = tempDist;
                            }

                            m_median_buffer[i].dist.at(s) -= maxDist;
                        }

                        int dist;
                        UpdateMediod(r, c, image, dist);
                        m_median_buffer[i].dist.at(oldPos) = dist;
                        m_median_buffer[i].pixels.at(oldPos) = image.at<unsigned char>(r,c);
                        m_median_buffer[i].pos++;
                        if(m_median_buffer[i].pos >= m_params.HistorySize())
                            m_median_buffer[i].pos = 0;
                    }
                }
            }
        }
        else
        {
            // calculate sum of L-inf distances for new point and
            // add distance from each sample point to this point to their L-inf sum
            int dist;
            for(unsigned int r = 0; r < m_params.Height(); ++r)
            {
                for(unsigned int c = 0; c < m_params.Width(); ++c)
                {
                    int index = r*m_params.Width()+c;
                    UpdateMediod(r, c, image, dist);
                    m_median_buffer[index].dist.push_back(dist);
                    m_median_buffer[index].pos = 0;
                    m_median_buffer[index].pixels.push_back(image.at<unsigned char>(r,c));
                }
            }
        }
    }
}

void PratiMediod::UpdateMediod(int r, int c, const cv::Mat& new_frame, int& dist)
{
    // calculate sum of L-inf distances for new point and
    // add distance from each sample point to this point to their L-inf sum
    unsigned int i = (r*m_params.Width()+c);

    m_median_buffer[i].medianDist = INT_MAX;

    int L_inf_dist = 0;
    for(unsigned int s = 0; s < m_median_buffer[i].dist.size(); ++s)
    {
        int maxDist = 0;
        for(int ch = 0; ch < 3; ++ch)
        {
            int tempDist = abs(m_median_buffer[i].pixels.at(s)[ch] - new_frame.at<cv::Vec3b>(r,c)[ch]);
            if(tempDist > maxDist)
                maxDist = tempDist;
        }

        // check if point from this frame in the image buffer is the median
        m_median_buffer[i].dist.at(s) += maxDist;
        if(m_median_buffer[i].dist.at(s) < m_median_buffer[i].medianDist)
        {
            m_median_buffer[i].medianDist = m_median_buffer[i].dist.at(s);
            m_median_buffer[i].median = m_median_buffer[i].pixels.at(s);
        }

        L_inf_dist += maxDist;
    }

    dist = L_inf_dist;

    // check if the new point is the median
    if(L_inf_dist < m_median_buffer[i].medianDist)
    {
        m_median_buffer[i].medianDist = L_inf_dist;
        m_median_buffer[i].median = new_frame.at<cv::Vec3b>(r,c);
    }
}

void PratiMediod::Combine(const cv::Mat& low_mask, const cv::Mat& high_mask, cv::Mat& output)
{
    for(unsigned int r = 0; r < m_params.Height(); ++r)
    {
        for(unsigned int c = 0; c < m_params.Width(); ++c)
        {
            output.at<unsigned char>(r,c) = BACKGROUND;

            if(r == 0 || c == 0 || r == m_params.Height()-1 || c == m_params.Width()-1)
                continue;

            if(high_mask.at<unsigned char>(r,c) == FOREGROUND)
            {
                output.at<unsigned char>(r,c) = FOREGROUND;
            }
            else if(low_mask.at<unsigned char>(r,c) == FOREGROUND)
            {
                // consider the pixel to be a F/G pixel if it is 8-connected to
                // a F/G pixel in the high mask
                // check if there is an 8-connected foreground pixel
                if(high_mask.at<unsigned char>(r-1,c-1))
                    output.at<unsigned char>(r,c) = FOREGROUND;
                else if(high_mask.at<unsigned char>(r-1,c))
                    output.at<unsigned char>(r,c) = FOREGROUND;
                else if(high_mask.at<unsigned char>(r-1,c+1))
                    output.at<unsigned char>(r,c) = FOREGROUND;
                else if(high_mask.at<unsigned char>(r,c-1))
                    output.at<unsigned char>(r,c) = FOREGROUND;
                else if(high_mask.at<unsigned char>(r,c+1))
                    output.at<unsigned char>(r,c) = FOREGROUND;
                else if(high_mask.at<unsigned char>(r+1,c-1))
                    output.at<unsigned char>(r,c) = FOREGROUND;
                else if(high_mask.at<unsigned char>(r+1,c))
                    output.at<unsigned char>(r,c) = FOREGROUND;
                else if(high_mask.at<unsigned char>(r+1,c+1))
                    output.at<unsigned char>(r,c) = FOREGROUND;
            }
        }
    }
}

void PratiMediod::CalculateMasks(int r, int c, const cv::Vec3b& pixel)
{
    int pos = r*m_params.Width()+c;

    // calculate l-inf distance between current value and median value
    unsigned char dist = 0;
    for(int ch = 0; ch < 3; ++ch)
    {
        int tempDist = abs(pixel[ch] - m_median_buffer[pos].median(ch));
        if(tempDist > dist)
            dist = tempDist;
    }
    m_background.at<cv::Vec3b>(r,c) = m_median_buffer[pos].median;

    // check if pixel is a B/G or F/G pixel according to the low threshold B/G model
    m_mask_low_threshold.at<unsigned char>(r,c) = BACKGROUND;
    if(dist > m_params.LowThreshold())
    {
        m_mask_low_threshold.at<unsigned char>(r,c) = FOREGROUND;
    }

    // check if pixel is a B/G or F/G pixel according to the high threshold B/G model
    m_mask_high_threshold.at<unsigned char>(r,c)= BACKGROUND;
    if(dist > m_params.HighThreshold())
    {
        m_mask_high_threshold.at<unsigned char>(r,c) = FOREGROUND;
    }
}


