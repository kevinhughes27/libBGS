/****************************************************************************
*
* PratiMediod.hpp
*
* Purpose: Implementation of the temporal median background
*          subtraction algorithm described in:
*
*          "Detecting Moving Objects, Shosts, and Shadows in Video Stream"
*           by R. Cucchiara et al (2003)
*
*          "Reliable Background Suppression for Complex Scenes"
*           by S. Calderara et al (2006)
*
* Author: Donovan Parks, September 2007
* Modified: Kevin Hughes, 2012
*
* Please note that this is not an implementation of the complete system
* given in the above papers. It simply implements the temporal median background
* subtraction algorithm.
*
******************************************************************************/

#ifndef PRATI_MEDIA_BGS_H
#define PRATI_MEDIA_BGS_H

#include "Bgs.hpp"

namespace bgs
{

class PratiParams : public BgsParams
{
public:
    PratiParams()
    {
        m_sampling_rate = 5;
        m_history_size = 16;
        m_weight = 5;
        m_low_threshold = 30;
        m_high_threshold = 2*m_low_threshold;    // Note: high threshold is used by post-processing
    }

    int &Weight() { return m_weight; }
    int &SamplingRate() { return m_sampling_rate; }
    int &HistorySize() { return m_history_size; }

    void write(cv::FileStorage& fs) const {} // write serialization
    void read(const cv::FileNode& node){} // read serialization

private:
    // The weight parameter controls the amount of influence given to previous background samples
    // see w_b in equation (2) of [1]
    // in [2] this value is set to 1
    int m_weight;
    // Number of samples to consider when calculating temporal mediod value
    int m_history_size;
    // Rate at which to obtain new samples
    int m_sampling_rate;
};

class PratiMediod : public Bgs
{
private:
    // sum of L-inf distances from a sample point to all other sample points
    struct MEDIAN_BUFFER
    {
        std::vector<cv::Vec3b> pixels;            // vector of pixels at give location in image
        std::vector<int> dist;                    // distance from pixel to all other pixels
        int pos;                                                // current position in circular buffer

        cv::Vec3b median;                                // median at this pixel location
        int medianDist;                                    // distance from median pixel to all other pixels
    };

public:
    PratiMediod();
    PratiMediod(const BgsParams& p);
    ~PratiMediod();

    void Save(std::string file = "PratiMediod.xml");
    void Load(std::string file = "PratiMediod.xml");
    void Load(float low_threshold, float high_threshold, std::string file = "PratiMediod.xml")
    {
        Load(file);
        m_params.LowThreshold() = low_threshold;
        m_params.HighThreshold() = high_threshold;
    }

    void Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask);
    void Update(const cv::Mat& image,  const cv::Mat& update_mask);

    cv::Mat Background() { return m_background; }

private:
    void Initalize(const cv::Mat& image);
    void CalculateMasks(int r, int c, const cv::Vec3b& pixel);
    void Combine(const cv::Mat& low_mask, const cv::Mat& high_mask, cv::Mat& output);
    void UpdateMediod(int r, int c, const cv::Mat& new_frame, int& dist);

    PratiParams m_params;
    std::vector<MEDIAN_BUFFER> m_median_buffer;
    cv::Mat m_mask_low_threshold;
    cv::Mat m_mask_high_threshold;
    cv::Mat m_background;
};

}

#endif
