/****************************************************************************
*
* AdaptiveMedian.hpp
*
* Purpose: Implementation of the simple adaptive median background 
*		   subtraction algorithm described in:
*
*          "Segmentation and tracking of piglets in images"
* 		    by McFarlane and Schofield
*
* Author: Donovan Parks, September 2007
* Modified: Kevin Hughes, 2012
*
******************************************************************************/

#ifndef _ADAPTIVE_MEDIAN
#define _ADAPTIVE_MEDIAN

#include "Bgs.hpp"

namespace bgs
{

class AdaptiveMedianParams : public BgsParams
{
public:
    AdaptiveMedianParams()
    {
        m_samplingRate = 7;
        m_learning_frames = 30;
        m_low_threshold = 40;
        m_high_threshold = 2*m_low_threshold;	// Note: high threshold is used by post-processing
    }

	int &SamplingRate() { return m_samplingRate; }
	int &LearningFrames() { return m_learning_frames; }

    void write(cv::FileStorage& fs) const {} // write serialization
    void read(const cv::FileNode& node){} // read serialization

private:
	int m_samplingRate;
	int m_learning_frames;
};

class AdaptiveMedian : public Bgs
{
public:

    AdaptiveMedian();
    AdaptiveMedian(const BgsParams& p);
    ~AdaptiveMedian();

    void Save(std::string file = "AdaptiveMedian.xml");
    void Load(std::string file = "AdaptiveMedian.xml");
    void Load(float low_threshold, float high_threshold, std::string file = "AdaptiveMedian.xml")
    {
        Load(file);
        m_params.LowThreshold() = low_threshold;
        m_params.HighThreshold() = high_threshold;
    }

    void Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask);
    void Update(const cv::Mat& image,  const cv::Mat& update_mask);

	cv::Mat Background() { return m_median; }

private:	
    void Initalize(const cv::Mat& image);
    void SubtractPixel(int r, int c, const cv::Vec3b pixel, unsigned char& low_threshold, unsigned char& high_threshold);
    void SubtractPixel(int r, int c, const unsigned char pixel, unsigned char& low_threshold, unsigned char& high_threshold);

	AdaptiveMedianParams m_params;
	cv::Mat m_median;
};

}

#endif
