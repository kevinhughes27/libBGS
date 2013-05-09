/****************************************************************************
*
* Mean.hpp
*
* Purpose: Implementation of a simple temporal mean background 
*		   subtraction algorithm.
*
* Author: Donovan Parks, September 2007
* Modified: Kevin Hughes, 2012
*
******************************************************************************/

#ifndef MEAN_
#define MEAN_

#include "Bgs.hpp"

namespace bgs
{

class MeanParams : public BgsParams
{
public:
    MeanParams()
    {
        m_alpha = 1e-6f;
        m_learning_frames = 30;
        m_low_threshold = 3*30*30;
        m_high_threshold = 2*m_low_threshold;	// Note: high threshold is used by post-processing
    }

    float &Alpha() { return m_alpha; }
	int &LearningFrames() { return m_learning_frames; }

    void write(cv::FileStorage& fs) const {} // write serialization
    void read(const cv::FileNode& node){} // read serialization

private:
	float m_alpha;
	int m_learning_frames;
};

class Mean : public Bgs
{
public:

    Mean();
    Mean(const BgsParams& p);
    ~Mean();

    void Save(std::string file = "Mean.xml");
    void Load(std::string file = "Mean.xml");
    void Load(float low_threshold, float high_threshold, std::string file = "Mean.xml")
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
    void SubtractPixel(int r, int c, const cv::Vec3b& pixel, unsigned char& lowThreshold, unsigned char& highThreshold);
    void SubtractPixel(int r, int c, const unsigned char pixel, unsigned char& low_threshold, unsigned char& high_threshold);

	MeanParams m_params;
	cv::Mat m_mean;
	cv::Mat m_background;
};

}

#endif
