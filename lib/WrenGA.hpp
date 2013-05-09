/****************************************************************************
*
* WrenGA.hpp
*
* Purpose: Implementation of the running Gaussian average background 
*		   subtraction algorithm described in:
*
*          "Pfinder: real-time tracking of the human body"
* 			by C. Wren et al (1997)
*
* Author: Donovan Parks, September 2007
* Modified: Kevin Hughes, 2012
*
* Please note that this is not an implementation of Pfinder. It implements
* a simple background subtraction algorithm where each pixel is represented
* by a single Gaussian and update using a simple weighting function.
*
******************************************************************************/

#ifndef WREN_GA_H
#define WREN_GA_H

#include "Bgs.hpp"

namespace bgs
{

class WrenParams : public BgsParams
{
public:
    WrenParams()
    {
        m_alpha = 0.005f;
        m_learning_frames = 30;
        m_low_threshold = 3.5f*3.5f;
        m_high_threshold = 2*m_high_threshold;	// Note: high threshold is used by post-processing
    }

    float &Alpha() { return m_alpha; }
	int &LearningFrames() { return m_learning_frames; }

    void write(cv::FileStorage& fs) const {} // write serialization
    void read(const cv::FileNode& node){} // read serialization

private:
	float m_alpha;
	int m_learning_frames;
};

class WrenGA : public Bgs
{
private:	
	struct GAUSSIAN
	{
		float mu[3];
		float var[3];
	};

public:
	WrenGA();
    WrenGA(const BgsParams& p);
	~WrenGA();

    void Save(std::string file = "WrenGA.xml");
    void Load(std::string file = "WrenGA.xml");
    void Load(float low_threshold, float high_threshold, std::string file = "WrenGA.xml")
    {
        Load(file);
        m_params.LowThreshold() = low_threshold;
        m_params.HighThreshold() = high_threshold;
    }

    void Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask);
    void Update(const cv::Mat& image, const cv::Mat& update_mask);

	cv::Mat Background() { return m_background; }

private:	
    void Initalize(const cv::Mat& image);
    void SubtractPixel(int r, int c, const cv::Vec3b& pixel, unsigned char& lowThreshold, unsigned char& highThreshold);

	WrenParams m_params;

	// Initial variance for the newly generated components. 
	float m_variance;

	// dynamic array for the mixture of Gaussians
    std::vector<GAUSSIAN> m_gaussian;

	cv::Mat m_background;
};

}

#endif
