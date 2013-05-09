/****************************************************************************
*
* ZivkovicGMM.hpp
*
* Purpose: Implementation of the Gaussian mixture model (GMM) background
*		   subtraction described in:
*
*          "Improved adaptive Gausian mixture model for background subtraction"
*           Z.Zivkovic, International Conference Pattern Recognition, UK, August, 2004
*
* This code is based on code by Z. Zivkovic
* Zivkovic's code can be obtained at: www.zoranz.net
*
* Author: Donovan Parks, September 2007
* Modified: Kevin Hughes, 2012
*
******************************************************************************/

#ifndef ZIVKOVIC_GMM_H
#define ZIVKOVIC_GMM_H

#include "Bgs.hpp"

namespace bgs
{

class ZivkovicParams : public BgsParams
{
public:
    ZivkovicParams()
    {
        m_alpha = 0.001f;
        m_max_modes = 3;
        m_low_threshold = 5.0f*5.0f;
        m_high_threshold = 2*m_low_threshold;	// Note: high threshold is used by post-processing
    }

    float &Alpha() { return m_alpha; }
	int &MaxModes() { return m_max_modes; }

    void write(cv::FileStorage& fs) const {} // write serialization
    void read(const cv::FileNode& node){} // read serialization

private:
	// alpha - speed of update - if the time interval you want to average over is T
	// set alpha=1/T. 
	float m_alpha;
	// Maximum number of modes (Gaussian components) that will be used per pixel
	int m_max_modes;
};

class ZivkovicAGMM : public Bgs
{
private:
	struct GMM
	{
		float sigma;
		float muR;
		float muG;
		float muB;
		float weight;
	};

public:
	ZivkovicAGMM();
    ZivkovicAGMM(const BgsParams& p);
	~ZivkovicAGMM();

    void Save(std::string file = "ZivkovicAGMM.xml");
    void Load(std::string file = "ZivkovicAGMM.xml");
    void Load(float low_threshold, float high_threshold, std::string file = "ZivkovicAGMM.xml")
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
    void SubtractPixel(long posPixel, const cv::Vec3b& pixel, unsigned char* pModesUsed, unsigned char& lowThreshold, unsigned char& highThreshold);
	
	// User adjustable parameters
	ZivkovicParams m_params;

	// Threshold when the component becomes significant enough to be included into
	// the background model. It is the TB = 1-cf from the paper. So I use cf=0.1 => TB=0.9
	// For alpha=0.001 it means that the mode should exist for approximately 105 frames before
	// it is considered foreground
	float m_bg_threshold; //1-cf from the paper

	// Initial variance for the newly generated components. 
	// It will will influence the speed of adaptation. A good guess should be made. 
	// A simple way is to estimate the typical standard deviation from the images.
	float m_variance;

	// This is related to the number of samples needed to accept that a component
	// actually exists. 
	float m_complexity_prior;
	
    //image
	int m_num_bands;	//only RGB now ==3

	// dynamic array for the mixture of Gaussians
    std::vector<GMM> m_modes;

	cv::Mat m_background;

	//number of Gaussian components per pixel
	unsigned char* m_modes_per_pixel;
};

}

#endif
