/****************************************************************************
*
* Eigenbackground.hpp
*
* Purpose: Implementation of the Eigenbackground background subtraction 
*		   algorithm developed by Oliver et al.
*
*          "A Bayesian Computer Vision System for Modeling Human Interactions"
*           Nuria Oliver, Barbara Rosario, Alex P. Pentland 2000
*
* Author: Donovan Parks, September 2007
* Updated: Kevin Hughes, 2012
*
******************************************************************************/

#ifndef _EIGENBACKGROUND_H_
#define _EIGENBACKGROUND_H_

#include "Bgs.hpp"

namespace bgs
{

class EigenbackgroundParams : public BgsParams
{
public:
    EigenbackgroundParams()
    {
        m_history_size = 100;
        m_dim = 20;
        m_precision = 2;
        m_low_threshold = 50;
        m_high_threshold = 2*m_low_threshold;	// Note: high threshold is used by post-processing
    }

    int &HistorySize() { return m_history_size; }
	int &EmbeddedDim() { return m_dim; }
    float &RetainedVar() { return m_var; }
    int &Precision() { return m_precision; }

	void write(cv::FileStorage& fs) const {} // write serialization
    void read(const cv::FileNode& node){} // read serialization

private:
	int m_history_size;			// number frames used to create eigenspace
    int m_dim;					// eigenspace dimensionality
    float m_var;
    int m_precision;
};

}

namespace bgs
{

class Eigenbackground : public Bgs
{
public:
	Eigenbackground();
    Eigenbackground(const BgsParams& p);
	~Eigenbackground();

    void Save(std::string file = "Eigenbackground.xml");
    void Load(std::string file = "Eigenbackground.xml");
    void Load(float low_threshold, float high_threshold, std::string file = "Eigenbackground.xml")
    {
        Load(file);
        m_params.LowThreshold() = low_threshold;
        m_params.HighThreshold() = high_threshold;
    }

    cv::Mat norm_0_255(cv::InputArray _src);
    void Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask);
    void Update(const cv::Mat& image,  const cv::Mat& update_mask);

	cv::Mat Background() { return m_background; }

private:
    void Initalize(const cv::Mat& image);
    void UpdateHistory(const cv::Mat& newFrame);

	EigenbackgroundParams m_params;
    cv::Mat m_pcaImages;
    int m_K;
    cv::PCA m_pca;
	cv::Mat m_background;
};

}

#endif
