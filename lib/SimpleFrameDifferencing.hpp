/****************************************************************************
*
* SimpleFrameDifferencing.hpp
*
* Purpose: Implementation of a simple frame differencing background
*		   subtraction algorithm.
*
* Author: Kevin Hughes, 2012
*
******************************************************************************/

#ifndef SIMPLEFRAMEDIFF_
#define SIMPLEFRAMEDIFF_

#include "Bgs.hpp"
#include <queue>

namespace bgs
{

class SimpleFrameDifferencingParams : public BgsParams
{
public:
    SimpleFrameDifferencingParams()
    {
        m_offset = 10;
        m_low_threshold = 3*30*30;
        m_high_threshold = 2*m_low_threshold; // Note: high threshold is used by post-processing
    }

    int &Offset() { return m_offset; }

    void write(cv::FileStorage& fs) const {} // write serialization
    void read(const cv::FileNode& node){} // read serialization

private:
    int m_offset;
};

class SimpleFrameDifferencing : public Bgs
{
public:

    SimpleFrameDifferencing();
    SimpleFrameDifferencing(const BgsParams& p);
    ~SimpleFrameDifferencing();

    void Save(std::string file = "SimpleFrameDifferencing.xml");
    void Load(std::string file = "SimpleFrameDifferencing.xml");
    void Load(float low_threshold, float high_threshold, std::string file = "SimpleFrameDifferencing.xml")
    {
        Load(file);
        m_params.LowThreshold() = low_threshold;
        m_params.HighThreshold() = high_threshold;
    }

    void Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask);
    void Update(const cv::Mat& image,  const cv::Mat& update_mask);

    cv::Mat Background() { return m_frameBuffer.front(); }

private:	
    void Initalize(const cv::Mat& image);
    void SubtractPixel(int r, int c, const cv::Vec3b& pixel, unsigned char& low_threshold, unsigned char& high_threshold);
    void SubtractPixel(int r, int c, const unsigned char pixel, unsigned char& low_threshold, unsigned char& high_threshold);

    SimpleFrameDifferencingParams m_params;
    std::queue<cv::Mat> m_frameBuffer;
};

}

#endif
