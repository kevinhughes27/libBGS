/****************************************************************************
*
* Bgs.hpp
*
* Purpose: Base class for BGS algorithms.
*
* Author: Donovan Parks, October 2007
* Modified: Kevin Hughes, 2012
*
******************************************************************************/

#ifndef BGS_H_
#define BGS_H_

#include <math.h>
#include <vector>
#include <algorithm>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "BgsParams.hpp"

namespace bgs
{

class Bgs
{
public:
	static const int BACKGROUND = 0;
	static const int FOREGROUND = 255;

    Bgs() {}
    Bgs(const BgsParams& p) {}
	virtual ~Bgs() {}

    virtual void Save(std::string file = "bgs.xml") = 0;
    virtual void Load(std::string file = "bgs.xml") = 0;
    virtual void Load(float low_threshold, float high_threshold, std::string file = "bgs.xml") = 0;

    // Subtract the current frame from the background model and produce a binary foreground mask using both a low and high threshold value.
    virtual void Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask) = 0;
	// Update the background model. Only pixels set to background in update_mask are updated.
    virtual void Update(const cv::Mat& image,  const cv::Mat& update_mask) = 0;

	// Return the current background model.
	virtual cv::Mat Background() = 0;

protected:
    virtual void Initalize(const cv::Mat& image) = 0;

    int m_frame_num;
};

}

#endif
