/****************************************************************************
*
* BgsParams.hpp
*
* Purpose: Base class for BGS parameters. Any parameters common to all BGS
*					 algorithms should be specified directly in this class.
*
* Author: Donovan Parks, May 2008
* Modified: Kevin Hughes, 2012
*
******************************************************************************/

#ifndef BGS_PARAMS_H_
#define BGS_PARAMS_H_

#include <opencv2/core/core.hpp>

namespace bgs
{

class BgsParams
{
public:
    virtual ~BgsParams() {}

	virtual void SetFrameSize(unsigned int width, unsigned int height)
	{
		m_width = width;
		m_height = height;
		m_size = width*height;
	}

	unsigned int &Width() { return m_width; }
	unsigned int &Height() { return m_height; }
    unsigned int &Size() { return m_size; }
    unsigned int &Channels() { return m_channels; }

    float &LowThreshold() { return m_low_threshold; }
    float &HighThreshold() { return m_high_threshold; }

	virtual void write(cv::FileStorage& fs) const = 0; // write serialization
    virtual void read(const cv::FileNode& node) = 0; // read serialization

protected:
	unsigned int m_width;
	unsigned int m_height;
    unsigned int m_size;
    unsigned int m_channels;
    float m_low_threshold;
    float m_high_threshold;
};

}

#endif
