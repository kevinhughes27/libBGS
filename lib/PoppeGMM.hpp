/****************************************************************************
*
* PoppeGMM.hpp
*
* Purpose: Implementation of the Gaussian mixture model (GMM) background
*          subtraction described in:
*
*          "Improved Background Mixture Models for Video Surveillance Applications"
*           by Chris Poppe, Ga¨etan Martens, Peter Lambert, and Rik Van de Walle 2007
*
*
* This code is based on code by Donovan Parks which was based on Z. Zivkovic's code
* Donovan Park's code can be obtained at http://dparks.wikidot.com/
* Zivkovic's code can be obtained at: www.zoranz.net
*
* Author: Kevin Hughes, Febuary 2012
*
******************************************************************************/

#ifndef Poppe_GMM_
#define Poppe_GMM_

#include "Bgs.hpp"

namespace bgs
{

class PoppeParams : public BgsParams
{
public:
    PoppeParams()
    {
        m_alpha = 0.001f;
        m_cgc = 1.8;
        m_max_modes = 3;
        m_low_threshold = 65;
        m_high_threshold = 2*m_low_threshold;    // Note: high threshold is used by post-processing
    }

    float &Alpha() { return m_alpha; }
    int &MaxModes() { return m_max_modes; }
    float &cgc() { return m_cgc; }

    void write(cv::FileStorage& fs) const {} // write serialization
    void read(const cv::FileNode& node){} // read serialization

private:
    // alpha - speed of update - if the time interval you want to average over is T
    // set alpha=1/T.
    float m_alpha;
    // Maximum number of modes (Gaussian components) that will be used per pixel
    int m_max_modes;
    // Consistent Gradual Change parameter - 1.8 gave best results in paper
    float m_cgc;
};

class PoppeGMM : public Bgs
{
private:
    struct GMM
    {
    float variance;
    float muR;
    float muG;
    float muB;
    float weight;
    float significants;        // this is equal to weight / standard deviation and is used to
                            // determine which Gaussians should be part of the background model
    };

    struct compareGMM
    {
        inline bool operator() (const GMM& gmm1, const GMM& gmm2)
        {
            return (gmm1.significants > gmm2.significants);
        }
    };

public:
    PoppeGMM();
    PoppeGMM(const BgsParams& p);
    ~PoppeGMM();

    void Save(std::string file = "PoppeGMM.xml");
    void Load(std::string file = "PoppeGMM.xml");
    void Load(float low_threshold, float high_threshold, std::string file = "PoppeGMM.xml")
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
    void SubtractPixel(long posPixel, const cv::Vec3b& pixel, unsigned char& numModes, unsigned char& lowThreshold, unsigned char& highThreshold);
    bool isPrevModel(const GMM& gmm1, const GMM& gmm2);
    bool isPrevPixel(const cv::Vec3b& pixel1, const cv::Vec3b& pixel2, float std);

    // User adjustable parameters
    PoppeParams m_params;

    // Threshold when the component becomes significant enough to be included into
    // the background model. It is the TB = 1-cf from the paper. So I use cf=0.1 => TB=0.9
    // For alpha=0.001 it means that the mode should exist for approximately 105 frames before
    // it is considered foreground
    float m_bg_threshold; //1-cf from the paper

    // Initial variance for the newly generated components.
    // It will will influence the speed of adaptation. A good guess should be made.
    // A simple way is to estimate the typical standard deviation from the images.
    float m_variance;

    // Dynamic array for the mixture of Gaussians
    std::vector<GMM> m_modes;

    // past pixel and model
    std::vector<cv::Vec3b> m_prevI;
    std::vector<GMM> m_prevModel;

    // Number of Gaussian components per pixel
    cv::Mat m_modes_per_pixel;

    // Current background model
    cv::Mat m_background;
};

}

#endif
