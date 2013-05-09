#include "GrimsonGMM.hpp"

using namespace bgs;

GrimsonGMM::GrimsonGMM()
{
    m_params = GrimsonParams();

    // Tbf - the threshold
    m_bg_threshold = 0.75f;	// 1-cf from the paper

    // Tgenerate - the threshold
    m_variance = 36.0f;		// sigma for the new mode

    m_frame_num = 0;
}

GrimsonGMM::GrimsonGMM(const BgsParams &p)
{
    m_params = (GrimsonParams&)p;

    // Tbf - the threshold
    m_bg_threshold = 0.75f;	// 1-cf from the paper

    // Tgenerate - the threshold
    m_variance = 36.0f;		// sigma for the new mode

    m_frame_num = 0;
}

GrimsonGMM::~GrimsonGMM()
{

}

void GrimsonGMM::Initalize(const cv::Mat& image)
{
    if(image.type() != CV_8UC3)
        CV_Error( CV_StsUnsupportedFormat, "Only 3-channel 8-bit images are supported in libBGS" );

    m_params.SetFrameSize(image.cols, image.rows);
    m_params.Channels() = image.channels();

    // GMM for each pixel
    m_modes.resize(m_params.Size()*m_params.MaxModes());

    // used modes per pixel
    m_modes_per_pixel = cv::Mat::zeros(m_modes_per_pixel.size(), m_modes_per_pixel.type());

	for(unsigned int i = 0; i < m_params.Size()*m_params.MaxModes(); ++i)
	{
		m_modes[i].weight = 0;
		m_modes[i].variance = 0;
		m_modes[i].muR = 0;
		m_modes[i].muG = 0;
		m_modes[i].muB = 0;
		m_modes[i].significants = 0;
	}

    m_background = cv::Mat(m_params.Height(), m_params.Width(), image.type());
}

void GrimsonGMM::Save(std::string file)
{

}

void GrimsonGMM::Load(std::string file)
{

}

void GrimsonGMM::Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask)
{
    if(m_frame_num == 0)
        Initalize(image);

    if(low_threshold_mask.empty())
        low_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    if(high_threshold_mask.empty())
        high_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    unsigned char low_threshold, high_threshold;
	long posPixel;

	// update each pixel of the image
	for(unsigned int r = 0; r < m_params.Height(); ++r)
	{
		for(unsigned int c = 0; c < m_params.Width(); ++c)
		{		
			// update model + background subtract
			posPixel=(r*m_params.Width()+c)*m_params.MaxModes();
			
            SubtractPixel(posPixel, image.at<cv::Vec3b>(r,c), m_modes_per_pixel.at<unsigned char>(r,c), low_threshold, high_threshold);
			
			low_threshold_mask.at<unsigned char>(r,c) = low_threshold;
			high_threshold_mask.at<unsigned char>(r,c) = high_threshold;

			m_background.at<cv::Vec3b>(r,c)[0] = (unsigned char)m_modes[posPixel].muR;
			m_background.at<cv::Vec3b>(r,c)[1] = (unsigned char)m_modes[posPixel].muG;
			m_background.at<cv::Vec3b>(r,c)[2] = (unsigned char)m_modes[posPixel].muB;
		}
	}

    m_frame_num++;
}

void GrimsonGMM::Update(const cv::Mat& image,  const cv::Mat& update_mask)
{
    // it doesn't make sense to have conditional updates in the GMM framework
}

void GrimsonGMM::SubtractPixel(long posPixel, const cv::Vec3b& pixel, unsigned char& numModes, unsigned char& low_threshold, unsigned char& high_threshold)
{
    // calculate distances to the modes (+ sort???)
    // here we need to go in descending order!!!
    long pos;
    bool bFitsPDF=false;
    bool bBackgroundLow=false;
    bool bBackgroundHigh=false;

    float fOneMinAlpha = 1-m_params.Alpha();

    float totalWeight = 0.0f;

    // calculate number of Gaussians to include in the background model
    int backgroundGaussians = 0;
    double sum = 0.0;
    for(int i = 0; i < numModes; ++i)
    {
        if(sum < m_bg_threshold)
        {
            backgroundGaussians++;
            sum += m_modes[posPixel+i].weight;
        }
        else
        {
            break;
        }
    }

    // update all distributions and check for match with current pixel
    for (int iModes=0; iModes < numModes; iModes++)
    {
        pos=posPixel+iModes;
        float weight = m_modes[pos].weight;

        // fit not found yet
        if (!bFitsPDF)
        {
            //check if it belongs to some of the modes
            //calculate distance
            float var = m_modes[pos].variance;
            float muR = m_modes[pos].muR;
            float muG = m_modes[pos].muG;
            float muB = m_modes[pos].muB;

            float dR=muR - pixel[0];
            float dG=muG - pixel[1];
            float dB=muB - pixel[2];

            // calculate the squared distance
            float dist = (dR*dR + dG*dG + dB*dB);

            if(dist < m_params.HighThreshold()*var && iModes < backgroundGaussians)
                bBackgroundHigh = true;

            // a match occurs when the pixel is within sqrt(fTg) standard deviations of the distribution
            if(dist < m_params.LowThreshold()*var)
            {
                bFitsPDF=true;

                // check if this Gaussian is part of the background model
                if(iModes < backgroundGaussians)
                    bBackgroundLow = true;

                //update distribution
                float k = m_params.Alpha()/weight;
                weight = fOneMinAlpha*weight + m_params.Alpha();
                m_modes[pos].weight = weight;
                m_modes[pos].muR = muR - k*(dR);
                m_modes[pos].muG = muG - k*(dG);
                m_modes[pos].muB = muB - k*(dB);

                //limit the variance
                float sigmanew = var + k*(dist-var);
                m_modes[pos].variance = sigmanew < 4 ? 4 : sigmanew > 5*m_variance ? 5*m_variance : sigmanew;
                m_modes[pos].significants = m_modes[pos].weight / sqrt(m_modes[pos].variance);
            }
            else
            {
                weight = fOneMinAlpha*weight;
                if (weight < 0.0)
                {
                    weight=0.0;
                    numModes--;
                }

                m_modes[pos].weight = weight;
                m_modes[pos].significants = m_modes[pos].weight / sqrt(m_modes[pos].variance);
            }
        }
        else
        {
            weight = fOneMinAlpha*weight;
            if (weight < 0.0)
            {
                weight=0.0;
                numModes--;
            }
            m_modes[pos].weight = weight;
            m_modes[pos].significants = m_modes[pos].weight / sqrt(m_modes[pos].variance);
        }

        totalWeight += weight;
    }

    // renormalize weights so they add to one
    double invTotalWeight = 1.0 / totalWeight;
    for (int iLocal = 0; iLocal < numModes; iLocal++)
    {
        m_modes[posPixel + iLocal].weight *= (float)invTotalWeight;
        m_modes[posPixel + iLocal].significants = m_modes[posPixel + iLocal].weight
                                                                                                / sqrt(m_modes[posPixel + iLocal].variance);
    }

    // Sort significance values so they are in desending order.
    std::sort(m_modes.begin()+posPixel, m_modes.begin()+posPixel+numModes, compareGMM());

    // make new mode if needed and exit
    if (!bFitsPDF)
    {
        if (numModes < m_params.MaxModes())
        {
            numModes++;
        }
        else
        {
            // the weakest mode will be replaced
        }

        pos = posPixel + numModes-1;

        m_modes[pos].muR = pixel[0];
        m_modes[pos].muG = pixel[1];
        m_modes[pos].muB = pixel[2];
        m_modes[pos].variance = m_variance;
        m_modes[pos].significants = 0;			// will be set below

    if (numModes==1)
            m_modes[pos].weight = 1;
        else
            m_modes[pos].weight = m_params.Alpha();

        //renormalize weights
        int iLocal;
        float sum = 0.0;
        for (iLocal = 0; iLocal < numModes; iLocal++)
        {
            sum += m_modes[posPixel+ iLocal].weight;
        }

        double invSum = 1.0/sum;
        for (iLocal = 0; iLocal < numModes; iLocal++)
        {
            m_modes[posPixel + iLocal].weight *= (float)invSum;
            m_modes[posPixel + iLocal].significants = m_modes[posPixel + iLocal].weight
                                                                                                / sqrt(m_modes[posPixel + iLocal].variance);

        }
    }

    // Sort significance values so they are in desending order.
    std::sort(m_modes.begin()+posPixel, m_modes.begin()+posPixel+numModes, compareGMM());

    if(bBackgroundLow)
    {
        low_threshold = BACKGROUND;
    }
    else
    {
        low_threshold = FOREGROUND;
    }

    if(bBackgroundHigh)
    {
        high_threshold = BACKGROUND;
    }
    else
    {
        high_threshold = FOREGROUND;
    }
}

