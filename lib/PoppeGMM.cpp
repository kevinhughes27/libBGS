#include "PoppeGMM.hpp"

using namespace bgs;

PoppeGMM::PoppeGMM()
{
    m_params = PoppeParams();

    // Tbf - the threshold
    m_bg_threshold = 0.75f;    // 1-cf from the paper

    // Tgenerate - the threshold
    m_variance = 36.0f;        // sigma for the new mode

    m_frame_num = 0;
}

PoppeGMM::PoppeGMM(const BgsParams &p)
{
    m_params = (PoppeParams&)p;

    // Tbf - the threshold
    m_bg_threshold = 0.75f;    // 1-cf from the paper

    // Tgenerate - the threshold
    m_variance = 36.0f;        // sigma for the new mode

    m_frame_num = 0;
}

PoppeGMM::~PoppeGMM()
{

}

void PoppeGMM::Initalize(const cv::Mat& image)
{
    if(image.type() != CV_8UC3)
        CV_Error( CV_StsUnsupportedFormat, "Only 3-channel 8-bit images are supported in libBGS" );

    m_params.SetFrameSize(image.cols, image.rows);
    m_params.Channels() = image.channels();

    // GMM for each pixel
    m_modes.resize(m_params.Size()*m_params.MaxModes());

    // previous Pixel for each pixel
    m_prevI.resize(m_params.Size());

    // previous model for each pixel
    m_prevModel.resize(m_params.Size());

    // used modes per pixel
    m_modes_per_pixel = cv::Mat::zeros(m_params.Width(), m_params.Height(), CV_8UC3);

    for(unsigned int i = 0; i < (int)m_modes.size(); ++i)
    {
        m_modes[i].weight = 0;
        m_modes[i].variance = 0;
        m_modes[i].muR = 0;
        m_modes[i].muG = 0;
        m_modes[i].muB = 0;
        m_modes[i].significants = 0;
    }

    for(unsigned int i = 0; i < (int)m_prevI.size(); ++i)
    {
        m_prevI[i][0] = 0;
        m_prevI[i][1] = 0;
        m_prevI[i][2] = 0;
    }

    for(unsigned int i = 0; i < (int)m_prevModel.size(); ++i)
    {
        m_prevModel[i].weight = 0;
        m_prevModel[i].variance = 0;
        m_prevModel[i].muR = 0;
        m_prevModel[i].muG = 0;
        m_prevModel[i].muB = 0;
        m_prevModel[i].significants = 0;
    }

    // background
    m_background = cv::Mat(m_params.Height(), m_params.Width(), image.type());
}

void PoppeGMM::Save(std::string file)
{

}

void PoppeGMM::Load(std::string file)
{

}

void PoppeGMM::Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask)
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

void PoppeGMM::Update(const cv::Mat& image,  const cv::Mat& update_mask)
{
    // it doesn't make sense to have conditional updates in the GMM framework
}

void PoppeGMM::SubtractPixel(long posPixel, const cv::Vec3b& pixel, unsigned char& numModes, unsigned char& low_threshold, unsigned char& high_threshold)
{

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

    long pos;
    long st_pos = posPixel/m_params.MaxModes();
    bool match=false;
    bool bBackgroundLow=false;
    bool bBackgroundHigh=false;
    float fOneMinAlpha = 1-m_params.Alpha();
    float totalWeight = 0.0f;

    // update all distributions and check for match with current pixel
    for (int iModes=0; iModes < numModes; iModes++)
    {
        pos=posPixel+iModes;
        float weight = m_modes[pos].weight;

        float var = 0;
        float muR = 0;
        float muG = 0;
        float muB = 0;

        float dR = 0;
        float dG = 0;
        float dB = 0;

        float dist = 0;

        // fit not found yet
        if (!match)
        {
            if(isPrevModel(m_modes[pos], m_prevModel[st_pos]))
            {
                if( isPrevPixel(pixel, m_prevI[st_pos], sqrt(var)) )
                {
                    match = true;
                    bBackgroundLow = true;
                    bBackgroundHigh = true;

                    var = m_modes[pos].variance;
                    muR = m_modes[pos].muR;
                    muG = m_modes[pos].muG;
                    muB = m_modes[pos].muB;

                    dR = muR - pixel[0];
                    dG = muG - pixel[1];
                    dB = muB - pixel[2];

                    // calculate the squared distance
                    dist = (dR*dR + dG*dG + dB*dB);
                }
                else
                {
                    //check if it belongs to some of the modes
                    //calculate distance
                    var = m_modes[pos].variance;
                    muR = m_modes[pos].muR;
                    muG = m_modes[pos].muG;
                    muB = m_modes[pos].muB;

                    dR = muR - pixel[0];
                    dG = muG - pixel[1];
                    dB = muB - pixel[2];

                    // calculate the squared distance
                    dist = (dR*dR + dG*dG + dB*dB);

                    // a match occurs when the pixel is within sqrt(fTg) standard deviations of the distribution
                    if(dist < m_params.LowThreshold()*var)
                    {
                        match = true;

                        if(iModes < backgroundGaussians)
                        {
                            bBackgroundLow = true;
                        }
                    }

                    if(dist < m_params.HighThreshold()*var && iModes < backgroundGaussians)
                        bBackgroundHigh = true;
                }
            }
            else
            {
                //check if it belongs to some of the modes
                //calculate distance
                var = m_modes[pos].variance;
                muR = m_modes[pos].muR;
                muG = m_modes[pos].muG;
                muB = m_modes[pos].muB;

                dR = muR - pixel[0];
                dG = muG - pixel[1];
                dB = muB - pixel[2];

                // calculate the squared distance
                dist = (dR*dR + dG*dG + dB*dB);

                // a match occurs when the pixel is within sqrt(fTg) standard deviations of the distribution
                if(dist < m_params.LowThreshold()*var)
                {
                    match = true;

                    if(iModes < backgroundGaussians)
                    {
                        bBackgroundLow = true;
                    }
                }

                if(dist < m_params.HighThreshold()*var && iModes < backgroundGaussians)
                    bBackgroundHigh = true;
            }


            if(match)
            {
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

        if(match == true && low_threshold == BACKGROUND)
        {
            long pos = posPixel/m_params.MaxModes();
            m_prevModel[pos] = m_modes[posPixel];
            m_prevI[pos] = pixel;
        }
    }

    // renormalize weights so they add to one
    double invTotalWeight = 1.0 / totalWeight;
    for (int iLocal = 0; iLocal < numModes; iLocal++)
    {
        m_modes[posPixel + iLocal].weight *= (float)invTotalWeight;
        m_modes[posPixel + iLocal].significants = m_modes[posPixel + iLocal].weight / sqrt(m_modes[posPixel + iLocal].variance);
    }

    // Sort significance values so they are in desending order.
    std::sort(m_modes.begin()+posPixel, m_modes.begin()+posPixel+numModes, compareGMM());

    // make new mode if needed and exit
    if (!match)
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
        m_modes[pos].significants = 0;            // will be set below

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
            m_modes[posPixel + iLocal].significants = m_modes[posPixel + iLocal].weight / sqrt(m_modes[posPixel + iLocal].variance);
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

bool PoppeGMM::isPrevModel(const GMM& gmm1, const GMM& gmm2)
{
    bool a = gmm1.variance == gmm2.variance;
    bool b = gmm1.muR == gmm2.muR;
    bool c = gmm1.muG == gmm2.muG;
    bool d = gmm1.muB == gmm2.muB;

    return a && b && c && d;
}

bool PoppeGMM::isPrevPixel(const cv::Vec3b& pixel1, const cv::Vec3b& pixel2, float std) {

    float dist = pow(pixel1[0] - pixel2[0],2) + pow(pixel1[1] - pixel2[1],2) + pow(pixel1[2] - pixel2[2],2);
    dist = sqrt(dist);

    if(dist < m_params.cgc() * std)
        return true;

    return false;
}
