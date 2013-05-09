#include "Eigenbackground.hpp"

using namespace bgs;

Eigenbackground::Eigenbackground()
{
    m_params = EigenbackgroundParams();
    m_frame_num = 0;
}

Eigenbackground::Eigenbackground(const BgsParams &p)
{
    m_params = (EigenbackgroundParams&)p;
    m_frame_num = 0;
}

Eigenbackground::~Eigenbackground()
{

}

void Eigenbackground::Initalize(const cv::Mat& image)
{
    if(image.type() != CV_8UC1 && image.type() != CV_8UC3)
        CV_Error( CV_StsUnsupportedFormat, "Only 1-channel or 3-channel 8-bit images are supported in libBGS" );

    m_params.SetFrameSize(image.cols, image.rows);
    m_params.Channels() = image.channels();

    m_pca = cv::PCA();

    m_background = cv::Mat::zeros(m_params.Height(), m_params.Width(), image.type());
}

void Eigenbackground::Save(std::string file)
{
    cv::FileStorage fs;
    fs.open(file, cv::FileStorage::WRITE);

    // Until I can get the cv::FileStorage working this is a dirty dirty way to save the params

    //fs << "params" << m_params;

    int w = m_params.Width();
    int h = m_params.Height();
    int c = m_params.Channels();
    int l = m_params.LowThreshold();
    int hi = m_params.HighThreshold();
    int hs = m_params.HistorySize();
    int ed = m_params.EmbeddedDim();
    int rv = m_params.RetainedVar();
    int p = m_params.Precision();

    fs << "m_paramsWidth" << w;
    fs << "m_paramsHeight" << h;
    fs << "m_paramsChannels" << c;
    fs << "m_paramsLowThresh" << l;
    fs << "m_paramsHighThresh" << hi;
    fs << "m_paramsHistorySize" << hs;
    fs << "m_paramsEmbeddedDim" << ed;
    fs << "m_paramsRetainedVar" << rv;
    fs << "m_paramsPrecision" << p;


    //fs << "m_pcaImages" << m_pcaImages;
    fs << "m_pcaEigenvectors" << m_pca.eigenvectors;
    fs << "m_pcaEigenvalues" << m_pca.eigenvalues;
    fs << "m_pcaMean" << m_pca.mean;
    fs << "m_background" << m_background;

    fs.release();
}

void Eigenbackground::Load(std::string file)
{
    // changing this load method will have implications for
    // the automated testing routines in testBGS branch bpca_paper

    cv::FileStorage fs;
    fs.open(file, cv::FileStorage::READ);

    // Until I can get the cv::FileStorage working this is a dirty dirty way to save the params

    //fs["Params"] >> m_params;

    int w;
    int h;
    int c;
    int l;
    int hi;
    int hs;
    int ed;
    int rv;
    int p;

    fs["m_paramsWidth"] >> w;
    fs["m_paramsHeight"] >> h;
    fs["m_paramsChannels"] >> c;
    fs["m_paramsWidth"] >> w;
    fs["m_paramsHeight"] >> h;
    fs["m_paramsChannels"] >> c;
    fs["m_paramsLowThresh"] >> l;
    fs["m_paramsHighThresh"] >> hi;
    fs["m_paramsHistorySize"] >> hs;
    fs["m_paramsEmbeddedDim"] >> ed;
    fs["m_paramsRetainedVar"] >> rv;
    fs["m_paramsPrecision"] >> p;

    m_params.SetFrameSize(w, h);
    m_params.Channels() = c;
    m_params.LowThreshold() = l;
    m_params.HighThreshold() = hi;
    m_params.HistorySize() = hs;
    m_params.EmbeddedDim() = ed;
    m_params.RetainedVar() = rv;
    m_params.Precision() = p;


    //fs["m_pcaImages"] >> m_pcaImages;

    cv::Mat eigenvectors;
    cv::Mat eigenvalues;
    cv::Mat mean;

    fs["m_pcaEigenvectors"] >> eigenvectors;
    fs["m_pcaEigenvalues"] >> eigenvalues;
    fs["m_pcaMean"] >> mean;

    m_pca = cv::PCA();
    m_pca.mean = mean;
    m_pca.eigenvalues = eigenvalues;
    m_pca.eigenvectors = eigenvectors;

    fs["m_background"] >> m_background;

    fs.release();

    m_frame_num = m_params.HistorySize(); // dont retrain
    m_frame_num++;
}

void Eigenbackground::Subtract(const cv::Mat& image, cv::Mat& low_threshold_mask, cv::Mat& high_threshold_mask)
{
    if(m_frame_num == 0)
        Initalize(image);

    if(low_threshold_mask.empty())
        low_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    if(high_threshold_mask.empty())
        high_threshold_mask.create(m_params.Height(), m_params.Width(), CV_8U);

    // create eigenbackground
    if(m_frame_num == m_params.HistorySize())
	{
        //std::cout << "==" << std::endl;
		
		// create the eigenspace
        if(m_params.EmbeddedDim() == 0)
        {
            m_pca = cv::PCA(m_pcaImages, cv::Mat(), CV_PCA_DATA_AS_ROW, m_params.RetainedVar());
        }
        else
            m_pca = cv::PCA(m_pcaImages, cv::Mat(), CV_PCA_DATA_AS_ROW, m_params.EmbeddedDim());

        m_background = norm_0_255(m_pca.mean.reshape(m_background.channels(), m_params.Height()));

        // free the image
        //m_pcaImages.release();
	}

    if(m_frame_num >= m_params.HistorySize())
	{
        //std::cout << ">=" << std::endl;

        // project new image into the eigenspace
        cv::Mat image_row = image.clone().reshape(1,1);
        cv::Mat point = m_pca.project(image_row);
		
		// reconstruct point
        cv::Mat reconstruction = m_pca.backProject(point);
        
		// calculate Euclidean distance between new image and its eigenspace projection
		int index = 0;
		for(unsigned int r = 0; r < m_params.Height(); ++r)
		{
			for(unsigned int c = 0; c < m_params.Width(); ++c)
			{
				double dist = 0;
				bool bgLow = true;
				bool bgHigh = true;

                if(m_params.Channels() == 3)
                {
                    for(int ch = 0; ch < m_background.channels(); ++ch)
                    {
                        if(m_params.Precision() == 1)
                            dist = abs(image.at<cv::Vec3b>(r,c)[ch] - reconstruction.at<float>(0,index));
                        else
                            dist = abs(image.at<cv::Vec3b>(r,c)[ch] - reconstruction.at<double>(0,index));

                        if(dist > m_params.LowThreshold())
                            bgLow = false;
                        if(dist > m_params.HighThreshold())
                            bgHigh = false;
                        index++;
                    }
                }
                else
                {
                    if(m_params.Precision() == 1)
                        dist = abs(image.at<unsigned char>(r,c) - reconstruction.at<float>(0,index));
                    else
                        dist = abs(image.at<unsigned char>(r,c) - reconstruction.at<double>(0,index));

                    if(dist > m_params.LowThreshold())
                        bgLow = false;
                    if(dist > m_params.HighThreshold())
                        bgHigh = false;
                    index++;
                }
				
				if(!bgLow)
				{
					low_threshold_mask.at<unsigned char>(r,c) = FOREGROUND;
				}
				else
				{
					low_threshold_mask.at<unsigned char>(r,c) = BACKGROUND;
				}

				if(!bgHigh)
				{
					high_threshold_mask.at<unsigned char>(r,c) = FOREGROUND;
				}
				else
				{
					high_threshold_mask.at<unsigned char>(r,c) = BACKGROUND;
				}
			}
		}
	}
	else 
	{
        //std::cout << "else" << std::endl;

        UpdateHistory(image);

		// set entire image to background since there is not enough information yet
		// to start performing background subtraction
		for(unsigned int r = 0; r < m_params.Height(); ++r)
		{
			for(unsigned int c = 0; c < m_params.Width(); ++c)
			{
				low_threshold_mask.at<unsigned char>(r,c) = BACKGROUND;
				high_threshold_mask.at<unsigned char>(r,c) = BACKGROUND;
			}
		}
	}

    m_frame_num++;
}

cv::Mat Eigenbackground::norm_0_255(cv::InputArray _src)
{
    cv::Mat src = _src.getMat();
    // Create and return normalized image:
    cv::Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, cv::NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

void Eigenbackground::Update(const cv::Mat& image,  const cv::Mat& update_mask)
{
    // the eigenbackground model is not updated (serious limitation!)
}

void Eigenbackground::UpdateHistory(const cv::Mat& image)
{
    cv::Mat image_row = image.clone().reshape(1,1);

    if(m_params.Precision() == 1)
        image_row.convertTo(image_row,CV_32FC1);
    else
        image_row.convertTo(image_row,CV_64FC1);

    m_pcaImages.push_back(image_row);
}
