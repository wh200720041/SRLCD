// OpenCV
#ifndef SALIENCY_DETECTION
#define SALIENCY_DETECTION 

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include "Util.h"
#include "saliency_region.h"
//#include "kcc.h"
class Saliency_Extraction {
	
public:
	//void init_param(int image_width,  int image_height);
	void saliency_extraction(const cv::Mat& image, cv::Mat& saliency_map); //input must be float image
	void saliency_extraction_eigen(Eigen::ArrayXXf image, Eigen::ArrayXXf saliency_map); //input must be float image
	void salience_filtering(const cv::Mat& saliency_map, cv::Mat& image_float, std::vector<SR>& result_arr);
	//void salience_filtering_eigen(Eigen::ArrayXXf saliency_map, Eigen::ArrayXXf image_float, std::vector<SR>& result_arr);
private:
	//spectral residual
	int average_filter_size = 7;
	int median_filter_size = 5;
	double extraction_treshold = 1.3;
	//int width;
	//int height;

	//salience filtering
	int min_width = 30;
	int min_height = 25;
	double min_fill_percentage = 0.0;
	double max_area_percentage = 0.3;
	double min_cov = 58.0;
	double min_edge_cov = 0.48;
	//50,0.4 21886 obj
	//55 0.45 16829 602
};

#endif
