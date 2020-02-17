#ifndef SR_CLASS
#define SR_CLASS 

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <Eigen/Dense>
#include "Util.h"
class SR {
public:
	SR(Eigen::ArrayXXcf salient_region_fft_in, Eigen::ArrayXXcf h_hat_star_in, float cov_in, float edge_cov_in, int frame_num_in);
	SR(Eigen::ArrayXXcf salient_region_fft_in, float cov_in, float edge_cov_in, int frame_num_in);
	//SR(Eigen::ArrayXXcf salient_region_fft_in, float cov_in, float edge_cov_in);
	SR(Eigen::ArrayXXf salient_region_in, float cov_in, float edge_cov_in);
	void set_h_star(Eigen::ArrayXXcf h_hat_star_in);
	void set_frame_num(int frame_num_in);
	void set_xy(int x_in, int y_in, int w_in, int h_in);
	Eigen::ArrayXXcf salient_region_fft;
	Eigen::ArrayXXf salient_region;
	Eigen::ArrayXXcf h_hat_star;
	float edge_cov;
	float cov;
	int x;
	int y;
	int w;
	int h;
	int frame_num;
};


#endif // ! 