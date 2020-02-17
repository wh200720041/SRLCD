// Copyright (c) <2017>, <Nanyang Technological University> All rights reserved.

// This file is part of correlation_flow.

//     correlation_flow is free software: you can redistribute it and/or modify
//     it under the terms of the GNU General Public License as published by
//     the Free Software Foundation, either version 3 of the License, or
//     (at your option) any later version.

//     Foobar is distributed in the hope that it will be useful,
//     but WITHOUT ANY WARRANTY; without even the implied warranty of
//     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//     GNU General Public License for more details.

//     You should have received a copy of the GNU General Public License
//     along with Foobar.  If not, see <http://www.gnu.org/licenses/>.
#ifndef KCC
#define KCC 

#include <fstream>
#include <iostream>
#include <Eigen/Dense>
#include <fftw3.h>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include "saliency_region.h"
#include "Util.h"
#include "LoopClass.h"
//#include <unsupported/Eigen/FFT>
using namespace std;
using namespace Eigen;



class KCC_Database
{
public:

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	std::vector<SR> image_database;
	//std::vector<Loop> loop_database;
	ArrayXXf similarity_matrix;
	//ArrayXXf loop_result;
	std::vector<Loop> query_database(SR& sr);
	// inline
	void kcc_train(const ArrayXXcf& image_fft, ArrayXXcf& h_hat_star);
	float kcc_test(const ArrayXXcf& image_train, const ArrayXXcf& image_test, ArrayXXcf& h_hat_star);
	void add_to_database(SR& sr);
	bool check_basic_papram(SR image1, SR image2);
	void init(int size);
private:
	
	//Eigen::FFT<float> fft;
	//kcc parameter
	float lambda = 0.7;
	float sigma = 0.3;
	int frame_space = 500;

	//filtering param
	float max_cov_error = 0.2;
	float max_edge_error = 0.2;
	float min_size_threshold = 0.6;
	float max_size_error = 0.2;
	float loop_threshold = 0.3;
};


#endif