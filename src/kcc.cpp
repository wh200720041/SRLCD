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

#include <math.h>
#include "kcc.h"

void KCC_Database::init(int size) {
	similarity_matrix = ArrayXXf::Zero(size, size);
	//loop_result = ArrayXXf::Zero(size, 1);
}
std::vector<Loop> KCC_Database::query_database(SR& sr) {
	//sample = Eigen::Map<ArrayXXf>(&sample_cv.at<float>(0, 0), width, height);
	float max_similarity = 0;
	cv::Mat cvimage(sr.salient_region.cols(), sr.salient_region.rows(), CV_32FC1, sr.salient_region.data());
	cv::transpose(cvimage, cvimage);
	std::vector<Loop> result;
	for (int i = 0;i < image_database.size();i++) {
		//if (sr.frame_num - image_database[i].frame_num < frame_space)  //68ms
		//	break;
		if (!check_basic_papram(sr, image_database[i])) {
			continue;
		}
		cv::resize(cvimage, cvimage, cv::Size(image_database[i].salient_region.cols(), image_database[i].salient_region.rows()));
		Eigen::Map<Eigen::ArrayXXf> image_eigen(&cvimage.at<float>(0, 0), cvimage.cols, cvimage.rows);;
		//imresize(sr.salient_region, resized_image, image_database[i].salient_region.rows(), );
		float similarity = kcc_test(image_database[i].salient_region_fft, fft(image_eigen.transpose()), image_database[i].h_hat_star);
		similarity_matrix(sr.frame_num, image_database[i].frame_num) = max(similarity_matrix(sr.frame_num, image_database[i].frame_num), similarity);
		if (similarity > loop_threshold && abs(sr.frame_num- image_database[i].frame_num)> frame_space) {
			//std::cout << "loop detecterd" << std::endl;
			result.push_back(Loop(sr.frame_num, image_database[i].frame_num, similarity));
		}
		if (max_similarity < similarity)
			max_similarity = similarity;
	}
	
	if(max_similarity<0.6)
		add_to_database(sr);
	return result;
	//cv::cv2eigen(salient_regions[0],);
}

void KCC_Database::add_to_database(SR& sr) {
	//注意这里要resize
	Eigen::ArrayXXcf h_hat_star;
	kcc_train(sr.salient_region_fft, h_hat_star);
	sr.set_h_star(h_hat_star);
	image_database.push_back(sr);
}




void KCC_Database::kcc_train(const ArrayXXcf& image_fft, ArrayXXcf& h_hat_star)
{
	int N = image_fft.rows()*image_fft.cols();
	ArrayXXf gaussian_mask = ArrayXXf::Zero(image_fft.rows(), image_fft.cols());
	float row_sum = image_fft.rows()*image_fft.rows() / 49.0;
	float col_sum = image_fft.cols()*image_fft.cols() / 49.0;
	for (int i = 0;i < image_fft.rows();i++) {
		for (int j = 0;j < image_fft.cols();j++) {
			gaussian_mask(i,j) = exp(-((i+1)*(i+1)/ row_sum +(j+1)*(j+1)/ col_sum)/2);
		}
	}
	ArrayXXcf g_hat = fft(gaussian_mask);
	//float xx = image_fft.square().abs().sum() / N; // Parseval's Theorem
	float xx = image_fft.abs2().sum() / N; // Parseval's Theorem
	ArrayXXf xy = ifft(image_fft * image_fft.conjugate());
	
	ArrayXXf xxyy = (xx + xx - 2 * xy) / N;
	for (int i = 0;i < xxyy.rows();i++) {
		for (int j = 0;j < xxyy.cols();j++) {
			if (xxyy(i, j) < 0)
				xxyy(i, j) = 0;
		}
	}
	ArrayXXcf kxx_hat = fft((-1 / (sigma*sigma)*xxyy).exp());
	//std::cout << gaussian_mask << std::endl;
	h_hat_star = g_hat / (kxx_hat + lambda);
}
float KCC_Database::kcc_test(const ArrayXXcf& image_train, const ArrayXXcf& image_test, ArrayXXcf& h_hat_star)
{
	int N = image_train.rows()*image_train.cols();
	//float xx = image_train.square().abs().sum() / N; // Parseval's Theorem
	//float yy = image_test.square().abs().sum() / N; // Parseval's Theorem
	//std::cout << "size = "<< image_train.rows()<<"*"<<image_train.cols() << std::endl;
	float xx = image_train.abs2().sum() / N;
	float yy = image_test.abs2().sum() / N;

	ArrayXXf xy = ifft(image_train * image_test.conjugate());
	ArrayXXf xxyy = (xx + yy - 2 * xy) / N;
	for (int i = 0;i < xxyy.rows();i++) {
		for (int j = 0;j < xxyy.cols();j++) {
			if (xxyy(i, j) < 0)
				xxyy(i, j) = 0;
		}
	}
	ArrayXXcf kxy_hat = fft((-1 / (sigma*sigma)*xxyy).exp());
	ArrayXXf result = ifft(kxy_hat*h_hat_star);
	float max_num = 0.0;
	for (int i = 0;i < xxyy.rows()/6;i++) {
		for (int j = 0;j < xxyy.cols()/6;j++) {
			if (result(i, j) > max_num)
				max_num = result(i, j);
		}
	}
	//std::cout << result.row(0) << std::endl;
	return max_num;
}
bool KCC_Database::check_basic_papram(SR image1, SR image2) {
	float common_row = min(image1.y + image1.h, image2.y + image2.h) - max(image1.y, image2.y);
	if(common_row<=0 || common_row/ max(image1.h, image2.h) < min_size_threshold)
		return false;
	float common_col = min(image1.x + image1.w, image2.x + image2.w) - max(image1.x, image2.x);
	if (common_col <= 0 || common_col / max(image1.w, image2.w) < min_size_threshold)
		return false;
	if ((float)abs(image1.w - image2.w) / max(image1.w, image2.w) > max_size_error)
		return false;
	if ((float)abs(image1.h - image2.h) / max(image1.h, image2.h) > max_size_error)
		return false;
	if (fabs((float)image1.w/image1.h - (float)image2.w / image2.h) / max((float)image1.w / image1.h, (float)image2.w / image2.h) > max_size_error)
		return false;
	if((image1.cov - image2.cov)/ image2.cov > max_cov_error)
		return false;
	if ((image1.edge_cov - image2.edge_cov)/ image2.edge_cov > max_edge_error)
		return false;
	return true;
}
/*

CorrelationFlow::CorrelationFlow(ros::NodeHandle nh):nh(nh)
{
    if(!nh.getParam("image_width", width)) ROS_ERROR("Can't get Param image_width");
    if(!nh.getParam("image_height", height)) ROS_ERROR("Can't get Param image_height");
    if(!nh.getParam("focal_x", focal_x)) ROS_ERROR("Can't get Param focal_x");
    if(!nh.getParam("focal_y", focal_y)) ROS_ERROR("Can't get Param focal_y");

    velocity = Vector3d::Zero();
    lowpass_weight = 0.10;
    if(nh.getParam("lowpass_weight", lowpass_weight))
        ROS_WARN("Get lowpass_weight:%f", lowpass_weight);

    lamda = 0.1;
    sigma = 0.2;

    rs_lamda = 0.001;
    rs_sigma = 0.2;

    yaw_rate = 0;
    rs_switch = true;

    ArrayXXf target = ArrayXXf::Zero(width, height);
    target(width/2, height/2) = 1;
    target_fft = fft(target);
    filter_fft = fft(ArrayXXf::Zero(width, height));
    filter_fft_rs = fft(ArrayXXf::Zero(width, height));

    initialized = false;

}


void CorrelationFlow::callback(const sensor_msgs::ImageConstPtr& msg)
{
    timer.tic();
    image = cv_bridge::toCvShare(msg, "mono8")->image;
    image(cv::Rect((image.cols-width)/2, (image.rows-height)/2, width, height)).convertTo(sample_cv, CV_32FC1, 1/255.0);
    
    sample = Eigen::Map<ArrayXXf>(&sample_cv.at<float>(0,0), width, height);
    
    sample_lp = log_polar(sample_cv);

    if (initialized == false)
    {   
        train_fft = fft(sample);
        kernel = gaussian_kernel();
        filter_fft = target_fft/(kernel + lamda);

        train_lp_fft = fft(sample_lp);
        kernel = kernel_lp();
        filter_fft_rs = target_fft/(kernel + rs_lamda);

        initialized = true;
        ros_time = msg->header.stamp.toSec();
        ROS_WARN("initialized.");
        return;
    }

    // update ROS TIME
    double dt = msg->header.stamp.toSec() - ros_time;

    ros_time = msg->header.stamp.toSec();

    compute_trans(sample);

    if (rs_switch == true)
        compute_rs(sample_lp);

    compute_velocity(dt);

    float trans_psr = get_psr(output, max_index[0], max_index[1]);

    float rs_psr;
    if (rs_switch == true)
        rs_psr = get_psr(output_rs, max_index_rs[0], max_index_rs[1]);
    else
        rs_psr = 0;

    publish(msg->header);

    timer.toc("callback:");

    ROS_WARN("vx=%+.4f, vy=%+.4f, vz=%+.7f m/s, wz=%+.7f degree/s with psr: %.1f rs_psr: %.1f", 
        velocity(0), velocity(1), velocity(2), yaw_rate, trans_psr, rs_psr);
}





inline ArrayXXcf CorrelationFlow::kcc_train(const ArrayXXcf& xf,int row, int col)
{
    int N = row * col;

    float xx = xf.square().abs().sum()/N; // Parseval's Theorem
    
	ArrayXXf xxyy = (xx + xx -2 * ifft(xf * xf.conjugate()))/N;

	ArrayXXcf kxx = fft((-1 / (sigma*sigma)*xxyy).exp());
    return fft((-1/(sigma*sigma)*xxyy).exp());
}




inline ArrayXXcf CorrelationFlow::kernel_lp()
{
    unsigned int N = height * width;

    train_lp_square = train_lp_fft.square().abs().sum()/N; // Parseval's Theorem

    float xx = train_lp_square;

    float yy = train_lp_square;

    train_lp_fft_conj = train_lp_fft.conjugate();
    
    xyf = train_lp_fft * train_lp_fft_conj;
    
    xy = ifft(xyf);

    xxyy = (xx+yy-2*xy)/N;

    return fft((-1/(rs_sigma*rs_sigma)*xxyy).exp());
}


inline ArrayXXcf CorrelationFlow::kernel_lp(const ArrayXXcf& xf)
{
    unsigned int N = height * width;

    float xx = xf.square().abs().sum()/N; // Parseval's Theorem

    float yy = train_lp_square;
    
    xyf = xf * train_lp_fft_conj;
    
    xy = ifft(xyf);

    xxyy = (xx+yy-2*xy)/N;

    return fft((-1/(rs_sigma*rs_sigma)*xxyy).exp());
}


inline ArrayXXf CorrelationFlow::log_polar(const cv::Mat img)
{
    cv::Mat log_polar_img;

    cv::Point2f center((float)img.cols/2, (float)img.rows/2);

    double radius = (double)img.rows / 2;

    double M = (double)img.cols / log(radius);

    cv::logPolar(img, log_polar_img, center, M, cv::INTER_LINEAR + cv::WARP_FILL_OUTLIERS);

    return Eigen::Map<ArrayXXf>(&log_polar_img.at<float>(0,0), img.cols, img.rows);
}


inline float CorrelationFlow::get_psr(const ArrayXXf& output, ArrayXXf::Index x, ArrayXXf::Index y)
{
    float side_lobe_mean = (output.sum()-max_response)/(output.size()-1);

    float std  = sqrt((output-side_lobe_mean).square().mean());

    return (max_response - side_lobe_mean)/std;
}


inline void CorrelationFlow::compute_trans(const ArrayXXf& xf)
{
    sample_fft = fft(xf);

    // correlation response of current frame
    kernel = gaussian_kernel(sample_fft);
    output = ifft(filter_fft*kernel);
    max_response = output.maxCoeff(&(max_index[0]), &(max_index[1]));
    
    // update filter
    train_fft = sample_fft;
    kernel = gaussian_kernel();
    filter_fft = target_fft/(kernel + lamda);
}


inline void CorrelationFlow::compute_rs(const ArrayXXf& xf)
{
    sample_fft = fft(xf);

    // correlation response of current frame
    kernel = kernel_lp(sample_fft);
    output_rs = ifft(filter_fft_rs*kernel);
    max_response_rs = output_rs.maxCoeff(&(max_index_rs[0]), &(max_index_rs[1]));
    
    // printf("%d %d\n", max_index_rs[0], max_index_rs[1]);
    // show_image(output_rs/max_response_rs,height,width,"rs_output");
    // cv::waitKey(1);
    // update filter
    train_lp_fft = sample_fft;
    kernel = kernel_lp();
    filter_fft_rs = target_fft/(kernel + rs_lamda);
}


inline void CorrelationFlow::compute_velocity(double dt)
{

    if(dt<1e-5) {ROS_WARN("image msg time stamp is INVALID, set dt=0.03s"); dt=0.03;}

    // veclocity calculation
    float vx = -1.0*((max_index[0]-width/2)/dt)/focal_x;
    float vy = -1.0*((max_index[1]-height/2)/dt)/focal_y;

    float vz = 0;
    if (rs_switch == true)
    {
        double radius = (double)height / 2;
        double M = (double)width / log(radius);
        float scale = exp((max_index_rs[0]-width/2)/M);
        vz = (scale-1)/dt;

        float rotation = (max_index_rs[1]-height/2)*360.0/height;
        yaw_rate = (rotation*M_PI/180.0)/dt;
    }
    
    // printf("scale=%f\n",scale);
    // printf("rotation=%f\n",rotation);

    Vector3d v = Vector3d(vx, vy, vz);
    velocity = lowpass_weight * v + (1-lowpass_weight) * velocity; // low pass filter
}
*/