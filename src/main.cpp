#include <iostream>

#include "Util.h"
#include "LoopClass.h"
#include "kcc.h"
#include "saliency_detection.h"
#include "saliency_region.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm> 
// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif



// ds nasty global buffers
cv::Ptr<cv::FeatureDetector> keypoint_detector = cv::ORB::create(1000);
cv::Ptr<cv::DescriptorExtractor> descriptor_extractor = cv::ORB::create();
//cv::Ptr<cv::Feature2D> fdetector = cv::xfeatures2d::SIFT::create(1000);//cv::BRISK::create();cv::AKAZE::create();cv::xfeatures2d::SURF::create(400, 4, 2, EXTENDED_SURF);

																			 /*
uint32_t number_of_processed_images = 0;
uint64_t number_of_stored_descriptors = 0;
double image_display_scale = 1;
double maximum_descriptor_distance = 50;
uint32_t number_of_images_interspace = 500;
*/
void performance_test(cv::VideoCapture video);
void groundtruth_test();
bool geometry_reidentification(cv::VideoCapture video,int i, int j);
bool temporal_reidentification(ArrayXXf loop_result, int num_count);
bool geometry_reidentification_gt(cv::VideoCapture video, int i, int j);
bool geometry_reidentification_feature_matching(cv::VideoCapture video, int i, int j);
KCC_Database Database;
int32_t main() {
	//groundtruth_test();
	//return -1;
	cv::VideoCapture video("../../../Dataset/KITTI/sequence00//KITTI1_grey.avi");
	int width = (int)video.get(cv::CAP_PROP_FRAME_WIDTH);
	int height = (int)video.get(cv::CAP_PROP_FRAME_HEIGHT);
	int video_size = (int)video.get(cv::CAP_PROP_FRAME_COUNT);
	int frame_rate = (int)video.get(cv::CAP_PROP_FPS);

	std::ofstream fout("saliency_feature_extraction.txt");
	std::cout << "timer start ...." << std::endl;
	Saliency_Extraction salienceFilter;
	//salienceFilter.init_param(m, n);
	Database.init(video_size);
	ArrayXXf loop_result = ArrayXXf::Zero(video_size, 1);
	//performance_test(video);

	auto t_start = std::chrono::high_resolution_clock::now();

	for(int num_count =0; num_count< video_size; num_count++){
		std::cout << num_count << std::endl;

		cv::Mat image = read_image(video, num_count);
		//color conversion
		cvtColor(image, image, CV_BGR2GRAY);
		//cv::resize(image, image, cv::Size(4, 3));
		//image.convertTo(image, CV_32F);
		//std::cout << image << std::endl;
		/*
		Eigen::Map<Eigen::ArrayXXf> eigenT(&image.at<float>(0, 0), image.cols, image.rows);
		ArrayXXf saliencymap;
		salienceFilter.saliency_extraction_eigen(eigenT.transpose(), saliencymap);
		*/
		
		/*
		
		auto feature_timer_start = std::chrono::high_resolution_clock::now();
		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		//keypoint_detector->detect(image, keypoints);
		//descriptor_extractor->compute(image, keypoints, descriptors);
		fdetector->detectAndCompute(image, cv::Mat(), keypoints, descriptors);
		//std::cout << keypoints[999].pt<< std::endl;
		//std::cout << descriptors.row(999) << std::endl;
		//std::cout << "point size"<< descriptors.rows <<"*"<<descriptors.cols<< std::endl;
		//std::cout << "point size" << keypoints.size() << std::endl;
		auto feature_timer_end = std::chrono::high_resolution_clock::now();
		std::cout << "feature time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(feature_timer_end - feature_timer_start).count()) << std::endl;
		fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(feature_timer_end - feature_timer_start).count()) << "\n";
		*/
		
		/* time evaluation
		image.convertTo(image, CV_32F);
		cv::Mat saliency_map = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8U);;
		auto saliency_timer_start = std::chrono::high_resolution_clock::now();
		salienceFilter.saliency_extraction(image, saliency_map);
		auto saliency_timer_end = std::chrono::high_resolution_clock::now();
		std::cout << "saliency time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(saliency_timer_end - saliency_timer_start).count()) << std::endl;
		fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(saliency_timer_end - saliency_timer_start).count()) << "\n";
		*/
		image.convertTo(image, CV_32F);
		cv::Mat saliency_map = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8U);;
		auto saliency_timer_start = std::chrono::high_resolution_clock::now();
		salienceFilter.saliency_extraction(image, saliency_map);
		
		std::vector<SR> sr_arr;
		salienceFilter.salience_filtering(saliency_map, image,sr_arr);
		
		//if(sr_arr.size()==0)
		//	std::cout << "no feature in current frame" << std::endl;
		for (int p = 0;p < sr_arr.size();p++) {
			cv::rectangle(image, cv::Rect(sr_arr[p].x, sr_arr[p].y, sr_arr[p].w, sr_arr[p].h), cv::Scalar(0), 2);
			sr_arr[p].set_frame_num(num_count);
			std::vector<Loop> loop_candidate = Database.query_database(sr_arr[p]);

			for (int i = 0; i < loop_candidate.size(); i++)
			{
				if (temporal_reidentification(loop_result, num_count)) {
					loop_result(num_count,0) = 1;
					//std::cout << "loop detecterd" << std::endl;
					//fout << "t1=" << loop_candidate[i].current_frame << '\t' << "coincides with t2=" << loop_candidate[i].loop_frame << "\n";
					break;
				}
				else if(loop_candidate[i].similarity>0.6){
					if (geometry_reidentification_feature_matching(video, num_count, loop_candidate[i].loop_frame)) {
						loop_result(num_count, 0) = 1;
						//std::cout << "loop detecterd" << std::endl;
						//fout << "t1=" << loop_candidate[i].current_frame << '\t' << "coincides with t2=" << loop_candidate[i].loop_frame << "\n";
						break;
					}else {
						continue;
					}
				}
			
				
				//fout << "t1=" << loop_candidate[i].current_frame << '\t' << "coincides with t2=" << loop_candidate[i].loop_frame << "\n";
			}
			
		}
		
		auto saliency_timer_end = std::chrono::high_resolution_clock::now();
		std::cout << "saliency time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(saliency_timer_end - saliency_timer_start).count()) << std::endl;
		fout << double(std::chrono::duration_cast<std::chrono::milliseconds>(saliency_timer_end - saliency_timer_start).count()) << "\n";
		//display_float_img(image);
		
		
	}
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "total object number" << Database.image_database.size() << std::endl;
	fout.close();
	std::cout << "average time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count()) / video_size << "ms" << std::endl;
	std::cout << "end" << std::endl;

	while (1);
	return 0;
}
bool geometry_reidentification(cv::VideoCapture video, int i, int j) {
	cv::Mat image1 = read_image(video, i);
	cv::Mat image2 = read_image(video, j);
	cvtColor(image1, image1, CV_BGR2GRAY);
	cvtColor(image2, image2, CV_BGR2GRAY);
	image1.convertTo(image1, CV_32F);
	image2.convertTo(image2, CV_32F);

	Eigen::Map<Eigen::ArrayXXf> image1_temp(&image2.at<float>(0, 0), image1.cols, image1.rows);
	Eigen::Map<Eigen::ArrayXXf> image2_temp(&image2.at<float>(0, 0), image1.cols, image1.rows);
	Eigen::ArrayXXf image1_eigen = image1_temp.transpose()/255;
	Eigen::ArrayXXf image2_eigen = image2_temp.transpose()/255;
	Eigen::ArrayXXcf image1_eigen_fft = fft(image1_eigen);
	Eigen::ArrayXXcf image2_eigen_fft = fft(image2_eigen);
	Eigen::ArrayXXcf h_hat;
	Database.kcc_train(image1_eigen_fft, h_hat);
	float kcc_result = Database.kcc_test(image1_eigen_fft, image2_eigen_fft, h_hat);
	if (kcc_result > 0.9)
		return true;
	else
		return false;
}
bool geometry_reidentification_gt(cv::VideoCapture video, int i, int j) {
	if (i >= 1570 && i < 1635)
		return true;
	if (i >= 2345 && i < 2460)
		return true;
	if (i >= 3288 && i < 3845)
		return true;
	if (i >= 4450 )
		return true;
	return false;
}
bool geometry_reidentification_feature_matching(cv::VideoCapture video, int i, int j) {
	cv::Mat image1 = read_image(video, i);
	cv::Mat image2 = read_image(video, j);
	cvtColor(image1, image1, CV_BGR2GRAY);
	cvtColor(image2, image2, CV_BGR2GRAY);
	//image1.convertTo(image1, CV_32F);
	//image2.convertTo(image2, CV_32F);
	std::vector<cv::KeyPoint> keypoints1;
	std::vector<cv::KeyPoint> keypoints2;
	cv::Mat descriptors1;
	cv::Mat descriptors2;
	keypoint_detector->detect(image1, keypoints1);
	keypoint_detector->detect(image2, keypoints2);
	descriptor_extractor->compute(image1, keypoints1, descriptors1);
	descriptor_extractor->compute(image2, keypoints2, descriptors2);
	
	std::vector<std::vector<cv::DMatch> > matches;
	
	cv::BFMatcher matcher(cv::NORM_HAMMING);
	matcher.knnMatch(descriptors1, descriptors2, matches,2);

	//cv::FlannBasedMatcher flann;
	//flann.knnMatch(descriptors1, descriptors2, matches, 2);


	std::vector< cv::DMatch > good_matches;
	float distance_th = 0.6;
	for (unsigned m = 0; m < matches.size(); m++) {
		if (matches[m][0].distance <= matches[m][1].distance * distance_th) {
			good_matches.push_back(matches[m][0]);
		}
	}

	//std::cout << j << ":" << good_matches.size() << std::endl;
	if (good_matches.size() > 20){
		return true;
	}
	else
		return false;
	
}

bool temporal_reidentification(ArrayXXf loop_result, int num_count) {
	for (int i = 1; i < 5; i++)
	{
		if (loop_result(num_count - i, 0) > 0.9)
			return true;
	}
	return false;
}
void performance_test(cv::VideoCapture video) {
	cv::Mat image = read_image(video, 0);
	cvtColor(image, image, CV_BGR2GRAY);
	image.convertTo(image, CV_32F);
	cv::resize(image, image, cv::Size(image.cols / 2, image.rows / 2));
	cv::Mat image_out;
	KCC_Database database;
	Eigen::Map<Eigen::ArrayXXf> eigenT(&image.at<float>(0, 0), image.cols, image.rows);
	eigenT = eigenT / 255;
	Eigen::ArrayXXcf image_in = fft(eigenT.transpose());
	Eigen::ArrayXXcf h_hat_star;
	database.kcc_train(image_in, h_hat_star);
	Eigen::ArrayXXcf image_test;
	auto t_start = std::chrono::high_resolution_clock::now();
	for (int i = 0; i < 5000; i++)
	{
		image_test = fft(eigenT);
		//database.kcc_test(image_in, image_in, h_hat_star);
	}
	auto t_end = std::chrono::high_resolution_clock::now();
	std::cout << "dbow time=" << double(std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count())<<"ms" << std::endl;
	
}
void groundtruth_test(void) {
	cv::VideoCapture video("../../../Dataset/KITTI/sequence00//KITTI1_4_4.avi");
	int video_size = (int)video.get(cv::CAP_PROP_FRAME_COUNT);
	std::ofstream fout("saliency_gt.txt");
	//geometry_reidentification_feature_matching(video, 4540, 4539);

	for (int i = 0; i < video_size; i++)
	{
		std::cout << i << std::endl;
		for (int j = 0; j < i-500; j++)
		{
			if (geometry_reidentification_feature_matching(video, i, j)) {
				fout << "t1=" << i << '\t' << "coincides with t2=" << j << "\n";
			}
		}
	}
	fout.close();
}
/*
void fftw_test(void) {
	cv::Mat test_mat = cv::Mat::ones(cv::Size(5, 4), CV_32F);
	test_mat.at<float>(0, 0) = 1;
	test_mat.at<float>(0, 1) = 3;
	test_mat.at<float>(0, 2) = 2;
	test_mat.at<float>(1, 0) = 4;
	test_mat.at<float>(1, 1) = 5;
	test_mat.at<float>(1, 2) = 8;
	test_mat.at<float>(2, 0) = 6;
	test_mat.at<float>(2, 1) = 5;
	test_mat.at<float>(2, 2) = 3;
	std::cout << test_mat << std::endl;

	cv::Mat planes[] = { cv::Mat_<float>(test_mat.clone()), cv::Mat::zeros(test_mat.size(), CV_32F) };
	cv::Mat complexImg;
	cv::merge(planes, 2, complexImg);
	cv::dft(complexImg, complexImg);
	std::cout << complexImg << std::endl;
	
	Eigen::Map<Eigen::ArrayXXf> eigenT(&test_mat.at<float>(0, 0), test_mat.cols, test_mat.rows);
	std::cout << eigenT << std::endl;
	Eigen::FFT<float> fft;
	ArrayXXcf result;
	fft.fwd(result,eigenT);
	

	//Eigen::Map<Eigen::ArrayXXcf> eigenc(&complexImg.at<float>(0, 0), test_mat.cols, test_mat.rows);
	//std::cout << "complex"<< eigenc << std::endl;
	Eigen::Map<Eigen::ArrayXXf> eigenT(&test_mat.at<float>(0, 0), test_mat.cols, test_mat.rows);
	std::cout << eigenT << std::endl;
	ArrayXXcf result = Database.fft(eigenT);
	std::cout << "fftresult " << std::endl;
	std::cout << result << std::endl;
	ArrayXXf mat = Database.ifft(result);
	std::cout << mat << std::endl;
	std::cout << mat << std::endl;

}

void test2(){
		cv::Mat image2 = read_image(video, 49);
		cvtColor(image2, image2, CV_BGR2GRAY);
		//cv::resize(image2, image2, cv::Size(5, 7));
		image2.convertTo(image2, CV_32F);
		Eigen::Map<Eigen::ArrayXXf> eigenT(&image.at<float>(0, 0), image.cols, image.rows);
		Eigen::Map<Eigen::ArrayXXf> eigenT2(&image2.at<float>(0, 0), image2.cols, image2.rows);
		eigenT = eigenT / 255;
		eigenT2 = eigenT2 / 256;
		//std::cout << eigenT.row(0) << std::endl;
		//std::cout << eigenT2.row(0) << std::endl;
		ArrayXXcf image_fft = Database.fft(eigenT);
		ArrayXXcf image_fft2 = Database.fft(eigenT2);
		ArrayXXcf h_hat;
		Database.kcc_train(image_fft, h_hat);
		float result = Database.kcc_test(image_fft, image_fft2, h_hat);
		//std::cout << h_hat.row(0) << std::endl;
		std::cout << result << std::endl;
		std::cout << result << std::endl;
		}
*/