#pragma once
#ifndef UTIL
#define UTIL 
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm> 
// DBoW3

#include <DBoW3.h>
#include <DescManip.h>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#ifdef USE_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

//Eigen 
#include <Eigen/Dense>
#include <fftw3.h>
#include <Eigen/Geometry>

using namespace Eigen;

ArrayXXcf fft(const ArrayXXf& x);
ArrayXXf ifft(const ArrayXXcf& xf);

cv::Mat read_image(cv::VideoCapture &video, int number);

void display_float_img(cv::Mat image);

void display_u8_img(cv::Mat image);

void display_salience_map(cv::Mat salience_map, cv::Mat image_float);

void display_eigen_image(const Eigen::ArrayXXf& image);

void imresize(const Eigen::ArrayXXf& image, Eigen::ArrayXXf& image_out, int row, int col);
#endif