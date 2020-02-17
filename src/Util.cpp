#include "Util.h"
fftwf_plan fft_plan;
ArrayXXcf fft(const ArrayXXf& x)
{
	ArrayXXcf xf = ArrayXXcf(x.rows() / 2 + 1, x.cols());
	//fftwf_complex* fourier = (fftwf_complex*)fftw_malloc(sizeof(fftwf_complex) * (3 / 2 + 1) * 3);

	//fftwf_complex *fourier = fftwf_alloc_complex(3* 3);
	//std::complex<float>* cl = new std::complex<float>[3 * (3/2+1)];
	fft_plan = fftwf_plan_dft_r2c_2d(x.cols(), x.rows(), (float(*))(x.data()),
		(float(*)[2])(xf.data()), FFTW_ESTIMATE); // reverse order for column major
	fftwf_execute(fft_plan);
	xf.conservativeResize(x.rows(), x.cols());
	//xf.row(4) = xf.row(3);
	//std::cout << xf.row(0) << std::endl;
	//std::cout << xf.row(1) << std::endl;
	//std::cout << xf.row(2) << std::endl;
	//std::cout<< xf.row(3) <<std::endl;
	for (int i = 0;i < x.rows() - (x.rows() / 2 + 1);i++) {
		xf(x.rows() / 2 + 1 + i, 0) = std::conj(xf((x.rows() + 1) / 2 - 1 - i, 0));
		for (int j = 1;j < x.cols();j++) {
			xf(x.rows() / 2 + 1 + i, j) = std::conj(xf((x.rows() + 1) / 2 - 1 - i, x.cols() - j));
		}
	}
	fftwf_destroy_plan(fft_plan);
	//xf.conservativeResize(x.rows()/2, x.cols());
	return xf;
}


ArrayXXf ifft(const ArrayXXcf& xf)
{
	ArrayXXcf xf_temp = xf;
	//int xf_rows = xf.rows();
	xf_temp.conservativeResize(xf.rows() / 2 + 1, xf.cols());
	ArrayXXf x = ArrayXXf(xf.rows(), xf.cols());

	fft_plan = fftwf_plan_dft_c2r_2d(xf.cols(), xf.rows(), (float(*)[2])(xf_temp.data()),
		(float(*))(x.data()), FFTW_ESTIMATE);
	fftwf_execute(fft_plan);
	fftwf_destroy_plan(fft_plan);
	return x / x.size();
}


cv::Mat read_image(cv::VideoCapture &video, int number) {
	video.set(cv::CAP_PROP_POS_FRAMES, number);
	cv::Mat image;
	video.read(image);
	return image;
}

void display_eigen_image(const Eigen::ArrayXXf& image) {
	Eigen::ArrayXXf image_cv = image * 255;
	//std::cout<< image_cv.rows() <<std::endl;
	cv::Mat display_img = cv::Mat::zeros(cv::Size(image_cv.cols(), image_cv.rows()), CV_32F);
	//std::cout << display_img.rows << std::endl;
	for (int i = 0;i < image_cv.rows();i++) {
		for (int j = 0;j < image_cv.cols();j++) {
			display_img.at<float>(i,j) = image_cv(i, j);
		}
	}
	//std::cout << image_cv.row(0) << std::endl;
	display_img.convertTo(display_img, CV_8UC1);
	cv::imshow("result", display_img);
	cv::waitKey(5);
}

void display_float_img(cv::Mat image) {
	cv::Mat display_img;
	image.convertTo(display_img, CV_8UC1);
	cv::imshow("result", display_img);
	cv::waitKey(5);
}

void display_u8_img(cv::Mat image) {
	cv::imshow("result", image);
	cv::waitKey(5);
}

void display_salience_map(cv::Mat salience_map, cv::Mat image_float) {
	cv::Mat display_img;
	image_float.convertTo(display_img, CV_8UC1);
	
	for (int i = 0;i < display_img.rows;i++) {
		for (int j = 0;j < display_img.cols;j++) {
			if (salience_map.at<unsigned char>(i, j) == 0)
				display_img.at<unsigned char>(i, j) = 0;
		}
	}
	cv::imshow("result", display_img);
	cv::waitKey(5);
}

void imresize(const Eigen::ArrayXXf& image, Eigen::ArrayXXf& image_out, int row, int col) {
	int origional_row = image.rows();
	int origional_col = image.cols();
	Eigen::ArrayXXf image_temp = ArrayXXf::Zero(row, origional_col);
	image_out = ArrayXXf::Zero(row, col);
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < origional_col;j++) {
			float x_f = i * (origional_row-1.0) / (row-1);
			int x = floor(x_f);
			image_temp(i, j) = (x + 1 - x_f) * image(x, j) + (x_f - x) * image(std::min(x + 1, row - 1), j) ;
		} 
	}
	for (int i = 0;i < row;i++) {
		for (int j = 0;j < col;j++) {
			float y_f = j * (origional_col-1.0) / (col-1);
			int y = floor(y_f);
			image_out(i, j) = (y+1-y_f) * image_temp(i, y) + (y_f - y) * image_temp(i, std::min(y + 1, col - 1));
		}
	}
	//std::cout << image_temp << std::endl;
	//std::cout << image_out << std::endl;
	return;
}