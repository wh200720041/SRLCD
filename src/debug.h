#ifndef DEBUG_H
#define DEBUG_H
#define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <string.h>
using namespace Eigen;
using namespace std;

// Jeffsan::CPTimer jtimer, timerX;
int img_num=0;


// void printrotation(tf::Quaternion r, string name = "rot:")
// {
//     std::cout<<name<<r.x()<<" "<<r.y()<<" "<<r.z()<<" "<<r.w()<<" end"<<std::endl;
// }

// void printtransform(tf::Transform t, string name = "tran:")
// {
//     std::cout<<name<<t.getOrigin().x()<<" "<<t.getOrigin().y()<<" "<<t.getOrigin().z()<<" end"<<std::endl;
// }

// void save_image(ArrayXXf im, int height, int width, string file_prex)
// {
//     cv::Mat im_cv(height, width, CV_32F, im.data());
//     cv::imwrite("/home/eee/Desktop/images/"+file_prex+to_string(img_num)+".png", im_cv*255);
// }

// void save_matrix(ArrayXXf im, string file_prex)
// {
// 	std::ofstream file("/home/eee/Desktop/images/"+file_prex+to_string(img_num)+".txt");
// 	file<<im;
// }

void show_image(ArrayXXf im, int height, int width, string name)
{
    cv::Mat im_cv(height, width, CV_32F, im.data());
    cv::imshow(name, im_cv);
}


#endif
