#include "saliency_region.h"

void SR::set_h_star(Eigen::ArrayXXcf h_hat_star_in) {
	h_hat_star = h_hat_star_in;
}
void SR::set_frame_num(int frame_num_in) {
	frame_num = frame_num_in;
}
void SR::set_xy(int x_in, int y_in, int w_in, int h_in) {
	x = x_in;
	y = y_in;
	w = w_in;
	h = h_in;

}
/*
SR::SR(Eigen::ArrayXXcf salient_region_fft_in, float cov_in, float edge_cov_in) {
	salient_region_fft = salient_region_fft_in;
	cov = cov_in;
	edge_cov = edge_cov_in;
}
*/
SR::SR(Eigen::ArrayXXcf salient_region_fft_in, float cov_in, float edge_cov_in, int frame_num_in) {
	salient_region_fft = salient_region_fft_in;
	cov = cov_in;
	edge_cov = edge_cov_in;
	frame_num = frame_num_in;
}
SR::SR(Eigen::ArrayXXcf salient_region_fft_in, Eigen::ArrayXXcf h_hat_star_in, float cov_in, float edge_cov_in, int frame_num_in) {
	salient_region_fft = salient_region_fft_in;
	h_hat_star = h_hat_star_in;
	cov = cov_in;
	edge_cov = edge_cov_in;
	frame_num = frame_num_in;
}
SR::SR(Eigen::ArrayXXf salient_region_in, float cov_in, float edge_cov_in) {
	salient_region = salient_region_in;
	salient_region_fft = fft(salient_region);
	cov = cov_in;
	edge_cov = edge_cov_in;
}