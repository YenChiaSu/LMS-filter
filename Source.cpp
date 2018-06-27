#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "opencv/opencv2/opencv.hpp"

#pragma comment(lib, "../../lib/x64/opencv_world320.lib")

#define lr 0.00001 // Set a constant learning rate

using namespace cv;
using namespace std;

double compute_gradient(int *h, double *e);
void init_weight_matrix(Mat &weight_matrix);
Mat LMSfilter(Mat half, Mat grey);
double WeightUpdate(double *grad, double *w, double *e, int *h);
void Reconstruct(const Mat &src, Mat &dst, Mat &kernel);

int main() {

	Mat GreyImg = imread("lena.jpg",0);
	Mat HImg = imread("lenah.jpg",0);
	if (!GreyImg.empty() || !HImg.empty()) {
		cout << "image load success!" << endl;
	}
	else{
		cout << "load image fail" << endl;
	}

	Mat normGreyImg, normHImg;

	//find lms filter
	normalize(GreyImg, normGreyImg, 0.0, 1.0, NORM_MINMAX);
	normalize(HImg, normHImg, 0.0, 1.0, NORM_MINMAX);
	Mat Lmsfilter = LMSfilter(normHImg, normGreyImg);
	
	normGreyImg.release();
	normHImg.release();
	//find max and min value in Grey image for normalize
	double min;
	double max;
	minMaxLoc(GreyImg, &min, &max);

	//normalize image in floating image
	GreyImg.convertTo(normGreyImg, CV_32FC1, 1 / (max - min), -min / (max - min));
	GreyImg.convertTo(normHImg, CV_32FC1, 1 / 254.0, -1 / 254.0);

	Mat reconstructImg;
	Reconstruct(normHImg, reconstructImg, Lmsfilter);
	normalize(reconstructImg, reconstructImg, 0, 255, NORM_MINMAX);
	reconstructImg.convertTo(reconstructImg, CV_8UC1);
	imwrite("lenar.jpg", reconstructImg);

	namedWindow("window1");
	imshow("window1", GreyImg);
	imshow("window2", reconstructImg);
	imshow("window3", HImg);
	waitKey(0);

	system("pause");
}

void init_weight_matrix(Mat &weight_matrix) {
	double a = 1 / 9.0, b = 1 / 9.0, c = 1 / 9.0, d = 1 / 9.0, e = 1 / 9.0, f = 1 / 9.0, g = 1 / 9.0, h = 1 / 9.0, i = 1 / 9.0;
	weight_matrix = (Mat_<double>(3, 3) << a, b, c, d, e, f, g, h, i);
	cout << weight_matrix << endl;
}

double compute_gradient(int *h, double *e) {
	return -2 * (*e)*(*h);
}

double WeightUpdate(double *grad, double *w, double *e, int *h) {
	if (*grad < 0) {
		*w = *w + lr * (*e) * (*h);
	}
	else if (*grad > 0){
		*w = *w + lr * (*e) * (*h);
	}
	else if (*grad == 0) {
		return *w;
	}
	return *w;
}

double Weight_Error(Mat &Weight, Mat &LaterWeight) {
	double SUM = 0;

	Mat ErrorMatrix;
	absdiff(Weight, LaterWeight, ErrorMatrix);
	SUM = sum(ErrorMatrix)[0];
	/*for (int i = 0; i < Weight.rows; i++) {
		for (int j = 0; j < Weight.cols; j++) {
			sum = sum + abs(Weight.ptr<double>(i)[j] - LaterWeight.ptr<double>(i)[j]);
		}
	}*/
	return SUM;
}
void Reconstruct(const Mat &src, Mat &dst, Mat &kernel) {
	Point anchor = Point(0, 0);
	filter2D(src, dst, src.depth(), kernel, anchor);
}

Mat LMSfilter(Mat half, Mat grey) {
	Mat weight_matrix;
	Mat Later_Weight;
	double WeightError = 11.0;
	init_weight_matrix(weight_matrix);

	weight_matrix.copyTo(Later_Weight);

	int m = 0, n = 0;
	int sum = 0;
	int epoch = 0;
	int iter;
	while (WeightError > 0.002) {
		iter = 0;
		for (int i = 0; i < half.rows - 2; i++) {
			for (int j = 0; j < half.cols - 2; j++) {

				iter++;
				int g = grey.ptr<uchar>(i)[j];

				int h1 = half.ptr<uchar>(i)[j];
				int h2 = half.ptr<uchar>(i)[j + 1];
				int h3 = half.ptr<uchar>(i)[j + 2];
				int h4 = half.ptr<uchar>(i + 1)[j];
				int h5 = half.ptr<uchar>(i + 1)[j + 1];
				int h6 = half.ptr<uchar>(i + 1)[j + 2];
				int h7 = half.ptr<uchar>(i + 2)[j];
				int h8 = half.ptr<uchar>(i + 2)[j + 1];
				int h9 = half.ptr<uchar>(i + 2)[j + 2];

				double ghat1 = weight_matrix.ptr<double>(m)[n] * h1;
				double ghat2 = weight_matrix.ptr<double>(m)[n + 1] * h2;
				double ghat3 = weight_matrix.ptr<double>(m)[n + 2] * h3;
				double ghat4 = weight_matrix.ptr<double>(m + 1)[n] * h4;
				double ghat5 = weight_matrix.ptr<double>(m + 1)[n + 1] * h5;
				double ghat6 = weight_matrix.ptr<double>(m + 1)[n + 2] * h6;
				double ghat7 = weight_matrix.ptr<double>(m + 2)[n] * h7;
				double ghat8 = weight_matrix.ptr<double>(m + 2)[n + 1] * h8;
				double ghat9 = weight_matrix.ptr<double>(m + 2)[n + 2] * h9;

				double ghat = ghat1 + ghat2 + ghat3 + ghat4 + ghat5 + ghat6 + ghat7 + ghat8 + ghat9;

				if (ghat > 1) { ghat = 1; }
				else if (ghat < 0) { ghat = 0; }
				double e = g - ghat;

				Mat Grad = (Mat_<double>(3, 3) << compute_gradient(&h1, &e), compute_gradient(&h2, &e), compute_gradient(&h3, &e), \
					compute_gradient(&h4, &e), compute_gradient(&h5, &e), compute_gradient(&h6, &e), \
					compute_gradient(&h7, &e), compute_gradient(&h8, &e), compute_gradient(&h9, &e));


				WeightUpdate(&Grad.ptr<double>(0)[0], &weight_matrix.ptr<double>(m)[n], &e, &h1);
				WeightUpdate(&Grad.ptr<double>(0)[1], &weight_matrix.ptr<double>(m)[n + 1], &e, &h2);
				WeightUpdate(&Grad.ptr<double>(0)[2], &weight_matrix.ptr<double>(m)[n + 2], &e, &h3);
				WeightUpdate(&Grad.ptr<double>(1)[0], &weight_matrix.ptr<double>(m + 1)[n], &e, &h4);
				WeightUpdate(&Grad.ptr<double>(1)[1], &weight_matrix.ptr<double>(m + 1)[n + 1], &e, &h5);
				WeightUpdate(&Grad.ptr<double>(1)[2], &weight_matrix.ptr<double>(m + 1)[n + 2], &e, &h6);
				WeightUpdate(&Grad.ptr<double>(2)[0], &weight_matrix.ptr<double>(m + 2)[n], &e, &h7);
				WeightUpdate(&Grad.ptr<double>(2)[1], &weight_matrix.ptr<double>(m + 2)[n + 1], &e, &h8);
				WeightUpdate(&Grad.ptr<double>(2)[2], &weight_matrix.ptr<double>(m + 2)[n + 2], &e, &h9);

				//cout << weight_matrix << endl;
				//cout << "Number of iteration " << iter << endl;
			}
		}
		cout << weight_matrix << endl;
		WeightError = Weight_Error(weight_matrix, Later_Weight);
		cout << WeightError << endl;
		epoch++;
		cout << epoch << endl;
		weight_matrix.copyTo(Later_Weight);
	}
	return weight_matrix;
}

