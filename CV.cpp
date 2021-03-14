#include <pch.h>
#include <opencv.hpp>
#include <math.h>
using namespace std;
using namespace cv;

#define PI 3.14159265

void onMouse_first(int event, int x, int y, int flags, void* param);
void onMouse_second(int event, int x, int y, int flags, void* param);
void sobelFunc(Mat image1, Mat image2);

int pointers_1[4][2];
int pointers_2[4][2];
int counter_1 = 0;
int counter_2 = 0;

Mat first;
Mat second;

bool activate = false;

int main() {

	first = imread("1st.jpg", IMREAD_GRAYSCALE);
	second = imread("2nd.jpg", IMREAD_GRAYSCALE);

	if (first.empty())
		return -1;

	if (second.empty())
		return -1;

	resize(first, first, Size(first.cols/8, first.rows/8));
	resize(second, second, Size(second.cols /8, second.rows /8));

	//int ddepth = CV_32F;
	//Mat dstGx, dstGy;
	//Sobel();
	//Mat dstImage(512, 512, CV_8UC3, Scalar(255, 255, 255));
	imshow("first", first);
	setMouseCallback("first", onMouse_first, (void*)&first);

	imshow("second", second);
	setMouseCallback("second", onMouse_second, (void*)&second);

	waitKey();

	return 0;
}

void onMouse_first(int event, int x, int y, int flags, void* param) {
	Mat *pMat = (Mat*)param;
	Mat image = Mat(*pMat);

	int ksize = 15;

	if (event == EVENT_LBUTTONDOWN && counter_1 < 4) {
		//Vec3b color = first.at < Vec3b > (y, x);
		rectangle(image, Point(x - float(ksize) / 2.0, y - float(ksize) / 2.0), Point(x + float(ksize) / 2.0, y + float(ksize) / 2.0), Scalar(0, 0, 255));

		pointers_1[counter_1][0] = x;
		pointers_1[counter_1][1] = y;

		counter_1 += 1;

		imshow("first", image);

		for (int i = 0; i < 4; i++) {
			cout << i << " x1:  " << pointers_1[i][0] << " y1:  " << pointers_1[i][1] << '\n';
		}
		cout << '\n';

		/*if (counter_1 == 4 && counter_2 == 4) {
			activate = true;
			sobelFunc(first, second);
			//activate = false;
		}*/
	}
	
	//imshow("first", image);
}

void onMouse_second(int event, int x, int y, int flags, void* param) {
	Mat *pMat = (Mat*)param;
	Mat image = Mat(*pMat);

	int ksize = 15;

	if (event == EVENT_LBUTTONDOWN && counter_2 < 4) {
		//Vec3b color = first.at < Vec3b > (y, x);
		rectangle(image, Point(x - float(ksize) / 2.0, y - float(ksize) / 2.0), Point(x + float(ksize) / 2.0, y + float(ksize) / 2.0), Scalar(0, 0, 255));

		pointers_2[counter_2][0] = x;
		pointers_2[counter_2][1] = y;

		counter_2 += 1;

		imshow("second", image);

		for (int i = 0; i < 4; i++) {
			cout << i << " x2:  " << pointers_2[i][0] << " y2:  " << pointers_2[i][1] << '\n';
		}
		cout << '\n';

		if (counter_1 == 4 && counter_2 == 4) {
			activate = true;
			sobelFunc(first, second);
			//activate = false;
		}
	}
	
	//sobelFunc(image);
}

void sobelFunc(Mat first, Mat second) {
	//float diff[4][15][15] = { 0, };
	int ksize = 15;

	Mat cropImg_1[4];
	Mat dstMag_1[4];

	Mat cropImg_2[4];
	Mat dstMag_2[4];

	float ori_1[15][15] = { 0, };
	float ori_2[15][15] = { 0, };

	if (activate){
		for (int i = 0; i < 4; i++) {

			Rect boundary(0, 0, first.cols, first.rows);
			Rect roi(pointers_1[i][0] - 7, pointers_1[i][1] - 7, 15, 15);
			cropImg_1[i] = first(roi & boundary);

			int ddepth = CV_32F;
			Mat dstGx, dstGy;
			Sobel(cropImg_1[i], dstGx, ddepth, 1, 0, ksize);
			Sobel(cropImg_1[i], dstGy, ddepth, 1, 0, ksize);

			magnitude(dstGx, dstGy, dstMag_1[i]);
			for (int j = 0; j < 15; j++) {
				for (int k = 0; k < 15; k++) {
					float orientation = atan2(dstGx.at<float>(j,k), dstGy.at<float>(j, k)) * 180 / PI;
					ori_1[j][k] = orientation;
				}
			}
			//ori[i] = hal_ni_fastAtan32f(dstGx, dstGy);
		}

		for (int i = 0; i < 4; i++) {

			Rect boundary(0, 0, second.cols, second.rows);
			Rect roi(pointers_2[i][0] - 7, pointers_2[i][1] - 7, 15, 15);
			cropImg_2[i] = second(roi & boundary);

			int ddepth = CV_32F;
			Mat dstGx, dstGy;
			Sobel(cropImg_2[i], dstGx, ddepth, 1, 0, ksize);
			Sobel(cropImg_2[i], dstGy, ddepth, 1, 0, ksize);

			magnitude(dstGx, dstGy, dstMag_2[i]);
			for (int j = 0; j < 15; j++) {
				for (int k = 0; k < 15; k++) {
					float orientation = atan2(dstGx.at<float>(j, k), dstGy.at<float>(j, k)) * 180 / PI;
					ori_2[j][k] = orientation;
				}
			}
			//orientation = atan2(dstGx, dstGy) * 180 / PI;
		}
		activate = false;
		cout << "dst mag: " << dstMag_2[0].size() << '\n';

		// histogram
		int bin = 8;




	}
	
	//imshow("crop", cropImg[0]);
	//waitKey();
	
	return;
}