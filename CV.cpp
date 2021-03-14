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

	float hist1[4][8] = { 0, };
	float hist2[4][8] = { 0, };

	if (activate){
		for (int i = 0; i < 4; i++) {

			Rect boundary(0, 0, first.cols, first.rows);
			Rect roi(pointers_1[i][0] - 7, pointers_1[i][1] - 7, 15, 15);
			cropImg_1[i] = first(roi & boundary);

			int ddepth = CV_32F;
			Mat dstGx, dstGy;
			Sobel(cropImg_1[i], dstGx, ddepth, 1, 0, ksize);
			Sobel(cropImg_1[i], dstGy, ddepth, 0, 1, ksize);
			//cout << "Gx: " << dstGx << '\n';

			magnitude(dstGx, dstGy, dstMag_1[i]);
			for (int j = 0; j < 15; j++) {
				for (int k = 0; k < 15; k++) {
					float orientation = atan2(dstGx.at<float>(j, k), dstGy.at<float>(j, k)) * 180 / PI;
					if (orientation < 0)
						orientation = 360 + orientation;
					ori_1[j][k] = orientation;
					//cout << "orientation: " << orientation << '\n';

					int index = int(ori_1[j][k]) % 360 / 45.;
					hist1[i][index] += 1; //(0.01*dstMag_1[i].at<float>(j, k));
				}
			}
		}

		for (int i = 0; i < 4; i++) {

			Rect boundary(0, 0, second.cols, second.rows);
			Rect roi(pointers_2[i][0] - 7, pointers_2[i][1] - 7, 15, 15);
			cropImg_2[i] = second(roi & boundary);

			int ddepth = CV_32F;
			Mat dstGx, dstGy;
			Sobel(cropImg_2[i], dstGx, ddepth, 1, 0, ksize);
			Sobel(cropImg_2[i], dstGy, ddepth, 0, 1, ksize);
			//cout << "dstGx: " << dstGx.size() << '\n';

			magnitude(dstGx, dstGy, dstMag_2[i]);
			for (int j = 0; j < 15; j++) {
				for (int k = 0; k < 15; k++) {
					float orientation = atan2(dstGx.at<float>(j, k), dstGy.at<float>(j, k)) * 180 / PI;
					if (orientation < 0)
						orientation = 360 + orientation;
					ori_2[j][k] = orientation;
					//cout << "orientation: " << orientation << '\n';
					int index = int(ori_2[j][k]) % 360 / 45.;
					hist2[i][index] += 1; //(0.01*dstMag_2[i].at<float>(j, k));
				}
			}
		}
		activate = false;
		cout << "dst mag: " << dstMag_2[0].size() << '\n';
		

		// histogram
		Mat histImage(256, 256, CV_8U);
		histImage = Scalar(255);
		int histSize = 8;
		int binW = cvRound((double)histImage.cols / histSize);
		//normalize(hist1[0], hist1[0], 0, 10, NORM_MINMAX);
		string name[8] = { "h11", "h12", "h13","h14",
						   "h21", "h22", "h23","h24" };

		for (int c = 0; c < 4; c++) {
			for (int k = 0; k < 8; k++) {
				int x1 = k * binW;
				int y1 = histImage.rows;
				int x2 = (k + 1) * binW;
				int y2 = histImage.rows - cvRound(hist1[c][k]);

				rectangle(histImage, Point(x1, y1), Point(x2, y2), Scalar(0));
			}
			imshow(name[c], histImage);
			histImage = Scalar(255);
			waitKey();
		}

		for (int c = 0; c < 4; c++) {
			for (int k = 0; k < 8; k++) {
				int x1 = k * binW;
				int y1 = histImage.rows;
				int x2 = (k + 1) * binW;
				int y2 = histImage.rows - cvRound(hist2[c][k]);

				rectangle(histImage, Point(x1, y1), Point(x2, y2), Scalar(0));
			}
			imshow(name[c+4], histImage);
			histImage = Scalar(255);
			waitKey();
		}



	}
	
	return;
}
