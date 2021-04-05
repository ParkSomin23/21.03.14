//http://www.gisdeveloper.co.kr/?p=6922
#include <opencv.hpp>
#include <math.h>
#include "opencv2/nonfree/nonfree.hpp"
using namespace std;
using namespace cv;

Mat drawlines(Mat img1, Mat img2, Mat lines, vector<Point2f>pts1, vector<Point2f> pts2);
Mat FMat(Mat img, vector<Point2f> kp1, vector<Point2f> kp2);
Mat computeEpilines(Mat F, vector<Point2f>kp, int mode);

int main() {
	Mat img1 = imread("epip1.jpg", IMREAD_GRAYSCALE);
	Mat img2 = imread("epip2.jpg", IMREAD_GRAYSCALE);

	if (img1.empty() || img2.empty())
		return -1;

	//resize(img1, img1, Size(img1.cols / 2, img1.rows / 2));
	//resize(img2, img2, Size(img2.cols / 2, img2.rows / 2));

	GaussianBlur(img1, img1, Size(5, 5), 0.0);
	GaussianBlur(img2, img2, Size(5, 5), 0.0);

	vector<KeyPoint> keypoints1;
	Mat descriptors1;

	vector<KeyPoint> keypoints2;
	Mat descriptors2;

	SIFT siftF(500, 3);
	siftF.detect(img1, keypoints1);
	siftF.detect(img2, keypoints2);

	siftF.compute(img1, keypoints1, descriptors1);
	siftF.compute(img2, keypoints2, descriptors2);

	vector<vector<DMatch>> matches;
	FlannBasedMatcher matcher;
	int k = 2;
	matcher.knnMatch(descriptors1, descriptors2, matches, k);

	vector<DMatch> goodMatches;
	float nndrRatio = 0.4f;
	for (int i = 0; i < matches.size(); i++) {
		if (matches.at(i).size() == 2 &&
			matches.at(i).at(0).distance <= nndrRatio * matches.at(i).at(1).distance)
			goodMatches.push_back(matches[i][0]);
	}

	cout << "goodMatch size: " << goodMatches.size() << endl;

	if (goodMatches.size() < 8)
		return 0;

	Mat imgMatches;
	drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches);
	imshow("goodMAtches", imgMatches);
	
	vector<Point2f> kp1;
	vector<Point2f> kp2;
	for (int i = 0; i < goodMatches.size(); i++) {
		kp1.push_back(keypoints1[goodMatches[i].queryIdx].pt);
		kp2.push_back(keypoints2[goodMatches[i].trainIdx].pt);
	}

	//epip
	Mat fundamental_matrix = findFundamentalMat(kp1, kp2, FM_8POINT);//, 3, 0.99); //FM_RANSAC
	cout << "Fundamental_matrix" << endl;
	cout << fundamental_matrix << endl;

	Mat l;
	l = FMat(img1, kp1, kp2);
	cout << "my F matrix" << endl;
	cout << l << endl;

	Mat lines1, lines2;
	//computeCorrespondEpilines(kp2, 2, fundamental_matrix, lines1);
	//computeCorrespondEpilines(kp1, 1, fundamental_matrix, lines2);
	lines1 = computeEpilines(fundamental_matrix, kp2, 2);
	lines2 = computeEpilines(fundamental_matrix, kp1, 1);

	Mat img3 = drawlines(img1, img2, lines1, kp1, kp2);
	Mat img4 = drawlines(img2, img1, lines2, kp2, kp1);

	imshow("1st", img3);
	imshow("2nd", img4);
	waitKey(0);

	return 0;
}

vector<Point2f> NormPoints(vector<Point2f> kp, Mat T) {
	float x_mean = 0;
 	float y_mean = 0;

	int size = kp.size();
	for (int i = 0; i < size; i++) {
		x_mean += kp[i].x;
		y_mean += kp[i].y;
	}

	x_mean /= float(size);
    y_mean /= float(size);


	Mat newKp(3, size, CV_32F);
	float meanDist = 0;
	for (int i = 0; i < kp.size(); i++) {
		float new_x = kp[i].x - x_mean;
		float new_y = kp[i].y - y_mean;

		newKp.at<float>(0, i) = new_x;
		newKp.at<float>(1, i) = new_y;
		newKp.at<float>(2, i) = 1;
		//push_back(Point3f(new_x, new_y, 1));

		meanDist += sqrt(pow(new_x, 2) + pow(new_y, 2));
	}

	meanDist /= float(size);
	float scale = sqrt(2) / meanDist;

	T.at<float>(0, 0) = scale;
	T.at<float>(0, 2) = -scale * x_mean;

	T.at<float>(1, 1) = scale;
	T.at<float>(1, 2) = -scale * y_mean;

	T.at<float>(2, 2) = 1;

//	cout << " T " << endl << T.size() << endl << T << endl << endl;
	//cout << "newKp" << endl << newKp.size() << endl;
	//cout << newKp.col(0) << endl;
	
	vector<Point2f> newpt;
	for (int i = 0; i < size; i++) {
		Mat pts = T * newKp.col(i);
		//cout << "newkp.col" << newKp.col(i).size() << endl;
		//cout << pts << endl << pts.size() << endl << endl;
		newpt.push_back(Point(pts.at<float>(0, 0), pts.at<float>(1, 0)));
	}

	return newpt;
}

Mat FMat(Mat img, vector<Point2f> keypoint1, vector<Point2f> keypoint2) { //kp1, kp2, FM_RANSAC, 3, 0.99
	/*
	F = (K'.T)^-1 * E * K^-1
	K : 카메라 파라미터 행렬
	E : Essential Matrix
	(p'.T) E p = 0
	*/

	//Normalize keypoint
	/*float w = float(img.cols);
	float h = float(img.rows);

	Mat T = (Mat_<float>(3, 3) << 2./w, 0, -1, 0, 2./h, -1, 0, 0, 1);
	*/

	Mat T1 = Mat::zeros(3, 3, CV_32F);
	Mat T2 = Mat::zeros(3, 3, CV_32F);

  	vector<Point2f> kp1 = NormPoints(keypoint1, T1);
	vector<Point2f> kp2 = NormPoints(keypoint2, T2);

	int size = kp1.size();
	Mat A(size, 9, CV_32F);
	for (int i = 0; i < kp1.size(); i++) {
		A.at<float>(i, 0) = kp1[i].x * kp2[i].x;
		A.at<float>(i, 1) = kp1[i].y * kp2[i].x;
		A.at<float>(i, 2) = kp2[i].x;

		A.at<float>(i, 3) = kp1[i].x * kp2[i].y;
		A.at<float>(i, 4) = kp1[i].x * kp2[i].y;
		A.at<float>(i, 5) = kp2[i].y;

		A.at<float>(i, 6) = kp1[i].x;
		A.at<float>(i, 7) = kp2[i].y;
		A.at<float>(i, 8) = 1;
	}

	SVD svd(A, SVD::FULL_UV);

	Mat F; 
	transpose(svd.vt, F);

	F = F.col(8);
	//F =  F.reshape(3, 3);
	
	Mat F2(3,3, CV_32F);
	F2.at<float>(0, 0) = F.at<float>(0, 0);
	F2.at<float>(0, 1) = F.at<float>(1, 0);
	F2.at<float>(0, 2) = F.at<float>(2, 0);
	F2.at<float>(1, 0) = F.at<float>(3, 0);
	F2.at<float>(1, 1) = F.at<float>(4, 0);
	F2.at<float>(1, 2) = F.at<float>(5, 0);
	F2.at<float>(2, 0) = F.at<float>(6, 0);
	F2.at<float>(2, 1) = F.at<float>(7, 0);
	F2.at<float>(2, 2) = F.at<float>(8, 0);


	SVD svdF(F2, SVD::FULL_UV);

	Mat d = Mat::zeros(3, 3, CV_32F);
	d.at<float>(0, 0) = svdF.w.at<float>(0);
	d.at<float>(1, 1) = svdF.w.at<float>(1);
	d.at<float>(2, 2) = 0.0;

	Mat fin = svdF.u * d * svdF.vt;

	transpose(T2, T2);
	fin = T2 * fin * T1;

	float F33 = fin.at<float>(2, 2);
	fin /= F33;
	//fin.at<float>(2, 2) = 1;

	return fin;

}


Mat drawlines(Mat img1, Mat img2, Mat lines, vector<Point2f>pts1, vector<Point2f> pts2) {

	Mat newImg(img1.cols, img1.rows, CV_32F);
	newImg = img1.clone();

	//cout << lines.at<float>(0, 0) << " " << lines.at<float>(0, 1) << " " << lines.at<float>(0, 2) << endl;
	for (int i = 0; i < lines.rows; i++) {
		int x0 = 0;
		int y0 = int(-lines.at<float>(i, 2) / lines.at<float>(i, 1));
		int x1 = int(img1.cols);
		int y1 = int(-(lines.at<float>(i, 2) + lines.at<float>(i, 0) * img1.cols) / lines.at<float>(i, 1));

		//line(dst, Point(pointers_1[i][0], pointers_1[i][1]), Point(pointers_2[idx][0] + first.cols, pointers_2[idx][1]), (255, 0, 0), 2);
		line(newImg, Point(x0, y0), Point(x1, y1), (0.255, 3), 1);
		circle(newImg, pts1[i], 5, (0.255, 3), 1);
		circle(newImg, pts2[i], 5, (0.255, 3), 1);
	}

	return newImg;
}

Mat computeEpilines(Mat F, vector<Point2f>kp, int mode) {
	/*
	l2 = F * x1
	l1 = F.T * x2

	ax + by + c = 0 (a, b, c)
	a^2 + b^2 =1 (normalized)
	*/

	F.convertTo(F, CV_32F);

	int size = kp.size();
	Mat line2(size, 3, CV_32F);
	for (int i = 0; i < size; i++) {
		
		Mat pt(3, 1, CV_32F);
		pt.at<float>(0, 0) = kp[i].x;
		pt.at<float>(1, 0) = kp[i].y;
		pt.at<float>(2, 0) = 1;

		Mat tmp;
		Mat Ft;
		if(mode == 1)
			tmp = F * pt;
		else {
			transpose(F, Ft);
			tmp = Ft * pt;
		}
		
		float a = tmp.at<float>(0, 0);
		float b = tmp.at<float>(1, 0);

		float t = sqrt(pow(a, 2) + pow(b, 2));

		line2.at<float>(i, 0) = a / t;
		line2.at<float>(i, 1) = b / t;
		line2.at<float>(i, 2) = tmp.at<float>(2, 0) / t;
	}

	return line2;
}