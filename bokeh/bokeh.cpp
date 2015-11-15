#define _USE_MATH_DEFINES
#include <cstdio>
#include <fstream>
#include <cmath>
#include <iostream>
#include <sstream>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>

using namespace std;
using namespace cv;

Mat1f meanfilter(Mat1f image, Mat1f result)
{
	int k = 4;
//	int total = (2 * k + 1)*(2 * k + 1);

	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			float temp = 0;
			float total = 0.0f;
			for (int u = -k; u <= k; u++)
			{
				for (int v = -k; v <= k; v++)
				{
					if (x + v >= image.cols || x + v < 0 || y + u >= image.rows || y + u < 0)
						continue;
					if (u*u + v*v <= k*k) {
						temp += image.at<float>(y + u, x + v);
						total += 1.2f;
					}
				}
			}
			result.at<float>(y, x) = (float)(temp / total);
		}
	}

	imshow("bright points (mean)", result);

	return result;

}

int main()
{
	Mat3f inputimg = imread("statue_img.png") / 255;
	cv::resize(inputimg.clone(), inputimg, cv::Size(inputimg.cols / 2, inputimg.rows / 2));
	//	imshow("input image", inputimg);

	Mat1f img = imread("statue_img.png", CV_LOAD_IMAGE_GRAYSCALE) / 255.0;
	//imshow("test", img);
	cv::resize(img.clone(), img, cv::Size(img.cols / 2, img.rows / 2));
	cv::imshow("input", inputimg);

	double maxval;
	double minval;
	minMaxLoc(img, &minval, &maxval);
	cout << minval << " " << maxval << endl;

	Mat1f image(inputimg.rows, inputimg.cols);
	image.setTo(Scalar(0));

	// 밝기가 밝은 point만 추출해서 image에 저장
	for (int y = 0; y < img.rows; y++)
	{
		for (int x = 0; x < img.cols; x++)
		{
			if (img.at<float>(y, x) > 0.985)
			{
			//	image.at<float>(y, x) = - (img.at<float>(y, x) * log(1 - img.at<float>(y, x)));
				image.at<float>(y, x) = img.at<float>(y, x);
			}
		}
	}
	imshow("bright points", image);

	Mat1f bright_image = image.clone();
	bright_image.setTo(Scalar(0));

	/*
	// input image에서 밝은 point들 빼주기
	for (int y = 0; y < inputimg.rows; y++)
	{
		for (int x = 0; x < inputimg.cols; x++)
		{
			inputimg.at<Vec3f>(y, x) = inputimg.at<Vec3f>(y, x) - cv::Vec3f(image.at<float>(y, x), image.at<float>(y, x), image.at<float>(y, x));
		}
	}
	*/

	// 밝은 point들에 mean filtering 적용해서 bokeh 만들기
	Mat1f result = meanfilter(image, bright_image);

	// input image에 가우시안 블러 입히기 (depth가 있다고 가정)
	GaussianBlur(inputimg, inputimg, Size(21, 21), 0, 0);

	// input image에 bokeh 입혀주기
	for (int y = 0; y < inputimg.rows; y++)
	{
		for (int x = 0; x < inputimg.cols; x++)
		{
			inputimg.at<Vec3f>(y, x) = inputimg.at<Vec3f>(y, x) + cv::Vec3f(result.at<float>(y, x), result.at<float>(y, x), result.at<float>(y, x));
		}
	}

	imshow("result", inputimg);
//	imwrite("not_subtract_source.png", inputimg * 255.0);
	imwrite("subtract_source.png", inputimg * 255.0);
	waitKey(0);

	return 0;

}
