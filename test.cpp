#include <opencv2/opencv.hpp>
#include "LFFD.h"
#include <chrono>

using namespace cv;

int main(int argc, char** argv) {

	if (argc !=3)
	{
		std::cout << " .exe mode_path image_file" << std::endl;
		return -1;
	}

	std::string model_path = argv[1];
	std::string image_file = argv[2];

	LFFD lffd(model_path);
	cv::Mat frame = cv::imread(image_file);
	std::vector<FaceInfo> face_info;

	ncnn::Mat inmat = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR, frame.cols, frame.rows);
	lffd.detect(inmat, face_info, 240, 320);

	for (int i = 0; i < face_info.size(); i++)
	{
		auto face = face_info[i];
		cv::Point pt1(face.x1, face.y1);
		cv::Point pt2(face.x2, face.y2);
		cv::rectangle(frame, pt1, pt2, cv::Scalar(0, 255, 0), 2);
	}

	cv::namedWindow("lffd", CV_WINDOW_NORMAL);
	cv::imshow("lffd", frame);
	cv::waitKey();
	return 0;
}