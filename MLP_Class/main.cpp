#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load an image from file (replace path with your actual image path)
    cv::Mat image = cv::imread("../pics/wallhaven-0p7qo3_1920x1080.png");

    // Check if the image was loaded properly
    if (image.empty()) {
        std::cerr << "Failed to load image!" << std::endl;
        return -1;
    }

    // Display the image in a window
    cv::imshow("Sample Image", image);

    // Wait for any key to be pressed
    cv::waitKey(0);

    return 0;
}