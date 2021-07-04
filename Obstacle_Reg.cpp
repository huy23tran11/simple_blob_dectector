//! [includes]
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <vector>
#include <map>
#include <iostream>

using namespace cv;
//! [includes]

int main()
{
    std::cout << "Running" << std::endl;
    //! [imread]
    std::string image_path = samples::findFile("ballon.jpeg");
    Mat img = imread(image_path, IMREAD_GRAYSCALE);

    //SimpleBlobDetector detector;
    SimpleBlobDetector::Params params;

    // Change thresholds
    params.minThreshold = 150;
    params.maxThreshold = 255;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);

 
    // Detect blobs.
    std::vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);
     
    // Draw detected blobs as red circles.
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS flag ensures the size of the circle corresponds to the size of blob
    Mat im_with_keypoints;
    drawKeypoints(img, keypoints, im_with_keypoints, Scalar(0,0,255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
     
    // Show blobs
    //std::cout << im_with_keypoints << std::endl;
    imshow("original img", img);
    imshow("keypoints", im_with_keypoints);
    int k = waitKey(0);
    std::cout << "Ending" << std::endl;
    //! [imread]

    return 0;
}