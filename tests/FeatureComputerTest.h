#ifndef FEATURECOMPUTERTEST_H

#define FEATURECOMPUTERTEST_H

#include <opencv2/core/core.hpp>
#include <opencv2/ts/ts.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class FeatureComputerTest : public cvtest::BaseTest 
{
    public:
    cv::Mat img;
    FeatureComputerTest(int hog, int ii, int lbp, int ltp)
    {
        const int chans = (hog > 0 ? hog + 1 : 0) + ii + lbp * 256 + ltp * 256;
        img = cv::Mat(20,20,CV_32FC(chans),0.0f);
        float *data = img.ptr<float>(0,0);
        // forall hog channels:
        //
        // | 1 * 10^cid  |  1 * 10^cid | 1 * 10^cid  | 1 * 10^cid...
        // | 2 * 10^cid  |  2 * 10^cid | 2 * 10^cid  | 2 * 10^cid...
        // | 3 * 10^cid  |  3 * 10^cid | 3 * 10^cid  | 3 * 10^cid....
        // | ...
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                for (int c = 0; c < chans; c++) {
                    if (hog > 0 && c == hog) {
                        // hog norm
                        *(data + i * img.step[0] / sizeof (float) + j * img.step[1] / sizeof (float) + c) = 1 * pow(10, c);
                    } else {
                        *(data + i * img.step[0] / sizeof (float) + j * img.step[1] / sizeof (float) + c) = i * pow(10, c);
                    }
                }
            }
        }
        cv::integral(img, img, CV_64F);
    }
};

#endif /* end of include guard: FEATURECOMPUTERTEST_H */

