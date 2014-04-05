/**
 *  This file is part of ltp-text-detector.
 *  Copyright (C) 2013 Michael Opitz
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <opencv2/ts/ts.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <memory>

#include <detector/Feature.h>
#include <detector/ConfigManager.h>
using namespace Detector;

class FeatureTests : public cvtest::BaseTest
{

public:
    FeatureTests(int hog, int ii, int lbp, int ltp)
    {
        ConfigManager::instance().reset(new ConfigManager());
        ConfigManager::instance()->set_hog_channels(hog);
        ConfigManager::instance()->set_hog_norm_channel_idx(hog);
        ConfigManager::instance()->set_integral_channels(ii+(hog > 0 ? 1 : 0)); // account for gradient magnitude
        ConfigManager::instance()->set_lbp_histograms(lbp);
        ConfigManager::instance()->set_ltp_histograms(ltp);
        ConfigManager::instance()->set_feature_width(10);
        ConfigManager::instance()->set_feature_height(10);

        const int chans = (hog > 0 ? hog + 1 : 0) + ii + lbp * 256 + ltp * 256;
        img = cv::Mat(20,20,CV_32FC(chans),0.0f);
        float *data = img.ptr<float>(0,0);
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
    cv::Mat img;
protected:
    void run(int) {

    }
};

// integral_channels, lbp, ltp
TEST(FeatureTests,compute_hogs) {
    FeatureTests tests(6,0,0,0);

    Feature f1 = Feature(Rect(Point(0,0), Point(9,9)),0, ConfigManager::instance());
    double res1 = f1.compute(tests.img, 0, 0);
    Feature f2 = Feature(Rect(Point(0,0), Point(9,9)),1, ConfigManager::instance());
    double res2 = f2.compute(tests.img, 0, 0);
    Feature f3 = Feature(Rect(Point(2,2), Point(5,5)),1, ConfigManager::instance());
    double res3 = f3.compute(tests.img, 0, 0);
    Feature f4 = Feature(Rect(Point(2,0), Point(5,9)),1, ConfigManager::instance());
    double res4 = f4.compute(tests.img, 0, 0);

    const double eps = 1e-10;
    ASSERT_TRUE(abs(res1 - sqrt((0 + 1*10 + 2*10 + 3*10 + 4*10 + 5*10 + 6*10 + 7*10 + 8*10 + 9*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res2 - sqrt((0 + 1*10*10 + 2*10*10 + 3*10*10 + 4*10*10 + 5*10*10 + 6*10*10 + 7*10*10 + 8*10*10 + 9*10*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res3 - sqrt((2*4*10 + 3*4*10 + 4*4*10 + 5*4*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res4 - sqrt((2*10*10 + 3*10*10 + 4*10*10 + 5*10*10) / (60.0f*100.0f))) < eps);
}

// integral_channels, lbp, ltp
TEST(FeatureTests, compute_hogs_offset) {
    FeatureTests tests(6,0,0,0);

    Feature f1 = Feature(Rect(Point(0,0), Point(9,9)),0, ConfigManager::instance());
    double res1 = f1.compute(tests.img, 5, 5);
    double res2 = f1.compute(tests.img, 5, 6);
    double res3 = f1.compute(tests.img, 6, 5);

    const double eps = 1e-10;
    ASSERT_TRUE(abs(res1 - sqrt((5*10 + 6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res2 - sqrt((5*10 + 6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res3 - sqrt((6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10 + 15*10) / (60.0f*100.0f))) < eps);
}

// integral_channels, lbp, ltp
TEST(FeatureTests, compute_integral_channels) {
    FeatureTests tests(0,2,0,0);

    Feature f1 = Feature(Rect(Point(0,0), Point(9,9)),0, ConfigManager::instance());
    double res1 = f1.compute(tests.img, 0, 0);
    Feature f2 = Feature(Rect(Point(0,0), Point(9,9)),1, ConfigManager::instance());
    double res2 = f2.compute(tests.img, 0, 0);
    Feature f3 = Feature(Rect(Point(0,0), Point(0,0)),1, ConfigManager::instance());
    double res3 = f3.compute(tests.img, 0, 0);

    const double eps = 1e-10;
    ASSERT_TRUE(abs(res1 - (0 + 1*10 + 2*10 + 3*10 + 4*10 + 5*10 + 6*10 + 7*10 + 8*10 + 9*10) / 100.0) < eps);
    ASSERT_TRUE(abs(res2 - (0 + 1*10*10 + 2*10*10 + 3*10*10 + 4*10*10 + 5*10*10 + 6*10*10 + 7*10*10 + 8*10*10 + 9*10*10) / 100.0) < eps);
    ASSERT_TRUE(res3 == 0);
}

TEST(FeatureTests, compute_hogs_channels_with_eigen_matrix) {
    FeatureTests tests(6,0,0,0);
    Matrix mat(1, tests.img.rows * tests.img.cols * tests.img.channels());

    const float *data = tests.img.ptr<float>(0,0);
    for (int c = 0; c < tests.img.channels(); c++) {
        for (int i = 0; i < tests.img.rows; i++) {
            for (int j = 0; j < tests.img.cols; j++) {
                mat(0,i) = *(data + i * tests.img.step[0] / sizeof (float) + j * tests.img.step[1] / sizeof (float) + c);
            }
        }
    }

    Feature f1 = Feature(Rect(Point(0,0), Point(9,9)),0, ConfigManager::instance());
    double res1 = f1.compute(mat);
    Feature f2 = Feature(Rect(Point(0,0), Point(9,9)),1, ConfigManager::instance());
    double res2 = f2.compute(mat);
    Feature f3 = Feature(Rect(Point(2,2), Point(5,5)),1, ConfigManager::instance());
    double res3 = f3.compute(mat);
    Feature f4 = Feature(Rect(Point(2,0), Point(5,9)),1, ConfigManager::instance());
    double res4 = f4.compute(mat);

    const double eps = 1e-10;
    ASSERT_TRUE(abs(res1 - sqrt((0 + 1*10 + 2*10 + 3*10 + 4*10 + 5*10 + 6*10 + 7*10 + 8*10 + 9*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res2 - sqrt((0 + 1*10*10 + 2*10*10 + 3*10*10 + 4*10*10 + 5*10*10 + 6*10*10 + 7*10*10 + 8*10*10 + 9*10*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res3 - sqrt((2*4*10 + 3*4*10 + 4*4*10 + 5*4*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res4 - sqrt((2*10*10 + 3*10*10 + 4*10*10 + 5*10*10) / (60.0f*100.0f))) < eps);
}
