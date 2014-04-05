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

#include <detector/HogFeature.h>
#include <detector/ConfigManager.h>
#include "FeatureComputerTest.h"
using namespace Detector;

class HogFeatureTests : public FeatureComputerTest
{

public:
    HogFeatureTests(int hog, int ii, int lbp, int ltp)
    : FeatureComputerTest(hog, ii, lbp, ltp) {}
protected:
    void run(int) {

    }
};

// integral_channels, lbp, ltp
TEST(HogFeatureTests,compute_hogs) {
    // make me a 6 channel test
    HogFeatureTests tests(6,0,0,0);
    const int hog_start = 0;
    const int hog_end   = 6;
    const int window_width  = 20;
    const int window_height = 20;
    const int norm_channel = 6;
    const bool l2_norm = false;

    std::shared_ptr<ConfigManager> mgr(ConfigManager::instance());

    HogFeature f1 = HogFeature(0, Rect(Point(0,0), Point(9,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res1 = f1.compute(tests.img, 0, 0);
    HogFeature f2 = HogFeature(1, Rect(Point(0,0), Point(9,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res2 = f2.compute(tests.img, 0, 0);
    HogFeature f3 = HogFeature(1, Rect(Point(2,2), Point(5,5)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res3 = f3.compute(tests.img, 0, 0);
    HogFeature f4 = HogFeature(1, Rect(Point(2,0), Point(5,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res4 = f4.compute(tests.img, 0, 0);

    const double eps = 1e-10;
    ASSERT_TRUE(abs(res1 - sqrt((0 + 1*10 + 2*10 + 3*10 + 4*10 + 5*10 + 6*10 + 7*10 + 8*10 + 9*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res2 - sqrt((0 + 1*10*10 + 2*10*10 + 3*10*10 + 4*10*10 + 5*10*10 + 6*10*10 + 7*10*10 + 8*10*10 + 9*10*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res3 - sqrt((2*4*10 + 3*4*10 + 4*4*10 + 5*4*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res4 - sqrt((2*10*10 + 3*10*10 + 4*10*10 + 5*10*10) / (60.0f*100.0f))) < eps);
}

// integral_channels, lbp, ltp
TEST(HogFeatureTests, compute_hogs_offset) {
    HogFeatureTests tests(6,0,0,0);
    const int hog_start = 0;
    const int hog_end   = 6;
    const int window_width  = 20;
    const int window_height = 20;
    const int norm_channel = 6;
    const bool l2_norm = false;

    HogFeature f1 = HogFeature(0, Rect(Point(0,0), Point(9,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    //Feature f1 = Feature(Rect(Point(0,0), Point(9,9)),0, ConfigManager::instance());
    double res1 = f1.compute(tests.img, 5, 5);
    double res2 = f1.compute(tests.img, 5, 6);
    double res3 = f1.compute(tests.img, 6, 5);

    const double eps = 1e-10;
    ASSERT_TRUE(abs(res1 - sqrt((5*10 + 6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res2 - sqrt((6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10 + 15*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res3 - sqrt((5*10 + 6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10) / (60.0f*100.0f))) < eps);
}

TEST(HogFeatureTests, compute_without_norm) {
    HogFeatureTests tests(6,0,0,0);
    const int hog_start = 0;
    const int hog_end   = 6;
    const int window_width  = 20;
    const int window_height = 20;
    const int norm_channel = -1;
    const bool l2_norm = false;

    HogFeature f1 = HogFeature(0, Rect(Point(0,0), Point(9,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    //Feature f1 = Feature(Rect(Point(0,0), Point(9,9)),0, ConfigManager::instance());
    double res1 = f1.compute(tests.img, 5, 5);
    double res2 = f1.compute(tests.img, 5, 6);
    double res3 = f1.compute(tests.img, 6, 5);

    const double eps = 1e-10;
    ASSERT_TRUE(abs(res1 - ((5*10 + 6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10))) < eps);
    ASSERT_TRUE(abs(res2 - ((6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10 + 15*10))) < eps);
    ASSERT_TRUE(abs(res3 - ((5*10 + 6*10 + 7*10 + 8*10 + 9*10 + 10*10 + 11*10 + 12*10 + 13*10 + 14*10))) < eps);
}

inline static float chan_sum1(int c) 
{
    return 5*10*pow(10,c) + 6*10 *pow(10,c) + 7*10 * pow(10,c) + 
        8*10 * pow(10,c) + 9*10 * pow(10,c) + 
        10*10 * pow(10,c)+ 11*10 * pow(10,c)+ 
        12*10 * pow(10,c)+ 13*10 * pow(10,c)+ 14*10 * pow(10,c);
}

inline static float chan_sum2(int c) 
{
    return 6*10*pow(10,c) + 7*10 *pow(10,c) + 8*10 * pow(10,c) + 
        9*10 * pow(10,c) + 10*10 * pow(10,c) + 
        11*10 * pow(10,c)+ 12*10 * pow(10,c)+ 
        13*10 * pow(10,c)+ 14*10 * pow(10,c)+ 15*10 * pow(10,c);
}

TEST(HogFeatureTests, l2_norm) {
    HogFeatureTests tests(6,0,0,0);
    const int hog_start = 0;
    const int hog_end   = 6;
    const int window_width  = 20;
    const int window_height = 20;
    const int norm_channel = -1;
    const bool l2_norm = true;

    HogFeature f1 = HogFeature(0, Rect(Point(0,0), Point(9,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res1 = f1.compute(tests.img, 5, 5);
    double res2 = f1.compute(tests.img, 5, 6);
    double res3 = f1.compute(tests.img, 6, 5);

    const double eps = 1e-10;
    float r1_norm = 0.0f;
    double vals[6];
    for (int i = 0; i < 6; i++) {
        vals[i] = chan_sum1(i);
        r1_norm += vals[i] * vals[i];
    }
    r1_norm = sqrt(r1_norm);
    for (int i = 0; i < 6; i++) {
        vals[i] /= r1_norm;
        vals[i] = std::min(0.2, vals[i]);
    }
    // compute the new norm
    r1_norm = 0.0f;
    for (int i = 0; i < 6; i++) {
        r1_norm += vals[i]*vals[i];
    }
    const float r1_value = vals[0] / sqrt(r1_norm);

    float r2_norm = 0.0f;
    for (int i = 0; i < 6; i++) {
        vals[i] = chan_sum2(i);
        r2_norm += chan_sum2(i)*chan_sum2(i);
    }
    r2_norm = std::min(0.2, sqrt(r2_norm));
    for (int i = 0; i < 6; i++) {
        vals[i] /= r2_norm;
        vals[i] = std::min(0.2, vals[i]);
    }
    r2_norm = 0.0f;
    for (int i = 0; i < 6; i++) {
        r2_norm += vals[i]*vals[i];
    }
    const float r2_value = vals[0] / sqrt(r2_norm);

    ASSERT_TRUE(abs(res1 - r1_value) < eps);
    ASSERT_TRUE(abs(res2 - r2_value) < eps);
    ASSERT_TRUE(abs(res3 - r1_value) < eps);
}

TEST(HogFeatureTests, compute_hogs_channels_with_eigen_matrix) {
    HogFeatureTests tests(6,0,0,0);
    const int hog_start = 0;
    const int hog_end   = 6;
    const int window_width  = 20;
    const int window_height = 20;
    const int norm_channel = 6;
    const bool l2_norm = false;
    Matrix mat(1, tests.img.rows * tests.img.cols * tests.img.channels());

    const float *data = tests.img.ptr<float>(0,0);
    for (int c = 0; c < tests.img.channels(); c++) {
        for (int i = 0; i < tests.img.rows; i++) {
            for (int j = 0; j < tests.img.cols; j++) {
                mat(0,i) = *(data + i * tests.img.step[0] / sizeof (float) + j * tests.img.step[1] / sizeof (float) + c);
            }
        }
    }

    HogFeature f1 = HogFeature(0, Rect(Point(0,0), Point(9,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res1 = f1.compute(mat);
    HogFeature f2 = HogFeature(1, Rect(Point(0,0), Point(9,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res2 = f2.compute(mat);
    HogFeature f3 = HogFeature(1, Rect(Point(2,2), Point(5,5)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res3 = f3.compute(mat);
    HogFeature f4 = HogFeature(1, Rect(Point(2,0), Point(5,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm);
    double res4 = f4.compute(mat);

    const double eps = 1e-10;
    ASSERT_TRUE(abs(res1 - sqrt((0 + 1*10 + 2*10 + 3*10 + 4*10 + 5*10 + 6*10 + 7*10 + 8*10 + 9*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res2 - sqrt((0 + 1*10*10 + 2*10*10 + 3*10*10 + 4*10*10 + 5*10*10 + 6*10*10 + 7*10*10 + 8*10*10 + 9*10*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res3 - sqrt((2*4*10 + 3*4*10 + 4*4*10 + 5*4*10) / (60.0f*100.0f))) < eps);
    ASSERT_TRUE(abs(res4 - sqrt((2*10*10 + 3*10*10 + 4*10*10 + 5*10*10) / (60.0f*100.0f))) < eps);
}
