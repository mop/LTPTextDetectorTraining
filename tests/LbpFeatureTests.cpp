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
#include "FeatureComputerTest.h"

#include <detector/LbpFeature.h>

using namespace Detector;

class LbpFeatureTest : public FeatureComputerTest
{
    public:
    LbpFeatureTest(int hog, int ii, int lbp, int ltp): FeatureComputerTest(hog, ii, lbp, ltp) {}
        // forall channels: 
        // | 1 * 10^cid  |  1 * 10^cid | 1 * 10^cid  | 1 * 10^cid...
        // | 2 * 10^cid  |  2 * 10^cid | 2 * 10^cid  | 2 * 10^cid...
        // | 3 * 10^cid  |  3 * 10^cid | 3 * 10^cid  | 3 * 10^cid....
        // | ...
};

TEST(LbpFeatureTest, test_counts_lbps_correctly) 
{
    LbpFeatureTest test(6, 1, 1, 0);
    const int lbp_start = 9;
    const int lbp_end   = 10;
    const int window_width  = 20;
    const int window_height = 20;

    LbpFeature f = LbpFeature(10, Rect(Point(0,0), Point(9,9)), lbp_start, lbp_end, window_width, window_height);

    const double res1 = f.compute(test.img, 0, 0);
    const double res2 = f.compute(test.img, 1, 0);
    const double res3 = f.compute(test.img, 0, 1);
    const double eps = 1e-10;

    ASSERT_TRUE(abs(res1 - 10.0 / (20*20)) < eps);
    ASSERT_TRUE(abs(res2 - 0) < eps);
    ASSERT_TRUE(abs(res3 - 10.0 / (20*20)) < eps);
}

