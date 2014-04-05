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
#include <fstream>
#include <random>

#include <detector/Adaboost.h>
#include <detector/Tree.h>
#include <detector/ConfigManager.h>
#include "util.h"

using namespace Detector;

class DollarRandomTest : public cvtest::BaseTest
{
};

TEST(DollarRandomTest, check_accuracy)
{
    Matrix train_data = readMatrix("tests/train_unittest.csv.gz");
    Matrix test_data = readMatrix("tests/test_unittest.csv.gz");

    std::vector<std::shared_ptr<BaseFeature> > features;
    ConfigManager::instance().reset(new ConfigManager());
    std::shared_ptr<ConfigManager> mgr(ConfigManager::instance());
    for (int i = 0; i < train_data.cols() - 1; i++) {
        features.push_back(
            std::make_shared<IntegralFeature>(
                0, Rect(Point(0,0), Point(0,0)), 
                mgr->feature_width(),
                mgr->feature_height()));
        features.back()->set_id(i);
    }

    Matrix train_y = train_data.topLeftCorner(train_data.rows(), 1);
    Matrix train_x = train_data.topRightCorner(train_data.rows(), train_data.cols() - 1);

    Adaboost boost(Adaboost::GENTLE, 256, 2);
    boost.set_features(features);
    boost.train_precomputed(train_x, train_y);

    Matrix test_x = test_data.topRightCorner(test_data.rows(), test_data.cols() - 1);
    float errs = 0.0f;
    for (int j = 0; j < test_data.rows(); j++) {
        float label = boost.predict_precomputed(test_x.row(j));
        if (label * test_data(j,0) < 0) {
            errs += 1.0f;
        }
    }
    std::cout << "TEST ERROR: " << errs / test_x.rows() << std::endl;
    ASSERT_TRUE(errs / test_x.rows() < 0.015);
}

TEST(DollarRandomTest, check_accuracy_non_uniform)
{
    Matrix train_data_input = readMatrix("tests/train_unittest.csv.gz");
    Matrix test_data = readMatrix("tests/test_unittest.csv.gz");
    Matrix train_data = train_data_input.bottomLeftCorner(3001, train_data_input.cols());

    std::vector<std::shared_ptr<BaseFeature> > features;
    std::shared_ptr<ConfigManager> mgr(ConfigManager::instance());
    for (int i = 0; i < train_data.cols() - 1; i++) {
        features.emplace_back(
            std::make_shared<IntegralFeature>(0, 
                Rect(Point(0,0), Point(0,0)), 
                mgr->feature_width(), mgr->feature_height()));
        features.back()->set_id(i);
    }

    Matrix train_y = train_data.topLeftCorner(train_data.rows(), 1);
    Matrix train_x = train_data.topRightCorner(train_data.rows(), train_data.cols() - 1);

    Adaboost boost(Adaboost::GENTLE, 256, 2);
    boost.set_features(features);
    boost.train_precomputed(train_x, train_y);

    Matrix test_x = test_data.topRightCorner(test_data.rows(), test_data.cols() - 1);
    float errs = 0.0f;
    for (int j = 0; j < test_data.rows(); j++) {
        float label = boost.predict_precomputed(test_x.row(j));
        if (label * test_data(j,0) < 0) {
            errs += 1.0f;
        }
    }
    std::cout << "TEST ERROR: " << errs / test_x.rows() << std::endl;
    ASSERT_TRUE(errs / test_x.rows() < 0.0523);
}
