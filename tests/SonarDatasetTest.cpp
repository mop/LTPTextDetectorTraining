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

class SonarDatasetTest : public cvtest::BaseTest
{
public:

private:
};

TEST(SonarDatasetTest, check_accuracy)
{
    Matrix data = readMatrix("tests/sonar2.csv");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, data.rows()-1);
    float total_errs[10];
    for (int N = 0; N < 10; ++N) {
        for (int i = 0; i < data.rows(); i++) {
            int j = distr(gen);
            if (j == i) continue;
            data.row(i).swap(data.row(j));
        }

        int nfolds = 10;
        int n_test_rows = ceil(float(data.rows()) / float(nfolds));
        std::vector<std::shared_ptr<BaseFeature> > features;
        for (int i = 0; i < data.cols() - 1; i++) {
            features.push_back(std::make_shared<IntegralFeature>(0, Rect(Point(0,0), Point(0,0)), 10, 10));
            features.back()->set_id(i);
        }

        float errs[nfolds];
        for (int i = 0; i < 10; i++) {
            int test_start = i*n_test_rows;
            int test_end = std::min(int(data.rows()), (i+1)*n_test_rows);
            Matrix test(test_end - test_start, data.cols());
            Matrix train(data.rows() - test.rows(), data.cols());

            // copy data
            int train_idx = 0, test_idx = 0;
            for (int j = 0; j < data.rows(); j++) {
                if (j >= test_start && j < test_end) {
                    test.row(test_idx++) = data.row(j);
                } else {
                    train.row(train_idx++) = data.row(j);
                }
            }

            Adaboost boost(Adaboost::GENTLE, 100, 2);
            boost.set_features(features);
            Matrix train_labels = train.topLeftCorner(train.rows(), 1);
            Matrix train_data   = train.topRightCorner(train.rows(), train.cols() - 1);
            boost.train_precomputed(train_data, train_labels);


            errs[i] = 0.0f;
            Matrix test_data = test.topRightCorner(test.rows(), test.cols() - 1);
            for (int j = 0; j < test.rows(); j++) {
                float label = boost.predict_precomputed(test.row(j));
                if (label * test(j,0) < 0) {
                    errs[i] += 1.0f;
                }
            }
            errs[i] /= test.rows();
        }

        float total_err = 0.0f;
        for (int i = 0; i < 10; i++) {
            total_err += errs[i];
        }
        total_errs[N] = total_err / 10.0f;
        std::cout << total_errs[N] << std::endl;
    }
    float avg = 0.0f;
    for (int i = 0; i < 10; i++) {
        avg += total_errs[i];
    }
    ASSERT_TRUE(avg / 10.0f < 0.25f);
    std::cout << "Average error: " << avg / 10.0f << std::endl;
}

TEST(SonarDatasetTest, check_accuracy_tree)
{
    Matrix data = readMatrix("tests/sonar2.csv");

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distr(0, data.rows()-1);
    float total_errs[10];
    for (int N = 0; N < 10; ++N) {
        for (int i = 0; i < data.rows(); i++) {
            int j = distr(gen);
            if (j == i) continue;
            data.row(i).swap(data.row(j));
        }

        int nfolds = 10;
        int n_test_rows = ceil(float(data.rows()) / float(nfolds));
        std::vector<std::shared_ptr<BaseFeature> > features;
        for (int i = 0; i < data.cols() - 1; i++) {
            features.push_back(std::make_shared<IntegralFeature>(0, Rect(Point(0,0), Point(0,0)), 10, 10));
            features.back()->set_id(i);
        }

        float errs[nfolds];
        for (int i = 0; i < 10; i++) {
            int test_start = i*n_test_rows;
            int test_end = std::min(int(data.rows()), (i+1)*n_test_rows);
            Matrix test(test_end - test_start, data.cols());
            Matrix train(data.rows() - test.rows(), data.cols());

            // copy data
            int train_idx = 0, test_idx = 0;
            for (int j = 0; j < data.rows(); j++) {
                if (j >= test_start && j < test_end) {
                    test.row(test_idx++) = data.row(j);
                } else {
                    train.row(train_idx++) = data.row(j);
                }
            }

            //Adaboost boost(Adaboost::GENTLE,1,2);
            //boost.set_features(features);
            Matrix train_labels = train.topLeftCorner(train.rows(), 1);
            Matrix train_data   = train.topRightCorner(train.rows(), train.cols() - 1);
            //boost.train_precomputed(train_data, train_labels);

            Matrix mins = train.colwise().minCoeff();
            Matrix maxs = train.colwise().maxCoeff();
            std::vector<double> weights(train_labels.rows(), 1.0f);
            std::vector<int> all_samples(train_labels.rows(), 0);
            for (int j = 0; j < train_labels.rows(); j++) {
                all_samples[j] = j;
            }
            Tree tree(0, 2, 2);
            std::vector<int> errors(train_labels.rows(), 0);
            tree.train(errors, train_labels, all_samples, features, train_data, mins, maxs, weights);

            errs[i] = 0.0f;
            Matrix test_data = test.topRightCorner(test.rows(), test.cols() - 1);
            for (int j = 0; j < test.rows(); j++) {
                //float label = boost.predict_precomputed(test.row(j));
                float label = tree.predict_precomputed(test.row(j));
                if (label * test(j,0) < 0) {
                    errs[i] += 1.0f;
                }
            }
            errs[i] /= test.rows();
        }

        float total_err = 0.0f;
        for (int i = 0; i < 10; i++) {
            total_err += errs[i];
        }
        total_errs[N] = total_err / 10.0f;
    }
    float avg = 0.0f;
    for (int i = 0; i < 10; i++) {
        avg += total_errs[i];
    }
    ASSERT_TRUE(avg / 10.0f < 0.35f);
    std::cout << "Average error: " << avg / 10.0f << std::endl;
}
