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

#include <detector/Tree.h>
#include <detector/Node.h>
#include <detector/ConfigManager.h>
#include <detector/IntegralFeature.h>

using namespace Detector;

class OptimizeTreeTest : public cvtest::BaseTest
{
public:

private:
};

static 
std::vector<std::shared_ptr<BaseFeature> > get_features()
{
    std::vector<std::shared_ptr<BaseFeature> > features;
    for (int i = 0; i < 2; i++) {
        features.push_back(std::make_shared<IntegralFeature>(0, Rect(Point(0,0), Point(0,0)), 10, 10));
        features.back()->set_id(i);
    }
    return features;
}

TEST(OptimizeTreeTest, optimize_simple)
{
    Matrix precomp(4, 2);
    Matrix data;
    Matrix labels(4,1);

    for (int i = 0; i < 4; i++) {
        if (i < 2) {
            precomp(i,0) = 1.0;
            labels(i,0)  = 1.0;
        } else {
            precomp(i,1) = 0.5;
            labels(i,0)  = -1.0;
        }
        precomp(i,1) = 0.0;
    }

    std::vector<int> all_samples;
    for (int i = 0; i < 4; i++) {
        all_samples.push_back(i);
    }
    std::vector<std::shared_ptr<BaseFeature> > features(get_features());


    std::vector<double> weights(4, 1.0);
    weights[0] = 0.99;
    weights[1] = 0.99;
    weights[2] = 1e-5;
    weights[3] = 1e-5;
    Tree tree(0, 2, 1);
    std::vector<int> errors(labels.rows(), 0);
    tree.train(
        errors, labels, all_samples, features, 
        precomp, 
        precomp.colwise().minCoeff(),
        precomp.colwise().maxCoeff(),
        weights);

    std::shared_ptr<Node> n1(tree.root()->left());
    std::shared_ptr<Node> n2(tree.root()->right());

    int npos = 0;
    npos += n1->fraction_pos() > 0;
    npos += n2->fraction_pos() > 0;

    ASSERT_TRUE(n1->is_leaf());
    ASSERT_TRUE(n2->is_leaf());

    ASSERT_TRUE(tree.predict_precomputed(precomp.row(0)) >= 0);
    ASSERT_TRUE(tree.predict_precomputed(precomp.row(1)) >= 0);
    ASSERT_TRUE(tree.predict_precomputed(precomp.row(2)) < 0);
    ASSERT_TRUE(tree.predict_precomputed(precomp.row(3)) < 0);
    ASSERT_TRUE(npos == 1);
}

TEST(OptimizeTreeTest, optimize_depth_2)
{
    int n = 4;
    Matrix precomp(n, 2);
    Matrix data;
    Matrix labels(n,1);

    for (int i = 0; i < n; i++) {
        if (i % 2 == 0) {
            labels(i,0)  = 1.0;
        } else {
            labels(i,0)  = -1.0;
        }
        precomp(i,0) = i / 4.0;
        precomp(i,1) = 0.0f;
    }

    std::vector<int> all_samples;
    for (int i = 0; i < n; i++) {
        all_samples.push_back(i);
    }
    std::vector<std::shared_ptr<BaseFeature> > features(get_features());

    std::vector<double> weights(n, 1.0);
    weights[0] = 1e-5;
    weights[1] = 100;
    weights[2] = 100;
    weights[3] = 1e-5;
    Tree tree(0, 2, 1);
    std::vector<int> errors(labels.rows(), 0);
    tree.train(
        errors,
        labels, all_samples, features, 
        precomp, precomp.colwise().minCoeff(), 
        precomp.colwise().maxCoeff(), weights);

    std::shared_ptr<Node> n1(tree.root()->left()->left());
    std::shared_ptr<Node> n2(tree.root()->left()->right());
    std::shared_ptr<Node> n3(tree.root()->right()->left());
    std::shared_ptr<Node> n4(tree.root()->right()->right());

    int npos = 0;
    npos += n1->fraction_pos() > 0;
    npos += n2->fraction_pos() > 0;
    npos += n3->fraction_pos() > 0;
    npos += n4->fraction_pos() > 0;

    ASSERT_TRUE(n1->is_leaf());
    ASSERT_TRUE(n2->is_leaf());
    ASSERT_TRUE(n3->is_leaf());
    ASSERT_TRUE(n4->is_leaf());

    ASSERT_TRUE(tree.predict_precomputed(precomp.row(0)) >= 0);
    ASSERT_TRUE(tree.predict_precomputed(precomp.row(1)) < 0);
    ASSERT_TRUE(tree.predict_precomputed(precomp.row(2)) >= 0);
    ASSERT_TRUE(tree.predict_precomputed(precomp.row(3)) < 0);
    //ASSERT_TRUE(npos == 2);
}
