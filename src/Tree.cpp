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
#include <detector/Tree.h>
#include <detector/Node.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <float.h>
#include <limits.h>
#include <boost/timer/timer.hpp>

#include <parallel/algorithm>

namespace Detector {
static double log2(double n)  {  
	if(n == 0) {
		return(-100000000.0); // hack
	} else {
		return log( n ) / log( 2.0 );
	}
}

static int get_depth_from_id(int id) {
	return int(log2(id+1.1));
}

Tree::Tree(int id, int max_depth, int min_sample_count)
: _id(id),
  _depth(0),
  _max_depth(max_depth),
  _min_sample_count(min_sample_count),
  _root(nullptr),
  _n_nodes(0),
  _n_leaves(0) 
  {}

void Tree::create_child(
    std::shared_ptr<Node> parent,
    ChildType child_type,
    std::vector<int> &errors,
    const Matrix& labels,
    const std::vector<double> &weights,
    const std::vector<int> &samples,
    int label,
    bool recompute) {
	// create (left or right) child node
	int newId = parent->id() * 2 + 1 + int(child_type);
    std::shared_ptr<Node> child(new Node(newId, labels, weights, samples, label, recompute));
	child->set_parent(parent);
	if (child_type == CHILD_LEFT) {
		parent->set_left(child);
	} else {
		parent->set_right(child);
	}
	
	++_n_nodes;
	
    // if this node doesn't have enough examples to produce two children
    // or if it's already at maximum depth, make it a leaf
    if (child->n_samples() < 2*_min_sample_count || get_depth_from_id(newId) >= _max_depth) {
		set_node_to_leaf(child, errors, labels);
	}
}

void Tree::optimize_node(
    std::shared_ptr<Node> node,
    std::vector<int> &errors,
    const Matrix& labels, 
    const Matrix &mins,
    const Matrix &maxs,
    const std::vector<double> &weights) {


	int node_depth = get_depth_from_id(node->id());
	if (node_depth > _depth) { _depth = node_depth; }
	if (_depth == _max_depth) {
        // this is never reached
		set_node_to_leaf(node, errors, labels);
		return;
	} else if (_depth > _max_depth) {
        // this is never reached
		std::cerr << "Error: maxDepth exceeded." << std::endl;
	}
    if (int(node->samples().size()) < _min_sample_count) {
        // this is never reached
        set_node_to_leaf(node, errors, labels);
        return;
    }

    // check if labels are OK
    double num_pos = 0;
    double num_neg = 0;
    for (unsigned int i = 0; i < node->samples().size(); ++i) {
        int idx = node->samples()[i];
        if (labels(idx,0) > 0) {
            num_pos = num_pos + (!weights.empty() ? weights[idx] : 1.0);
        } else {
            num_neg = num_neg + (!weights.empty() ? weights[idx] : 1.0);
        }
    }
    if (num_pos == 0 || num_neg == 0) {
        set_node_to_leaf(node, errors, labels);
        return;
    }

    std::vector<char> mask(_features.size(), 0);
    if (weights.empty()) {
        float frac = sqrt(_features.size()) / float(_features.size());
        //frac = 0.8;
        for (int i = 0; i < int(_features.size() * frac); ++i) {
            mask[i] = true;
        }
        for (size_t i = 0; i < mask.size(); ++i) {
            int j = rand() % mask.size();
            std::swap(mask[i], mask[j]);
        }
    } else {
        mask.assign(mask.size(), 1);
    }

    int best_rect = -1;
    double best_err = FLT_MAX;
    const int nbins = 256;

    std::vector<std::pair<double, int> > best_responses(
        node->samples().size(), std::make_pair(FLT_MAX, -1));

    boost::timer::cpu_timer t;
    t.start();
    const Matrix &fr = *_feature_responses;
    #pragma omp parallel for
    for (size_t r = 0; r < _features.size(); r++) {
        if (!mask[r]) continue;
        float bin_pos[nbins];
        float bin_neg[nbins];
        memset(bin_pos, 0.0, sizeof (float) * nbins);
        memset(bin_neg, 0.0, sizeof (float) * nbins);
        float sum_pos = 0.0;
        float sum_neg = 0.0;

        auto samples = node->samples();
        const float norm = std::max(1e-5f, float(maxs(r) - mins(r)));
        const float feature_min = mins(r);

        for (unsigned int i = 0; i < samples.size(); ++i) {
            const int idx = samples[i];
            const float raw_val = fr(idx,r);
            const float val = (raw_val - feature_min) / (norm);
            int bin = (int) (std::max(0.0f, val) * nbins);
            bin = std::min(bin, nbins-1);

            // negative class
            const float weight = (!weights.empty()) ? weights[idx] : 1.0;
            if (labels(idx,0) <= 0) {
                bin_neg[bin] += weight;
                sum_neg += weight;
            // positive class
            } else {
                bin_pos[bin] += weight;
                sum_pos += weight;
            }
        }
        // find the best bin

        float best_feature_err = FLT_MAX;

        for (int i = 1; i < nbins; i++) {
            bin_pos[i] = bin_pos[i] + bin_pos[i-1];
            bin_neg[i] = bin_neg[i] + bin_neg[i-1];
        }
        std::vector<float> all_errs(nbins, 0.0f);
        for (int i = 0; i < nbins/4; i++) {
            __m128 left_pos  = _mm_load_ps(&bin_pos[i*4]);
            __m128 right_pos = _mm_set_ps1(sum_pos) - _mm_load_ps(&bin_pos[i*4]);
            __m128 left_neg  = _mm_load_ps(&bin_neg[i*4]);
            __m128 right_neg = _mm_set_ps1(sum_neg) - _mm_load_ps(&bin_neg[i*4]);

            __m128 err_classify_left  = _mm_add_ps(right_pos, left_neg);
            __m128 err_classify_right = _mm_add_ps(right_neg, left_pos);
            __m128 errs = _mm_min_ps(err_classify_left, err_classify_right);
            _mm_store_ps(&all_errs[i*4], errs);
        }
        float mi = *std::min_element(all_errs.begin(), all_errs.end());
        if (mi < best_feature_err) {
            best_feature_err = mi;
        }
//        non vectorized code
//        for (int i = 0; i < nbins-1; i++) {
//            num_left_pos += bin_pos[i];
//            num_right_pos -= bin_pos[i];
//            num_right_neg -= bin_neg[i];
//            num_left_neg += bin_neg[i];
//
//            // misclass
//            // if we assign left node a positive class label -> error == num_left_neg + num_left_pos
//            const float err_classify_left = num_right_pos + num_left_neg;
//            // if we assign right node a positive class label -> error == num_left_pos + num_right_neg
//            const float err_classify_right = num_right_neg + num_left_pos;
//            const float my_err = std::min(err_classify_left, err_classify_right);
//
//            if (my_err < best_feature_err) {
//                best_feature_err = my_err;
//            }
//        }

        #pragma omp critical
        if (best_feature_err < best_err) {
            best_err = best_feature_err;
            best_rect = r;
            //best_responses.swap(responses);
        }
    }

    //std::cout << "FOUND DIM IN: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
    t.start();
    auto samples = node->samples();
    best_responses.resize(samples.size());
    #pragma omp parallel for
    for (size_t i = 0; i < samples.size(); i++) {
        const int idx = samples[i];
        best_responses[i] = std::make_pair((*_feature_responses)(idx, best_rect), idx);
    }

    // find the accurate threshold...
    __gnu_parallel::sort(
        best_responses.begin(),
        best_responses.end(), [] (
            const std::pair<double, int> &p1,
            const std::pair<double, int> &p2) -> bool {
        return p1.first < p2.first;
    });
    //std::cout << "SORTED IN: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;
    t.start();


    double best_thresh_err = FLT_MAX;
    double bias = 0.0;
    const float epsilon = 2*FLT_MIN;
    int label_left = 1;

    std::vector<float> cum_pos(best_responses.size(), 0.0f);
    std::vector<float> cum_neg(best_responses.size(), 0.0f);
    #pragma omp parallel for
    for (size_t i = 0; i < best_responses.size(); i++) {
        const int sample_idx = best_responses[i].second;
        const double weight = (!weights.empty()) ? weights[sample_idx] : 1.0;
        if (labels(sample_idx,0) < 0) {
            cum_neg[i] = weight;
        } else {
            cum_pos[i] = weight;
        }
    }
    // do cumsum
    for (size_t i = 1; i < cum_pos.size(); i++) {
        cum_pos[i] = cum_pos[i] + cum_pos[i-1];
        cum_neg[i] = cum_neg[i] + cum_neg[i-1];
    }
    const size_t length = (best_responses.size()) / 4;
    const unsigned int pad = best_responses.size() % 4;
    std::vector<float> all_errs(best_responses.size(), 0.0f);
    std::vector<float> all_labels(best_responses.size(), 0.0f);

    #pragma omp parallel for
    for (unsigned int i = 0; i < length; ++i) {
        __m128 num_left_pos = _mm_load_ps(&cum_pos[i*4]);
        __m128 num_left_neg = _mm_load_ps(&cum_neg[i*4]);
        __m128 num_right_pos = _mm_sub_ps(_mm_set_ps1(num_pos), _mm_load_ps(&cum_pos[i*4]));
        __m128 num_right_neg = _mm_sub_ps(_mm_set_ps1(num_neg), _mm_load_ps(&cum_neg[i*4]));
        __m128 err_classify_left = _mm_add_ps(num_right_pos, num_left_neg);
        __m128 err_classify_right = _mm_add_ps(num_right_neg, num_left_pos);
        __m128 my_err = _mm_min_ps(err_classify_left, err_classify_right);
        __m128 labels = _mm_cmple_ps(err_classify_left, err_classify_right);
        _mm_store_ps(&all_errs[i*4], my_err);
        _mm_store_ps(&all_labels[i*4], labels);
    }

    for (unsigned k = 0; k < pad; k++) {
        const unsigned int i = length*4 + k;

        const float num_left_pos = cum_pos[i];
        const float num_left_neg = cum_neg[i];
        const float num_right_pos = num_pos - cum_pos[i];
        const float num_right_neg = num_neg - cum_neg[i];
        const float err_classify_left = num_right_pos + num_left_neg;
        const float err_classify_right = num_right_neg + num_left_pos;

        all_errs[i] = std::min(err_classify_left, err_classify_right);
        all_labels[i] = err_classify_left <= err_classify_right;
    }

    for (size_t i = 0; i < all_errs.size(); i++) {
        if (all_errs[i] < best_thresh_err && (best_responses[i].first + epsilon) < 
                          best_responses[i+1].first) {
            best_thresh_err = all_errs[i];
            bias = (best_responses[i].first + best_responses[i+1].first) / 2.0;
            label_left = (all_labels[i] > 0)*2-1;
        }
    }
    //std::cout << "FOUND SPLIT IN: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

    //std::cout << "bias: " << bias << " err: " << best_thresh_err << std::endl;
    std::vector<int> bestSamplesL;
    std::vector<int> bestSamplesR;
    bestSamplesL.reserve(labels.size());
    bestSamplesR.reserve(labels.size());

    int left[2] = {0,0};
    int right[2] = {0,0};
    for (unsigned int i = 0; i < samples.size(); ++i) {
        int idx = samples[i];
        double val = (*_feature_responses)(idx, best_rect);

        if (val > bias) {
            right[labels(idx,0)>0]++;
            bestSamplesR.push_back(idx);
        } else {
            left[labels(idx,0)>0]++;
            bestSamplesL.push_back(idx);
        }
    }

    if (bestSamplesL.size() <= 0 || bestSamplesR.size() <= 0) {
        std::cout << num_pos << " " << num_neg << std::endl;
        std::cout << "ERRRRORRRR: " << bestSamplesL.size() << " " << bestSamplesR.size() << std::endl;
        set_node_to_leaf(node, errors, labels);
        return;
    }

    node->set_feature(_features[best_rect]);
    node->set_split_bias(bias);

	create_child(node, CHILD_LEFT, errors, labels, weights, bestSamplesL, label_left, false);
	create_child(node, CHILD_RIGHT, errors, labels, weights, bestSamplesR, -label_left, false);

	if (!node->left()->is_leaf()) {
		optimize_node(node->left(), errors, labels, mins, maxs, weights);
	} 

	if (!node->right()->is_leaf()) {
		optimize_node(node->right(), errors, labels, mins, maxs, weights);
	}
}

void Tree::train(
    std::vector<int> &errors,
    const Matrix &labels,
    const std::vector<int> &bag_samples,
    const std::vector<std::shared_ptr<BaseFeature> > &features,
    const Matrix &feature_responses,
    const Matrix &mins,
    const Matrix &maxs,
    const std::vector<double> &weights) {

    _features = features;
    _feature_responses = &feature_responses;

	_root.reset(new Node(0, labels, weights, bag_samples, 1));
	++_n_nodes;
	optimize_node(_root, errors, labels, mins, maxs, weights);
}

void Tree::set_node_to_leaf(std::shared_ptr<Node> node, std::vector<int> &errors, const Matrix &labels) {
	node->set_to_leaf();
    std::vector<int> samples = node->samples();
    float label = node->fraction_pos();
    #pragma omp parallel for
    for (size_t i = 0; i < samples.size(); i++) {
        const int idx = samples[i];
        if (labels(idx,0) * label < 0) {
            errors[idx] = true;
        }
    }
	++_n_leaves;
}
}

