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
#ifndef NODE_H
#define NODE_H

#include <vector>
#include <memory>
#include <iostream>

#include <detector/config.h>

#include "DetectorCommon.h"
#include "BaseFeature.h"

#include <boost/archive/xml_oarchive.hpp>
#include <boost/utility.hpp>

#include <opencv2/core/core.hpp>

namespace Detector {

// Node in a multivariate regression tree
class Node;
class Node : public std::enable_shared_from_this<Node>, public boost::noncopyable {
public:
    /**
     *  @param id the id of the node
     *  @param weights are the weights of the node
     *  @param samples are the sample-indices of the node
     *  @param label is the label of the node
     *  @param recompute_impurity indicates if the impurity 
     *         should be recomputed for the node
     */
    Node(
      int id,
      const Matrix& labels,
      const std::vector<double> &weights,
      const std::vector<int>& samples,
      int label,
      bool recompute_impurity=false);
    Node();
    ~Node() = default;

    // returns the node that this feature vector ends up at
    std::shared_ptr<Node> get_terminal_node(const RowVector& features, std::vector<float> &cache, std::vector<bool> &cache_mask);
    std::shared_ptr<Node> get_terminal_node(const cv::Mat& ii, int x, int y, std::vector<float> &cache, std::vector<bool> &cache_mask);
    std::shared_ptr<Node> get_terminal_node(const std::vector<cv::Mat> &ii, int x, int y, std::vector<float> &cache, std::vector<bool> &cache_mask);
    std::shared_ptr<Node> get_terminal_node(const RowVector& features);
    inline std::shared_ptr<Node> get_terminal_node_precomputed(const RowVector& features)
    {
        if (is_leaf_) { return shared_from_this(); }
        // compute the index
        float val = features(feature_->get_id());
        return val <= split_bias_ ? left_->get_terminal_node_precomputed(features) : 
                                    right_->get_terminal_node_precomputed(features);
    }
    // prediction functions
    float predict(const RowVector& features, std::vector<float> &cache, std::vector<bool> &cache_mask);
    float predict(const cv::Mat& ii, int x, int y, std::vector<float> &cache, std::vector<bool> &cache_mask);
    float predict(const std::vector<cv::Mat> &ii, int x, int y, std::vector<float> &cache, std::vector<bool> &cache_mask);
    float predict(const RowVector& features);
    float predict_precomputed(const RowVector& features);

    int id() const { return id_; };
    float fraction_pos() const { return fraction_pos_; }
    int n_samples() const { return n_samples_; };

    // returns index of the sample at given position
    int sample(int pos) const { return samples_[pos]; };
    const std::vector<int>& samples() const { return samples_; };

    bool is_leaf() const { return is_leaf_; };
    void set_to_leaf() { is_leaf_ = true; };
    float get_split_bias() const { return split_bias_; }
    std::shared_ptr<BaseFeature> feature() const { return feature_; }

    std::shared_ptr<Node> left() {
        assert(!is_leaf_);
        return left_;
    };
    
    void set_left(const std::shared_ptr<Node> &left) {
        assert(!is_leaf_);
        left_ = left;
    };

    std::shared_ptr<Node> right() {
        assert(!is_leaf_);
        return right_;
    };

    void set_right(const std::shared_ptr<Node> &right) {
        assert(!is_leaf_);
        right_ = right;
    };

    void set_parent(const std::shared_ptr<Node> &parent) { parent_ = parent; };
    void reset_parents() 
    { 
        if (is_leaf_) return;
        if (left_)
            left_->set_parent(shared_from_this());
        if (right_)
            right_->set_parent(shared_from_this());
    }

    void set_feature(const std::shared_ptr<BaseFeature> &feature) { feature_ = feature; };
    void set_split_bias(double split_bias) { split_bias_ = split_bias; };

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(id_);
        ar & BOOST_SERIALIZATION_NVP(n_samples_);
        ar & BOOST_SERIALIZATION_NVP(feature_);
        ar & BOOST_SERIALIZATION_NVP(split_bias_);
        ar & BOOST_SERIALIZATION_NVP(fraction_pos_);
        ar & BOOST_SERIALIZATION_NVP(is_leaf_);
        // note that we avoid serializing the parent here!!!
        Node *p = nullptr;
        ar & boost::serialization::make_nvp("px", p);
        ar & BOOST_SERIALIZATION_NVP(left_);
        ar & BOOST_SERIALIZATION_NVP(right_);
    }

private:
    void init_node(const Matrix& labels, 
                   const std::vector<double> &weights, 
                   bool recompute_fraction_pos=false);

    int id_;
    int n_samples_;
    std::vector<int> samples_;
    
    std::shared_ptr<BaseFeature> feature_;
    double split_bias_;

    float fraction_pos_;
    bool is_leaf_;
    
    std::weak_ptr<Node> parent_;
    std::shared_ptr<Node> left_;
    std::shared_ptr<Node> right_;
    
};

inline float Node::predict_precomputed(const RowVector& features)
{
    return get_terminal_node_precomputed(features)->fraction_pos_;
}

inline float Node::predict(
    const RowVector& features,
    std::vector<float> &cache,
    std::vector<bool> &cache_mask)
{
    return get_terminal_node(features, cache, cache_mask)->fraction_pos_;
}

inline float Node::predict(
    const cv::Mat& ii, int x, int y, 
    std::vector<float> &cache, std::vector<bool> &cache_mask)
{
    return get_terminal_node(ii, x, y, cache, cache_mask)->fraction_pos_;
}


inline float Node::predict(
    const std::vector<cv::Mat> &ii, int x, int y, 
    std::vector<float> &cache, std::vector<bool> &cache_mask)
{
    return get_terminal_node(ii, x, y, cache, cache_mask)->fraction_pos_;
}
inline float Node::predict(const RowVector& features)
{
    return get_terminal_node(features)->fraction_pos_;
}

inline std::shared_ptr<Node> Node::get_terminal_node(const RowVector& features)
{
    if (is_leaf_) { return shared_from_this(); }
    // compute the index
    double val = feature_->compute(features);
    return val <= split_bias_ ? left_->get_terminal_node(features) :
                                right_->get_terminal_node(features);
}

inline std::shared_ptr<Node> Node::get_terminal_node(
    const RowVector& features, 
    std::vector<float> &cache,
    std::vector<bool> &mask) 
{
    if (is_leaf_) { return shared_from_this(); }
    // compute the index
    double val;
    assert(cache.empty() || (int(cache.size()) > feature_->get_id() && feature_->get_id() >= 0));
    if (!cache.empty() && feature_->get_id() >= 0 && 
        int(cache.size()) > feature_->get_id() && mask[feature_->get_id()]) {
        val = cache[feature_->get_id()];
    } else {
        val = feature_->compute(features);
        if (!cache.empty() && int(cache.size()) > feature_->get_id()) {
            cache[feature_->get_id()] = val;
            mask[feature_->get_id()] = true;
        } 
    }
    return val <= split_bias_ ? 
           left_->get_terminal_node(features, cache, mask) :
           right_->get_terminal_node(features, cache, mask);
}

inline std::shared_ptr<Node> Node::get_terminal_node(
    const std::vector<cv::Mat> &ii, int x, int y, std::vector<float> &cache, std::vector<bool> &mask) 
{
    if (is_leaf_) { return shared_from_this(); }
    // compute the index
    double val;
    assert(cache.empty() || (int(cache.size()) > feature_->get_id() && feature_->get_id() >= 0));
    if (!cache.empty() && feature_->get_id() >= 0 && int(cache.size()) > feature_->get_id() && mask[feature_->get_id()]) {
        val = cache[feature_->get_id()];
    } else {
        val = feature_->compute(ii, x, y);
        if (!cache.empty() && int(cache.size()) > feature_->get_id()) {
            cache[feature_->get_id()] = val;
            mask[feature_->get_id()] = true;
        } 
    }
    return val <= split_bias_ ? left_->get_terminal_node(ii, x, y, cache, mask) : right_->get_terminal_node(ii, x, y, cache, mask);
}

inline std::shared_ptr<Node> Node::get_terminal_node(
    const cv::Mat& ii, int x, int y, std::vector<float> &cache, std::vector<bool> &mask) 
{
    if (is_leaf_) { return shared_from_this(); }
    // compute the index
    double val;
    assert(cache.empty() || (int(cache.size()) > feature_->get_id() && feature_->get_id() >= 0));
    if (!cache.empty() && feature_->get_id() >= 0 && int(cache.size()) > feature_->get_id() && mask[feature_->get_id()]) {
        val = cache[feature_->get_id()];
    } else {
        val = feature_->compute(ii, x, y);
        if (!cache.empty() && int(cache.size()) > feature_->get_id()) {
            cache[feature_->get_id()] = val;
            mask[feature_->get_id()] = true;
        } 
    }
    return val <= split_bias_ ? left_->get_terminal_node(ii, x, y, cache, mask) : right_->get_terminal_node(ii, x, y, cache, mask);
}

}
#endif
