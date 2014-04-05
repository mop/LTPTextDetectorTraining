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
#ifndef TREER_H
#define TREER_H

#include <vector>

#include "DetectorCommon.h"
#include "BaseFeature.h"
#include "Node.h"

#include <boost/archive/xml_oarchive.hpp>
#include <boost/utility.hpp>


namespace Detector {

class Node;
class Rect;
class Point;

/**
 * This class represents an tree of the sliding window detector
 */
class Tree : public boost::noncopyable {
public:
    // create a tree to be trained
    Tree(int id=-1,
         int max_depth=-1,
         int min_samples_count=0);
	~Tree() = default;

    // return pointer to the leaf node where given feature vector lands
    std::shared_ptr<Node> get_terminal_node(const RowVector& features);
    // takes an input feature vector and returns the prediction
    // (mean labels of nodes' suggestions)
    float predict(const RowVector& x, std::vector<float> &cache, std::vector<bool> &cache_mask);
    float predict(const cv::Mat& ii, int x, int y, std::vector<float> &cache, std::vector<bool> &cache_mask);
    float predict(const std::vector<cv::Mat> &ii, int x, int y, 
                  std::vector<float> &cache,
                  std::vector<bool> &cache_mask);
    float predict(const RowVector& x);
    float predict_precomputed(const RowVector &x);
    // train the tree based on the given data
    void train(
        std::vector<int> &errs,
        const Matrix &labels,
        const std::vector<int> &bag_samples,
        const std::vector<std::shared_ptr<BaseFeature> > &features,
        const Matrix &feature_responses,
        const Matrix &mins, 
        const Matrix &maxs,
        const std::vector<double> &weights);
    // getters
    int id() const { return _id; }
    int n_leaves() const { return _n_leaves; }
    int n_nodes() const { return _n_nodes; }
    std::shared_ptr<Node> root() { return _root; }

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(_id);
        ar & BOOST_SERIALIZATION_NVP(_depth);
        ar & BOOST_SERIALIZATION_NVP(_max_depth);
        ar & BOOST_SERIALIZATION_NVP(_min_sample_count);

        ar & BOOST_SERIALIZATION_NVP(_n_leaves);
        ar & BOOST_SERIALIZATION_NVP(_root);
        _root->reset_parents();
        ar & BOOST_SERIALIZATION_NVP(_n_nodes);
    }

private:
    enum ChildType {
        CHILD_LEFT,
        CHILD_RIGHT
    };

	// create (left or right) child node connected to parent and update metadata
	void create_child(
      std::shared_ptr<Node> parent,
      ChildType child_type,
      std::vector<int> &errs,
      const Matrix& labels,
      const std::vector<double> &weights,
      const std::vector<int>& samples, 
      int label,
      bool recompute_impurity);
  // choose the best split out of n_dim_trials*n_thresh_trials random splits for
  // the current node and recursively optimize all children
	void optimize_node(
      std::shared_ptr<Node> node,
      std::vector<int> &errs,
      const Matrix &labels, 
      const Matrix &mins, 
      const Matrix &maxs,
      const std::vector<double> &weights);
      // this function should be called instead of Node::setToLeaf() directly
	// because it also increments leaf count in the tree
	void set_node_to_leaf(std::shared_ptr<Node> node, std::vector<int> &errors, const Matrix &labels);
    std::vector<int> get_dimension_ids() const;
    std::vector<int> get_dimension_permuted_vector(int ndims) const;


    //! The ID of the tree
    int _id;
    //! The depth of the tree
	int _depth;
    //! The maximum depth of the tree
	int _max_depth;
    //! The minimum sample count in the leafs
    int _min_sample_count;
    //! The root node of the tree
    std::shared_ptr<Node> _root;
    //! The number of nodes in the tree
	int _n_nodes;
    //! The number of leaves in the tree
	int _n_leaves;

    //! This is temporary data used only in training
    //! It consists of the sampled features and channels
    std::vector<std::shared_ptr<BaseFeature> > _features;
    //! This is temporary ata used only in training.
    //! It consists of the feature responses for each training sample
    //! Each row indicates the sample and each column the feature response.
    const Matrix *_feature_responses;
};


inline float Tree::predict_precomputed(const RowVector &x) 
{ 
    return _root->predict_precomputed(x); 
}


inline float Tree::predict(
    const RowVector& x,
    std::vector<float> &cache,
    std::vector<bool> &cache_mask) 
{
	return _root->predict(x, cache, cache_mask);
}

inline float Tree::predict(
    const cv::Mat &ii,
    int x, int y,
    std::vector<float> &cache, std::vector<bool> &cache_mask) 
{
    return _root->predict(ii, x, y, cache, cache_mask);
}

inline float Tree::predict(
    const std::vector<cv::Mat> &ii,
    int x, int y,
    std::vector<float> &cache, std::vector<bool> &cache_mask) 
{
    return _root->predict(ii, x, y, cache, cache_mask);
}

inline float Tree::predict(const RowVector& x) 
{
	return _root->predict(x);
}

inline std::shared_ptr<Node> Tree::get_terminal_node(const RowVector& features) 
{
    return _root->get_terminal_node(features);
}


}

#endif
