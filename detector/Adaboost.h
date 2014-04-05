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
#ifndef ADABOOST_H

#define ADABOOST_H

#include "DetectorCommon.h"
//#include "Feature.h"
#include "ConfigManager.h"
#include "Tree.h"
#include "BaseFeature.h"
#include "HogFeature.h"
#include "IntegralFeature.h"
#include "LbpFeature.h"
#include "serialization/std_shared_ptr.hpp"

#include <vector>
#include <memory>

#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>

#include <opencv2/core/core.hpp>

namespace Detector {
class Tree;
class Node;

/**
 *  This class is responsible for creating a boosted tree ensemble.
 */
class Adaboost 
{
public:
    enum Type {
        GENTLE = 1
    };
    enum CacheType {
        NONE = 0,
        PRECOMPUTE = 1,
        LAZY = 2
    };
    Adaboost(int type=GENTLE, int n_trees=200, int max_depth=5): 
        _type(type), 
        _max_depth(max_depth), 
        _n_trees(n_trees)  {}
    ~Adaboost() {}

    //! Returns the maximum depth
    int get_max_depth() const { return _max_depth; }
    //! Returns the tree ensemble
    std::vector<std::shared_ptr<Tree> > get_trees() const { return _trees; }

    //! Trains the ensemble
    void train(const Matrix &data, const Matrix &labels, int min_examples, const Matrix &validation_data = Matrix(), const Matrix &validation_labels = Matrix(), bool softcascade=true);
    //! Trains from precomputed data (bypasses the feature computation step)
    void train_precomputed(const Matrix &data, const Matrix &labels);
    //! Predicts from precomputed features
    float predict_precomputed(const RowVector& x);
    //! Predicts with a given cache strategy
    float predict(const RowVector& x, int cache_type = NONE);
    //! Predicts with an integral image
    float predict(const cv::Mat& ii, int x, int y, int cache_type = NONE);
    //! Predicts with an integral image
    float predict(const std::vector<cv::Mat> &ii, int x, int y, int cache_type = NONE);
    //! Returns true if this is a softcascade
    bool is_softcascade() const { return !_stage_thresholds.empty(); }
    //! Returns the stage thresholds
    std::vector<double> get_stage_thresholds() const { return _stage_thresholds; }
    //! Returns the last stage threshold
    double get_last_stage_threshold() const { return _stage_thresholds.back(); }
    //! Returns the weights of the tree
    std::vector<double> get_tree_weights() const { return _weights; }
    //! Adds a constant to the stage thresholds
    void add_to_stage_thresholds(float v);
    //! Sets the stage thresholds to a particular value
    void set_stage_thresholds(float v);

    //! Loads the features from a file
    void load_features(const std::string &filename="features_0.txt");
    //! Dumps the features to a file
    void dump_features(const std::string &filename="features_0.txt");
    //! Sets the features
    void set_features(const std::vector<std::shared_ptr<BaseFeature> > &features) { _features = features; }

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) 
    {
        // register our sub-types
        ar.register_type(static_cast<HogFeature*>(nullptr));
        ar.register_type(static_cast<IntegralFeature*>(nullptr));
        ar.register_type(static_cast<LbpFeature*>(nullptr));

        ar & BOOST_SERIALIZATION_NVP(_trees);
        ar & BOOST_SERIALIZATION_NVP(_weights);
        ar & BOOST_SERIALIZATION_NVP(_stage_thresholds);
        ar & BOOST_SERIALIZATION_NVP(_type);
        ar & BOOST_SERIALIZATION_NVP(_max_depth);
        ar & BOOST_SERIALIZATION_NVP(_features);
        std::shared_ptr<ConfigManager> mgr(ConfigManager::instance());
        ar & BOOST_SERIALIZATION_NVP(mgr);
        ConfigManager::instance().swap(mgr);
    }
    void remove_softcascade() { _stage_thresholds.clear(); }
private:
    //! Precomputes all features
    Matrix precompute_features(const Matrix &data);
    //! Dumps all feature responses
    void dump_feature_responses(const Matrix &labels);

    //! The tree-ensemble
    std::vector<std::shared_ptr<Tree> > _trees; 
    //! The corresponding weight vectors
    std::vector<double> _weights;
    //! The corresponding stage thresholds
    std::vector<double> _stage_thresholds;
    //! The type of the boosting algorithm
    int _type;
    //! The maximum depth of the trees
    int _max_depth;
    //! The number of trees
    int _n_trees;
    //! Precomputed feature responses used in training
    Matrix _feature_responses;
    //! The list of used features
    std::vector<std::shared_ptr<BaseFeature> > _features;

    template <class T>
    void prepare_cache(int cache_type, std::vector<float> &cache, 
        std::vector<bool> &cache_mask, const T &precompute_func);
};

template <class T>
void Adaboost::prepare_cache(int cache_type, std::vector<float> &cache, 
    std::vector<bool> &cache_mask, const T &precompute_func)
{
    switch (cache_type) {
    case Adaboost::PRECOMPUTE:
        cache.resize(_features.size(), 0.0f);
        cache_mask.resize(_features.size(), false);
        for (size_t i = 0; i < _features.size(); i++) {
            cache[i] = precompute_func(_features[i]);
            cache_mask[i] = true;
        }
        break;
    case Adaboost::LAZY:
        cache.resize(_features.size(), 0.0f);
        cache_mask.resize(_features.size(), false);
        break;
    case Adaboost::NONE:
    default:
        break;
    }
}


}

#endif /* end of include guard: ADABOOST_H */
