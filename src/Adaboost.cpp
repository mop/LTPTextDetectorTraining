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
#include <detector/Adaboost.h>
#include <detector/Tree.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <boost/timer/timer.hpp>

using namespace Detector;

static const float SOFTCASCADE_DROPOUT_RESULT = -100.0f;

void Adaboost::load_features(const std::string &filename)
{
    _features.clear();
    _features.reserve(ConfigManager::instance()->n_rand_features());

    std::ifstream ifs(filename.c_str());
    std::string line;
    int id = 0;
    while (std::getline(ifs, line)) {
        _features.push_back(BaseFeature::parse(line));
        _features.back()->set_id(id++);
    }
}

Matrix Adaboost::precompute_features(const Matrix &data)
{
    if (_features.empty() || data.rows() <= 0) return Matrix();
    Matrix feature_responses = Matrix::Zero(data.rows(), _features.size());

    #pragma omp parallel for
    for (unsigned int i = 0; i < data.rows(); ++i) {
        RowVector v = data.row(i);
        for (unsigned int j = 0; j < _features.size(); ++j) {
            feature_responses(i,j) = _features[j]->compute(v);
        }
    }
    return feature_responses;
}

void Adaboost::dump_features(const std::string &filename)
{
    std::ofstream ofs(filename.c_str());
    for (unsigned int i = 0; i < _features.size(); ++i) {
        ofs << _features[i]->to_string() << std::endl;
    }
}

static int count_npos(const Matrix &labels) {
    int pos = 0;
    for (unsigned int i = 0; i < labels.rows(); i++) {
        if (labels(i,0) > 0) ++pos;
    }
    return pos;
}

static std::vector<double> 
get_weights(const Matrix &labels)
{
    int npos = count_npos(labels);
    int nneg = labels.rows() - npos;
    std::vector<double> weights(labels.rows(), 1.0 / labels.rows());
    for (unsigned int i = 0; i < weights.size(); ++i) {
        if (labels(i,0) > 0)
            weights[i] = 1.0 / double(2.0 * npos);
        else
            weights[i] = 1.0 / double(2.0 * nneg);
    }
    return weights;
}

void Adaboost::dump_feature_responses(const Matrix &labels)
{
    std::ofstream ofs("responses.txt");
    for (int i = 0; i < labels.rows(); i++) {
        ofs << labels(i,0);
        for (int j = 0; j < _feature_responses.cols(); j++) {
            ofs << "," << _feature_responses(i,j);
        }
        ofs << std::endl;
    }
    ofs.close();
}

float Adaboost::predict_precomputed(const RowVector &data)
{
    float result = 0;
    for (int i = 0; i < int(_trees.size()); i++) {
        result += (_trees[i]->predict_precomputed(data) > 0 ? 1.0 : -1.0) * _weights[i];
    }
    return result;
}

void Adaboost::train_precomputed(const Matrix &data, const Matrix &labels)
{
    // get the features
    //load_features();
    //dump_features();

    _trees.clear();
    _trees.reserve(_n_trees);

    std::vector<int> all_samples;
    for (int i = 0; i < labels.rows(); ++i) 
        all_samples.push_back(i);

    std::vector<double> weights(get_weights(labels));
    Matrix mins = data.colwise().minCoeff();
    Matrix maxs = data.colwise().maxCoeff();

    for (int i = 0; i < _n_trees; ++i) {
        std::shared_ptr<Tree> tree(new Tree(i, _max_depth, 2));
        std::vector<int> errors(labels.rows(), 0);
        tree->train(errors, labels, all_samples, _features, data, mins, maxs, weights);

        double err = 0.0;
        // need to calculate the weighted error :(
        #pragma omp parallel for reduction(+:err)
        for (unsigned int j = 0; j < labels.rows(); ++j) {
            if (errors[j]) {
                err += weights[j];
            } 
        }

        err = std::max(1e-5, std::min(1.0-1e-5, err));
        float alpha = 0.5 * log((1 - err) / err);
        double normalize = 0.0;

        #pragma omp parallel for reduction(+:normalize)
        for (unsigned int j = 0; j < labels.rows(); ++j) {
            if (errors[j]) {
                weights[j] *= exp(alpha);
            } else {
                weights[j] *= exp(-alpha);
            }
            normalize += weights[j];
        }

        // renormalize the weights
        #pragma omp parallel for
        for (unsigned int j = 0; j < weights.size(); ++j) {
            weights[j] = weights[j] / (normalize <= 0.0 ? 1e-5f : normalize);
        }

        _trees.push_back(tree);
        _weights.push_back(alpha);
    }
}

void Adaboost::train(const Matrix &data, const Matrix &labels, int min_examples, const Matrix &validation_data, const Matrix &validation_labels, bool softcascade)
{
    // get the features
    //load_features();
    //dump_features();

    _feature_responses = precompute_features(data);
    //dump_feature_responses(labels);

    // free up memory
    Matrix &d = const_cast<Matrix&>(data);
    d.resize(0,0);

    _trees.clear();
    _trees.reserve(_n_trees);
    _weights.reserve(_n_trees);

    std::vector<int> all_samples;
    for (int i = 0; i < labels.rows(); ++i) 
        all_samples.push_back(i);

    std::vector<double> weights(get_weights(labels));

    Matrix mins = _feature_responses.colwise().minCoeff();
    Matrix maxs = _feature_responses.colwise().maxCoeff();

    for (int i = 0; i < _n_trees; ++i) {
        boost::timer::cpu_timer iteration_timer;
        iteration_timer.start();
        boost::timer::cpu_timer t;
        t.start();
        std::shared_ptr<Tree> tree(new Tree(i, _max_depth, min_examples));
        std::vector<int> errors(labels.rows(), 0);
        tree->train(errors, labels, 
            all_samples, 
            _features, _feature_responses, 
            mins, maxs, weights);
        //std::cout << "Trained tree in: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

        t.start();
        double err = 0.0;
        // need to calculate the weighted error :(
        #pragma omp parallel for reduction(+:err)
        for (unsigned int j = 0; j < labels.rows(); ++j) {
            if (errors[j]) {
                err += weights[j];
            } 
        }
        //std::cout << "Measured errors in: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

        err = std::max(1e-5, std::min(1.0-1e-5, err));
        const float alpha = std::max(-5.0, std::min(5.0, 0.5 * log((1 - err) / err)));
        double normalize = 0.0;

        t.start();
        #pragma omp parallel for reduction(+:normalize)
        for (unsigned int j = 0; j < labels.rows(); ++j) {
            if (errors[j]) {
                weights[j] *= exp(alpha);
            } else {
                weights[j] *= exp(-alpha);
            }
            normalize += weights[j];
        }
        //std::cout << "Computed normalization in: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

        t.start();
        // renormalize the weights
        #pragma omp parallel for
        for (unsigned int j = 0; j < weights.size(); ++j) {
            weights[j] = weights[j] / (normalize <= 0.0 ? 1e-5f : normalize);
        }
        //std::cout << "Recomputed weights in: " << boost::timer::format(t.elapsed(), 5, "%w") << std::endl;

        std::cout << "Iteration: " << i 
                  << " Error: " << err  
                  << " Alpha: " << alpha 
                  << " Time: " << boost::timer::format(
                        iteration_timer.elapsed(), 5, "%w") << std::endl;

        _trees.push_back(tree);
        _weights.push_back(alpha);
    }

    float target_recall = ConfigManager::instance()->recall_threshold();
    std::vector<float> predictions(validation_data.rows(), 0.0f);

    #pragma omp parallel for
    for (int i = 0; i < validation_data.rows(); i++) {
        if (validation_labels(i, 0) > 0) {
            predictions[i] = predict(validation_data.row(i), false);
        }
    }

    if (validation_labels.rows() == 0) {
        _stage_thresholds.clear();
        for (size_t i = 0; i < _trees.size(); i++) {
            _stage_thresholds.push_back(-1.0f);
        }
        return;
    }
    float min_el = *std::min_element(predictions.begin(), predictions.end());
    float max_el = *std::max_element(predictions.begin(), predictions.end());

    int n_threshs = 20;
    std::vector<float> thresholds(n_threshs);
    for (int i = 0; i < n_threshs; i++) {
        thresholds[i] = min_el + (max_el - min_el) / float(n_threshs) * i;
    }

    std::vector<float> recalls(n_threshs, 0.0f);
    #pragma omp parallel for
    for (int i = 0; i < n_threshs; i++) {
        int detected = 0;
        int n_positives = 0;
        for (int j = 0; j < validation_labels.rows(); j++) {
            if (validation_labels(j, 0) > 0) {
                n_positives++;
                if (predictions[j] > thresholds[i]) {
                    detected++;
                }
            }
            recalls[i] = float(detected) / float (n_positives);
        }
    }
    std::cout << "Recalls: ";
    for (float r : recalls) {
        std::cout <<  r << ",";
    }
    std::cout << std::endl;

    int best_recall = 0;
    for (size_t i = 1; i < recalls.size(); i++) {
        if (recalls[i] >= target_recall) {
            best_recall = i;
        } else {
            break;
        }
    }

    // find the softcascade thresholds
    //std::cout << validation_data.rows() << " " << validation_labels.rows() << std::endl;
    std::vector<unsigned char> valid(validation_data.rows(), 0);
    #pragma omp parallel for
    for (int i = 0; i < validation_data.rows(); i++) {
        if (validation_labels(i,0) > 0 && predictions[i] > thresholds[best_recall]) {
            valid[i] = 1;
        }
    }

    _feature_responses.resize(0,0);
    std::vector<float> responses(validation_data.rows(), 0.0f);
    _stage_thresholds.clear(); _stage_thresholds.resize(_n_trees);
    if (!softcascade || validation_data.rows() <= 0) return;
    std::cout << "Searching thresholds with " << validation_data.rows() << " data points" << std::endl;
    for (int i = 0; i < _n_trees; i++) {
        boost::timer::cpu_timer t;
        //std::cout << "stage: " << i << std::endl;
        #pragma omp parallel for
        for (int j = 0; j < validation_data.rows(); j++) {
            if (valid[j] && validation_labels(j,0) > 0) {
                responses[j] += (_trees[i]->predict(
                    validation_data.row(j)) > 0 ? 1.0 : -1.0) * _weights[i];
            }
        }

        float stage_threshold = std::numeric_limits<float>::max();
        for (int j =0; j < validation_data.rows(); j++) {
            if (valid[j] && validation_labels(j,0) > 0) {
                stage_threshold = std::min(responses[j], stage_threshold);
            }
        }
        _stage_thresholds[i] = stage_threshold;
        std::cout << boost::timer::format(t.elapsed(), 6, "%w") << std::endl;
    }
}

float Adaboost::predict(const cv::Mat &ii, int x, int y, int cache_type) 
{
    std::vector<float> cache;
    std::vector<bool> cache_mask;
    prepare_cache(cache_type, cache, cache_mask, [&ii, &x, &y](
            const std::shared_ptr<BaseFeature> &f) -> float {
        return f->compute(ii, x, y);
    });

    float result = 0;
    for (int i = 0; i < int(_trees.size()); ++i) {
        result += ((_trees[i]->predict(ii, x, y, cache, cache_mask)) > 0 ? 1.0 : -1.0) * _weights[i];
        if (is_softcascade() && result < _stage_thresholds[i] - 1e-5) {
            return SOFTCASCADE_DROPOUT_RESULT;
        }
    }
    return result;
}

float Adaboost::predict(const std::vector<cv::Mat> &ii, int x, int y, int cache_type) 
{
    std::vector<float> cache;
    std::vector<bool> cache_mask;
    prepare_cache(cache_type, cache, cache_mask, [&ii, &x, &y](
            const std::shared_ptr<BaseFeature> &f) -> float {
        return f->compute(ii, x, y);
    });

    float result = 0;
    for (int i = 0; i < int(_trees.size()); ++i) {
        result += ((_trees[i]->predict(ii, x, y, cache, cache_mask)) > 0 ? 1.0 : -1.0) * _weights[i];
        if (is_softcascade() && result < _stage_thresholds[i] - 1e-5) {
            return SOFTCASCADE_DROPOUT_RESULT;
        }
    }
    return result;
}

float Adaboost::predict(const RowVector &x, int cache_type)
{
    std::vector<float> cache;
    std::vector<bool> cache_mask;
    prepare_cache(cache_type, cache, cache_mask, [&x](
            const std::shared_ptr<BaseFeature> &f) -> float {
        return f->compute(x);
    });

    float result = 0;
    for (int i = 0; i < int(_trees.size()); ++i) {
        result += ((_trees[i]->predict(x, cache, cache_mask)) > 0 ? 1.0 : -1.0) * _weights[i];
        if (is_softcascade() && result < _stage_thresholds[i] - 1e-5) {
            return SOFTCASCADE_DROPOUT_RESULT;
        }
    }
    return result;
}

void Adaboost::add_to_stage_thresholds(float v)
{
    for (size_t i = 0; i < _stage_thresholds.size(); i++) {
        _stage_thresholds[i] += v;
    }
}

void Adaboost::set_stage_thresholds(float v)
{
    for (size_t i = 0; i < _stage_thresholds.size(); i++) {
        _stage_thresholds[i] = v;
    }
}

