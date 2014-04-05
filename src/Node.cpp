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
#include <detector/Node.h>

#include <iostream>
#include <fstream>
#include <limits>
#include <float.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <detector/config.h>

using std::vector;
namespace Detector {

Node::Node(
    int id,
    const Matrix& labels,
    const std::vector<double> &weights,
    const vector<int> &samples,
    int label,
    bool recompute_impurity)
: id_(id),
  n_samples_(samples.size()),
  samples_(samples),
  split_bias_(-1),
  fraction_pos_(label)
{
	init_node(labels, weights, recompute_impurity);
}

Node::Node()
: id_(0), 
  n_samples_(0),
  split_bias_(-1),
  fraction_pos_(-1.0),
  is_leaf_(0) {}

void Node::init_node(const Matrix& labels, const std::vector<double> &weights, bool recompute_fraction_pos) 
{
    if (recompute_fraction_pos) {
        double sum = 0.0;
        double norm = 0.0;
        for (size_t i = 0; i < samples_.size(); ++i) {
            double val = labels(samples_[i],0);
            double w = weights[samples_[i]];
            sum += (val * w);
            norm += w;
        }
        assert (norm > 0);
        fraction_pos_ = sum / norm; /// (norm <= 0.0f ? 1e-5 : norm);
    }

	is_leaf_ = false;
}

}
