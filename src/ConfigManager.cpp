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
#include <detector/ConfigManager.h>

#include <iostream>
#include <fstream>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace Detector {
std::shared_ptr<ConfigManager> ConfigManager::_instance;
ConfigManager::ConfigManager()
: _feature_width(-1), _feature_height(-1), 
  _max_w(-1), _max_h(-1),
  _min_w(-1), _min_h(-1),
  _n_rand_features(-1),
  _fraction_second_order(0.0f),
  _hog_channels(8),
  _hog_norm_channel_idx(8),
  _integral_channels(4),
  _lbp_histograms(8),
  _ltp_histograms(16),
  _l2_norm(false),
  _recall_threshold(0.95),
  _max_depth(-1),
  _min_sample_count(-1),
  _num_trees(-1),
  _generate_std_hog(false),
  _generate_std_lbp(false),
  _generate_std_ltp(false),
  _generate_random_hog(false)

{
}

ConfigManager::ConfigManager(const std::string &filename)
: _feature_width(-1), _feature_height(-1), 
  _max_w(-1), _max_h(-1),
  _min_w(-1), _min_h(-1),
  _n_rand_features(-1),
  _fraction_second_order(0.0f),
  _hog_channels(8),
  _hog_norm_channel_idx(8),
  _lbp_histograms(8),
  _ltp_histograms(16),
  _l2_norm(false),
  _recall_threshold(0.95),
  _max_depth(-1),
  _min_sample_count(-1),
  _num_trees(-1),
  _generate_std_hog(false),
  _generate_std_lbp(false),
  _generate_std_ltp(false),
  _generate_random_hog(false)
{
    po::options_description desc("Options");
    desc.add_options()
        ("feature_width", po::value<int>(&_feature_width), "Width of window")
        ("feature_height", po::value<int>(&_feature_height), "Height of window")
        ("max_w", po::value<int>(&_max_w), "Maximal width of a feature")
        ("min_w", po::value<int>(&_min_w), "Minimal width of a feature")
        ("max_h", po::value<int>(&_max_h), "Maximal height of a feature")
        ("min_h", po::value<int>(&_min_h), "Minimal height of a feature")
        ("n_rand_features", po::value<int>(&_n_rand_features), "Number of random features")
        ("fraction_second_order", po::value<float>(&_fraction_second_order), "Fraction of second order features")
        ("hog_norm_channel_idx", po::value<int>(&_hog_norm_channel_idx), "Normalization channel of HOG features (disable: -1)")
        ("hog_channels", po::value<int>(&_hog_channels), "Number of HOG histogram channels used")
        ("ignore_channels", po::value<std::vector<int> >(&_ignore_channels)->multitoken(), "channels to ignore in feature generation")
        ("integral_channels", po::value<int>(&_integral_channels), "Number of Integral channels used")
        ("lbp_histograms", po::value<int>(&_lbp_histograms), "Number of LBP histograms used (8)")
        ("ltp_histograms", po::value<int>(&_ltp_histograms), "Number of LTP histograms used (16)")
        ("l2_norm", po::value<bool>(&_l2_norm), "Use l2 norm for HOG")
        ("recall_threshold", po::value<float>(&_recall_threshold), "Recall threshold for the softcascade")
        ("max_depth,d", po::value<int>(&_max_depth), "Maximum depth of the tree ensemble")
        ("min_sample_count,m", po::value<int>(&_min_sample_count), "Minimum sample count")
        ("num_trees,n", po::value<int>(&_num_trees), "Number of trees")
        ("output,o", po::value<std::string>(&_output)->required(), "output of the model file")
        ("output_validation", po::value<std::string>(&_output_validation)->required(), "output of the validation results")
        ("train_file,i", po::value<std::string>(&_train_file)->required(), "file containing the traing samples")
        ("validation_file,v", po::value<std::string>(&_validation_file)->required(), "file containing the validation samples")
        ("feature_file,f", po::value<std::string>(&_feature_file), "file the features to use")
        ("generate_std_hog", po::value<bool>(&_generate_std_hog), "generate and dump std hog features")
        ("generate_std_lbp", po::value<bool>(&_generate_std_lbp), "generate and dump std lbp features")
        ("generate_std_ltp", po::value<bool>(&_generate_std_ltp), "generate and dump std ltp features")
        ("generate_random_hog", po::value<bool>(&_generate_random_hog), "generate and dump random hog features");

    std::ifstream ifs(filename.c_str());

    auto vm = po::variables_map();
    po::store(po::parse_config_file(ifs, desc), vm);
    ifs.close();
    po::notify(vm);

    if (!_generate_std_hog && !_generate_std_lbp && !_generate_std_ltp && !_generate_random_hog) {
        if (_feature_file == "") {
            std::cerr << "No features specified" << std::endl;
            assert(false);
        }

        if (!fs::exists(_feature_file)) {
            std::cerr << "No features specified" << std::endl;
            assert(false);
        }
    }

    if (!fs::exists(_train_file)) {
        std::cerr << "Training file does not exist" << std::endl;
        assert(false);
    }

    if (!fs::exists(_validation_file)) {
        std::cerr << "Validation file does not exist" << std::endl;
        assert(false);
    }
}

ConfigManager::~ConfigManager() {}

}
