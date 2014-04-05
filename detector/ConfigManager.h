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
#ifndef CONFIGMANAGER_H

#define CONFIGMANAGER_H

#include <string>
#include <memory>
#include <boost/archive/xml_oarchive.hpp>

namespace Detector {
//! This class is responsible for managing the configuration
class ConfigManager 
{
public:
    ConfigManager(const std::string &filename);
    ConfigManager();
    ~ConfigManager();

    //! Returns the width of the window
    int feature_width() const { return _feature_width; }
    //! Returns the height of the window
    int feature_height() const { return _feature_height; }
    //! Returns the width of the window + 1, which is the integal image feature width
    int ii_feature_width() const { return _feature_width + 1; }
    //! Returns the height of the window + 1, which is the integal image feature height
    int ii_feature_height() const { return _feature_height + 1; }

    //! Maximal width for featureset generation
    int max_w() const { return _max_w; }
    //! Maximal height for featureset generation
    int max_h() const { return _max_h; }
    //! Minimal width for featureset generation
    int min_w() const { return _min_w; }
    //! Minimal height for featureset generation
    int min_h() const { return _min_h; }

    //! Returns the number of random features to be generated
    int n_rand_features() const { return _n_rand_features; }
    //! Returns the fraction of second order features
    float fraction_second_order() const { return _fraction_second_order; }

    //! Returns the number of HOG channels
    int hog_channels() const { return _hog_channels; }
    //! Returns the norm index of the hog channel (== last channel)
    int hog_norm_channel_index() const { return _hog_channels; }

    //! Returns the number of integral channels
    int integral_channels() const { return _integral_channels; }
    //! Returns the offset in the feature 
    int integral_channel_offset() const { return _integral_channels > 0 ? hog_size() : -1; }

    //! Number of lbp histograms
    int lbp_histograms() const { return _lbp_histograms; }
    //! Number of LBP feature maps
    int lbp_channels() const { return _lbp_histograms * 256; }
    //! Offset of the lbps
    int lbp_channel_offset() const { return _lbp_histograms > 0 ? hog_size() + integral_size() : -1; }
    
    //! Number of ltp histograms
    int ltp_histograms() const { return _ltp_histograms; }
    //! Number of LTP feature maps
    int ltp_channels() const { return _ltp_histograms * 256; }
    //! Offset of the lbps
    int ltp_channel_offset() const { return _ltp_histograms > 0 ? hog_size() + lbp_size() + integral_size() : -1; }

    int hog_channels_start() const { return 0; }
    int hog_channels_end() const { return hog_size() > 0 ? hog_channels_start() + hog_channels() : 0; }

    int integral_channels_start() const { return hog_channels_end(); }
    int integral_channels_end() const { return integral_channels() > 0 ? integral_channels_start() + integral_channels() : integral_channels_start(); }

    int lbp_channels_start() const { return integral_channels_start(); }
    int lbp_channels_end() const { return lbp_size() > 0 ? lbp_channels_start() + lbp_channels() : lbp_channels_start(); }
    int ltp_channels_start() const { return lbp_channels_end(); }
    int ltp_channels_end() const { return ltp_size() > 0 ? ltp_channels_start() + ltp_channels() : ltp_channels_start(); }

    //! Returns the softcascade recall threshold
    float recall_threshold() const { return _recall_threshold; }

    //! Returns the maximum decision tree depth
    int max_depth() const { return _max_depth; }
    //! Returns the minimum number of samples in a leaf
    int min_sample_count() const { return _min_sample_count; }
    //! Returns the number of trees
    int num_trees() const { return _num_trees; }
    //! Returns the output of the model file
    std::string output() const { return _output; }
    //! Returns the output of the validation results
    std::string output_validation() const { return _output_validation; }
    //! Returns the the training file (input)
    std::string train_file() const { return _train_file; }
    //! Returns the validation file (input)
    std::string validation_file() const { return _validation_file; }
    //! Returns the feature file (input/output)
    std::string feature_file() const { return _feature_file; }
    //! Flag which indicates if the std hog features should be generated
    bool generate_std_hog() const { return _generate_std_hog; }
    //! Flag which indicates if the std lbp features should be generated
    bool generate_std_lbp() const { return _generate_std_lbp; }
    //! Flag which indicates if the std ltp features should be generated
    bool generate_std_ltp() const { return _generate_std_ltp; }
    //! Flag which indicates if random hogs should be generated
    bool generate_random_hog() const { return _generate_random_hog; }
    //! Returns true if the given channel is ignored
    bool is_channel_ignored(int c) const { return std::find(_ignore_channels.begin(), _ignore_channels.end(), c) != _ignore_channels.end(); }
    //! Returns a list of channels ignored
    std::vector<int> get_ignore_channels() const { return _ignore_channels; }
    //! Returns if we should use L2 norm
    bool is_l2_norm() const { return _l2_norm; }

    //! Returns the instance of the singleton
    static std::shared_ptr<ConfigManager>& instance() { return _instance; }

    int hog_size() const { return _hog_channels > 0 ? _hog_channels * ii_feature_width() * ii_feature_height() : 0; } 
    int integral_size() const { return _integral_channels > 0 ? (_integral_channels) * ii_feature_width() * ii_feature_height() : 0; } 
    int lbp_size() const { return _lbp_histograms > 0 ? _lbp_histograms * feature_width() * feature_height() : 0; }
    int ltp_size() const { return _ltp_histograms > 0 ? _ltp_histograms * feature_width() * feature_height() : 0; }

    void set_hog_channels(int c) { _hog_channels = c; }
    void set_hog_norm_channel_idx(int c) { _hog_norm_channel_idx = c; }
    void set_integral_channels(int c) { _integral_channels = c; }
    void set_lbp_histograms(int c) { _lbp_histograms = c; }
    void set_ltp_histograms(int c) { _ltp_histograms = c; }
    void set_feature_width(int w) { _feature_width = w; }
    void set_feature_height(int h) { _feature_height = h; }

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(_feature_width);
        ar & BOOST_SERIALIZATION_NVP(_feature_height);
        ar & BOOST_SERIALIZATION_NVP(_max_w);
        ar & BOOST_SERIALIZATION_NVP(_max_h);
        ar & BOOST_SERIALIZATION_NVP(_min_w);
        ar & BOOST_SERIALIZATION_NVP(_min_h);

        ar & BOOST_SERIALIZATION_NVP(_n_rand_features);
        ar & BOOST_SERIALIZATION_NVP(_fraction_second_order);

        ar & BOOST_SERIALIZATION_NVP(_hog_channels);
        ar & BOOST_SERIALIZATION_NVP(_hog_norm_channel_idx);
        ar & BOOST_SERIALIZATION_NVP(_l2_norm);
        ar & BOOST_SERIALIZATION_NVP(_integral_channels);
        ar & BOOST_SERIALIZATION_NVP(_lbp_histograms);
        ar & BOOST_SERIALIZATION_NVP(_ltp_histograms);
        ar & BOOST_SERIALIZATION_NVP(_ignore_channels);

        ar & BOOST_SERIALIZATION_NVP(_recall_threshold);

        ar & BOOST_SERIALIZATION_NVP(_max_depth);
        ar & BOOST_SERIALIZATION_NVP(_min_sample_count);
        ar & BOOST_SERIALIZATION_NVP(_num_trees);
        ar & BOOST_SERIALIZATION_NVP(_output);
        ar & BOOST_SERIALIZATION_NVP(_output_validation);
        ar & BOOST_SERIALIZATION_NVP(_train_file);
        ar & BOOST_SERIALIZATION_NVP(_validation_file);
        ar & BOOST_SERIALIZATION_NVP(_feature_file);
        ar & BOOST_SERIALIZATION_NVP(_generate_std_hog);
        ar & BOOST_SERIALIZATION_NVP(_generate_std_lbp);
        ar & BOOST_SERIALIZATION_NVP(_generate_std_ltp);
        ar & BOOST_SERIALIZATION_NVP(_generate_random_hog);
    }
private:
    static std::shared_ptr<ConfigManager> _instance;

    int _feature_width;
    int _feature_height;
    int _max_w;
    int _max_h;
    int _min_w;
    int _min_h;

    int _n_rand_features;
    float _fraction_second_order;

    int _hog_channels;
    int _hog_norm_channel_idx;
    int _integral_channels;
    int _lbp_histograms;
    int _ltp_histograms;
    std::vector<int> _ignore_channels;
    bool _l2_norm;

    float _recall_threshold;

    int _max_depth;
    int _min_sample_count;
    int _num_trees;
    std::string _output;
    std::string _output_validation;
    std::string _train_file;
    std::string _validation_file;
    std::string _feature_file;
    bool _generate_std_hog;
    bool _generate_std_lbp;
    bool _generate_std_ltp;
    bool _generate_random_hog;
};
}

#endif /* end of include guard: CONFIGMANAGER_H */
