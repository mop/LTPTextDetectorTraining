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
#include <detector/Feature.h>
#include <detector/ConfigManager.h>

using namespace Detector;

#include <iostream>

/*
double Detector::query_ii(const RowVector &features, int channel, const Rect &rect, const ConfigManager &mgr)
{
    const int ii_w = mgr.ii_feature_width();
    const int ii_h = mgr.ii_feature_height();

    int pic_start = channel * (ii_w) * (ii_h);
    int idx1 = pic_start + (rect.p2.y+1)*(ii_w) + rect.p2.x+1;
    int idx2 = pic_start + (rect.p1.y)*(ii_w) + rect.p1.x;
    int idx3 = pic_start + (rect.p2.y+1)*(ii_w) + rect.p1.x;
    int idx4 = pic_start + (rect.p1.y)*(ii_w) + rect.p2.x+1;
    return features(idx1) - features(idx3) - features(idx4) + features(idx2);
}

double Detector::query_ii(
    const cv::Mat &ii, 
    int x, int y, 
    int channel, 
    const Rect &rect)
{
    const double *ptr = ii.ptr<double>(0,0);
    const int s0 = ii.step[0] / sizeof (double);
    const int s1 = ii.step[1] / sizeof (double);
    const double v1 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p2.x + 1) + channel);
    const double v2 = *(ptr + s0 * (y + rect.p1.y) + s1 * (x + rect.p2.x + 1) + channel);
    const double v3 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p1.x) + channel);
    const double v4 = *(ptr + s0 * (y + rect.p1.y) + s1 * (x + rect.p1.x) + channel);

    return v1 - v2 - v3 + v4;
}

static inline double norm_channels(
        double *result, 
        const int channel, 
        const int hog_channels, 
        const double clamp_value=0.2)
{
    auto sum_squared = [](const double val, const double &vec_val) -> double {
        return val + vec_val*vec_val;
    };
    double magnitude = std::accumulate(result, result + hog_channels, 0.0, sum_squared);
    magnitude = std::max(1e-5, std::sqrt(std::max(1e-5, magnitude)));
    for (int i = 0; i < hog_channels; i++) {
        result[i] = result[i] / magnitude;
        result[i] = std::min(clamp_value, result[i]);
    }

    magnitude = std::accumulate(result, result + hog_channels, 0.0, sum_squared);
    magnitude = std::max(1e-5, std::sqrt(std::max(1e-5, magnitude)));
    return result[channel] / magnitude;
}

double Detector::query_and_norm_l2_clamp_ii(
    const RowVector &features, 
    const int channel, 
    const int hog_channels,
    const Rect &rect,
    const ConfigManager &mgr, 
    const double clamp_value)
{
    const int ii_w = mgr.ii_feature_width();
    const int ii_h = mgr.ii_feature_height();

    std::vector<double> result(hog_channels);
    for (int i = 0; i < hog_channels; i++) {
        int pic_start = i * (ii_w) * (ii_h);
        int idx1 = pic_start + (rect.p2.y+1)*(ii_w) + rect.p2.x+1;
        int idx2 = pic_start + (rect.p1.y)*(ii_w) + rect.p1.x;
        int idx3 = pic_start + (rect.p2.y+1)*(ii_w) + rect.p1.x;
        int idx4 = pic_start + (rect.p1.y)*(ii_w) + rect.p2.x+1;
        result[i] = features(idx1) - features(idx3) - features(idx4) + features(idx2);
    }
    return norm_channels(&result[0], channel, hog_channels, clamp_value);
}

double Detector::query_and_norm_l2_clamp_ii(
    const cv::Mat &ii,
    const int x, const int y, 
    const int channel, 
    const int hog_channels,
    const Rect &rect, 
    const double clamp_value)
{
    const double *ptr = ii.ptr<double>(0,0);
    std::vector<double> result(hog_channels);
    for (int i = 0; i < hog_channels; i++) {
        const double v1 = *(ptr + ii.step[0] / sizeof (double) * (y + rect.p2.y+1) + ii.step[1] / sizeof (double) * (x + rect.p2.x + 1) + i);
        const double v2 = *(ptr + ii.step[0] / sizeof (double) * (y + rect.p1.y) + ii.step[1] / sizeof (double) * (x + rect.p2.x + 1) + i);
        const double v3 = *(ptr + ii.step[0] / sizeof (double) * (y + rect.p2.y+1) + ii.step[1] / sizeof (double) * (x + rect.p1.x) + i);
        const double v4 = *(ptr + ii.step[0] / sizeof (double) * (y + rect.p1.y) + ii.step[1] / sizeof (double) * (x + rect.p1.x) + i);
        result[i] = v1 - v2 - v3 + v4;
    }

    return norm_channels(&result[0], channel, hog_channels, clamp_value);
}
*/

Feature::Feature(const Rect &r1, int c, std::shared_ptr<ConfigManager> mgr)
: _rect1(r1), _rect2(Rect()), _channel1(c), _channel2(0), _type(FIRST_ORDER), _config(mgr)
{
}

Feature::Feature(const Rect &r1, int c1, const Rect &r2, int c2, std::shared_ptr<ConfigManager> mgr)
: _rect1(r1), _rect2(r2), _channel1(c1), _channel2(c2), _type(SECOND_ORDER), _config(mgr)
{
}

Feature::~Feature() {}

double Feature::compute_hog(const RowVector &r) const
{
    const int feature_w = _config->feature_width();
    const int feature_h = _config->feature_height();
    const Rect whole_patch(Point(0,0), Point(feature_w-1, feature_h-1));
    const int norm_channel_idx = _config->hog_norm_channel_index();

    if (_type == FIRST_ORDER) {
        if (_config->is_l2_norm())
            return query_and_norm_l2_clamp_ii(r, 
                _channel1, _rect1, 
                _config->hog_channels_start(), _config->hog_channels_end(),
                _config->feature_width(), _config->feature_height());

        double val = query_ii(r, _channel1, _rect1, feature_w, feature_h);
        if (norm_channel_idx < 0)
            return val;

        double norm = query_ii(r, norm_channel_idx, whole_patch, feature_w, feature_h);
        norm = std::max(norm, val);
        return (val < 1e-5 || norm < 1e-5) ? 0.0 : sqrt(val / (norm + 1e-5));
    } else {
        double val1 = query_ii(r, _channel1, _rect1, feature_w, feature_h);
        double norm1 = std::max(query_ii(r, norm_channel_idx, whole_patch, feature_w, feature_h), val1);
        double val2 = query_ii(r, _channel2, _rect2, feature_w, feature_h);
        double norm2 = std::max(query_ii(r, norm_channel_idx, whole_patch, feature_w, feature_h), val2);
        val1 = (val1 < 1e-5 || norm1 < 1e-5) ? 0.0 : sqrt(val1 / (norm1 + 1e-5));
        val2 = (val2 < 1e-5 || norm2 < 1e-5) ? 0.0 : sqrt(val2 / (norm2 + 1e-5));
        if (val2 > 1.0 || val1 > 1.0) {
            std::cout << val2 << " " << val1 << std::endl;
        }
        val1 = std::min(1.0, std::max(val1, 0.0));
        val2 = std::min(1.0, std::max(val2, 0.0));
        // val1 \in [0,1], val2 \in [0,1]
        // -> worst case: 0 - 1 = -1 and 1 - 0 == 1
        // (val1 - val2) \in [-1,1]
        // -> ((val1 - val2) + 1.0) / 2.0 \in [0,1]
        if (val1 - val2 < -1) {
            std::cout << (val1 - val2) << std::endl;
        }
        return ((val1 - val2) + 1.0) / 2.0;
    }
}

double Feature::compute_int(const RowVector &r, int chan_offset) const
{
    const double norm = _config->feature_width() * _config->feature_height();
    const int feature_w = _config->feature_width();
    const int feature_h = _config->feature_height();
    if (_type == FIRST_ORDER) {
        double val = query_ii(r, _channel1 + chan_offset, _rect1, feature_w, feature_h);
        return val / std::max(norm, val);
    } else {
        double val1 = query_ii(r, _channel1 + chan_offset, _rect1, feature_w, feature_h) / norm;
        double val2 = query_ii(r, _channel2 + chan_offset, _rect2, feature_w, feature_h) / norm;
        if (val2 > 1.0 || val1 > 1.0) {
            std::cout << val2 << " " << val1 << std::endl;
        }
        val1 = std::min(1.0, std::max(val1, 0.0));
        val2 = std::min(1.0, std::max(val2, 0.0));
        // val1 \in [0,1], val2 \in [0,1]
        // -> worst case: 0 - 1 = -1 and 1 - 0 == 1
        // (val1 - val2) \in [-1,1]
        // -> ((val1 - val2) + 1.0) / 2.0 \in [0,1]
        if (val1 - val2 < -1) {
            std::cout << (val1 - val2) << std::endl;
        }
        return ((val1 - val2) + 1.0) / 2.0;
    }
}

double Feature::compute_int(const cv::Mat &ii, int x, int y, int chan_offset) const
{
    const double norm = 
        _config->feature_width() *
        _config->feature_height();
    if (_type == FIRST_ORDER) {
        double val  = query_ii(ii, x, y, _channel1 + chan_offset, _rect1);
        return val / norm;
    } else {
        double val1 = query_ii(ii, x, y, _channel1 + chan_offset, _rect1) / norm;
        double val2 = query_ii(ii, x, y, _channel2 + chan_offset, _rect2) / norm;
        if (val2 > 1.0 || val1 > 1.0) {
            std::cout << val2 << " " << val1 << std::endl;
        }
        val1 = std::min(1.0, std::max(val1, 0.0));
        val2 = std::min(1.0, std::max(val2, 0.0));
        // val1 \in [0,1], val2 \in [0,1]
        // -> worst case: 0 - 1 = -1 and 1 - 0 == 1
        // (val1 - val2) \in [-1,1]
        // -> ((val1 - val2) + 1.0) / 2.0 \in [0,1]
        if (val1 - val2 < -1) {
            std::cout << (val1 - val2) << std::endl;
        }
        return ((val1 - val2) + 1.0) / 2.0;
    }
}

double Feature::compute_hog(const cv::Mat &ii, int x, int y) const
{
    const int feature_w = _config->feature_width();
    const int feature_h = _config->feature_height();
    const Rect whole_patch(Point(0,0), Point(feature_w-1, feature_h-1));
    const int norm_channel_idx = _config->hog_norm_channel_index();
    if (_type == FIRST_ORDER) {
        if (_config->is_l2_norm())
            return query_and_norm_l2_clamp_ii(ii, x, y, 
                _channel1, _rect1, _config->hog_channels_start(), _config->hog_channels_end(),
                _config->feature_width(), _config->feature_height());
        double val = query_ii(ii, x, y, _channel1, _rect1);
        if (norm_channel_idx < 0)
            return val;
        //return val;
        //double norm = query_ii(r, NORM_CHANNEL_IDX, _rect1);
        double norm = std::max(val, query_ii(ii, x, y, norm_channel_idx, whole_patch));
        return (val < 1e-5 || norm < 1e-5) ? 0.0 : sqrt(val / (norm + 1e-5));
    } else {
        double val1 = query_ii(ii, x, y, _channel1, _rect1);
        double norm1 = std::max(query_ii(ii, x, y, norm_channel_idx, whole_patch), val1);
        double val2 = query_ii(ii, x, y, _channel2, _rect2);
        double norm2 = std::max(query_ii(ii, x, y, norm_channel_idx, whole_patch), val2);
        val1 = (val1 < 1e-5 || norm1 < 1e-5) ? 0.0 : sqrt(val1 / (norm1 + 1e-5));
        val2 = (val2 < 1e-5 || norm2 < 1e-5) ? 0.0 : sqrt(val2 / (norm2 + 1e-5));
        if (val2 > 1.0 || val1 > 1.0) {
            std::cout << val2 << " " << val1 << std::endl;
        }
        val1 = std::min(1.0, std::max(val1, 0.0));
        val2 = std::min(1.0, std::max(val2, 0.0));
        // val1 \in [0,1], val2 \in [0,1]
        // -> worst case: 0 - 1 = -1 and 1 - 0 == 1
        // (val1 - val2) \in [-1,1]
        // -> ((val1 - val2) + 1.0) / 2.0 \in [0,1]
        if (val1 - val2 < -1) {
            std::cout << (val1 - val2) << std::endl;
        }
        return ((val1 - val2) + 1.0) / 2.0;
    }
}

static inline double compute_lbp_count(const RowVector &row, const Rect &r, int lbp_pattern, int lbp_offset, const ConfigManager &mgr)
{
    int lbp_chan = lbp_pattern / 256;
    lbp_pattern  = lbp_pattern % 256;

    int cols = mgr.feature_width();
    int rows = mgr.feature_height();

    double result = 0.0;
    for (int i = r.p1.y; i < r.p2.y+1; ++i) {
        for (int j = r.p1.x; j < r.p2.x + 1; ++j) {
            int idx = lbp_offset + lbp_chan * (cols*rows) + i*cols + j;
            if (row(0, idx) == lbp_pattern) {
                result += 1.0;
            }
        }
    }
    return result / (cols * rows);
}

static inline double 
compute_lbp_count(const cv::Mat &ii, int x, int y, const Rect &r, int lbp_pattern, int lbp_offset, const ConfigManager &mgr)
{
    int lbp_chan = lbp_pattern / 256;
    lbp_pattern  = lbp_pattern % 256;

    int cols = mgr.feature_width();
    int rows = mgr.feature_height();

    const double *data = ii.ptr<double>(0,0);
    double result = 0.0;
    for (int i = r.p1.y; i < r.p2.y+1; ++i) {
        for (int j = r.p1.x; j < r.p2.x + 1; ++j) {
            double val = *(data + (y+i) * ii.step[0] / sizeof (double) + (x + i) * ii.step[0] / sizeof (double) + lbp_offset+lbp_chan);
            if (val == lbp_pattern) {
                result += 1.0;
            }
        }
    }
    return result / (cols * rows);
}

double Feature::compute_binary_pattern(const RowVector &r, int channel_offset, int feature_offset) const
{
    // compute the lbp stuff
    // count the number of lbp features
    if (_type == FIRST_ORDER) {
        // first order...
        return compute_lbp_count(r, _rect1, _channel1 - channel_offset, feature_offset, *_config);
    } else {
        // second order...
        double val1 = compute_lbp_count(r, _rect1, _channel1 - channel_offset, feature_offset, *_config);
        double val2 = compute_lbp_count(r, _rect2, _channel2 - channel_offset, feature_offset, *_config);
        return (val1 - val2 + 1.0) / 2.0;
    }
}

double Feature::compute_binary_pattern(const cv::Mat &ii, int x, int y, int channel_offset, int feature_offset) const
{
    // compute the lbp stuff
    // count the number of lbp features
    if (_type == FIRST_ORDER) {
        // first order...
        return compute_lbp_count(ii, x, y, _rect1, _channel1 - channel_offset, channel_offset+1, *_config);
    } else {
        // second order...
        double val1 = compute_lbp_count(ii, x, y, _rect1, _channel1 - channel_offset, channel_offset+1, *_config);
        double val2 = compute_lbp_count(ii, x, y, _rect2, _channel2 - channel_offset, channel_offset+1, *_config);
        return (val1 - val2 + 1.0) / 2.0;
    }
}

double Feature::compute(const RowVector &r) const
{
    const int hog_channels_start = _config->hog_channels_start();
    const int hog_channels_end   = _config->hog_channels_end();
    const int int_channels_start = _config->integral_channels_start();
    const int int_channels_end   = _config->integral_channels_end();
    const int lbp_channels_start = _config->lbp_channels_start();
    const int lbp_channels_end   = _config->lbp_channels_end();
    const int ltp_channels_start = _config->ltp_channels_start();
    const int ltp_channels_end   = _config->ltp_channels_end();
    // we have to handle here several cases. We can have a feature pool 
    // consisting of HOG, LBP and LTP combinations.
    // Hence we need to infere here the correct feature pool for the given feature index
    // given the used features in the pool...
    if (_channel1 >= hog_channels_start && _channel1 < hog_channels_end) 
        return compute_hog(r);
    if (_channel1 >= int_channels_start && _channel1 < int_channels_end) 
        return compute_int(r, 0);
    if (_channel1 >= lbp_channels_start && _channel1 < lbp_channels_end)
        return compute_binary_pattern(r, lbp_channels_start, _config->lbp_channel_offset());
    if (_channel1 >= ltp_channels_start && _channel1 < ltp_channels_end)
        return compute_binary_pattern(r, ltp_channels_start, _config->ltp_channel_offset());
    assert(false);
}

double Feature::compute(const cv::Mat &ii, int x, int y) const
{
    const int hog_channels_start = _config->hog_channels_start();
    const int hog_channels_end   = _config->hog_channels_end();
    const int int_channels_start = _config->integral_channels_start();
    const int int_channels_end   = _config->integral_channels_end();
    const int lbp_channels_start = _config->lbp_channels_start();
    const int lbp_channels_end   = _config->lbp_channels_end();
    const int ltp_channels_start = _config->ltp_channels_start();
    const int ltp_channels_end   = _config->ltp_channels_end();
    // we have to handle here several cases. We can have a feature pool 
    // consisting of HOG, LBP and LTP combinations.
    // Hence we need to infere here the correct feature pool for the given feature index
    // given the used features in the pool...
    if (_channel1 >= hog_channels_start && _channel1 < hog_channels_end) 
        return compute_hog(ii, x, y);
    if (_channel1 >= int_channels_start && _channel1 < int_channels_end) 
        return compute_int(ii, x, y, 0);
    if (_channel1 >= lbp_channels_start && _channel1 < lbp_channels_end)
        return compute_binary_pattern(ii, x, y, lbp_channels_start, _config->lbp_channel_offset());
    if (_channel1 >= ltp_channels_start && _channel1 < ltp_channels_end)
        return compute_binary_pattern(ii, x, y, ltp_channels_start, _config->ltp_channel_offset());
    assert(false);
}

std::string Feature::to_string() const
{
    std::stringstream strm;
    strm << _type << " " << _channel1 << " " << _rect1.p1.x << " " << _rect1.p1.y << " " << _rect1.p2.x << " " << _rect1.p2.y;
    if (_type == SECOND_ORDER) {
        strm << " " << _channel2 << " " << _rect2.p1.x << " " << _rect2.p1.y << " " << _rect2.p2.x << " " << _rect2.p2.y;
    }
    return strm.str();
}

Feature Feature::create_rand()
{
    const float frac_second_order = ConfigManager::instance()->fraction_second_order();
    float val = rand() % 100 / 100.0;
    int type = FIRST_ORDER;
    if (val < frac_second_order) {
        type = SECOND_ORDER;
    }

    static std::default_random_engine generator;
    const int nchans = ConfigManager::instance()->hog_channels() + ConfigManager::instance()->integral_channels();
    std::uniform_int_distribution<int> distribution(0, nchans-1);
    if (type == FIRST_ORDER) {
        Rect r = Rect::create_rand();
        int c;
        do {
            c = distribution(generator);
        } while (ConfigManager::instance()->is_channel_ignored(c));
        return Feature(r, c, ConfigManager::instance());
    } else if (type == SECOND_ORDER) {
        int c1;
        do {
            c1 = distribution(generator);
        } while (ConfigManager::instance()->is_channel_ignored(c1));
        int c2 = c1;
        // hogs can mix, others not
        if (c1 < ConfigManager::instance()->hog_channels()) {
            std::uniform_int_distribution<int> distribution(0, ConfigManager::instance()->hog_channels()-1);
            do {
                c2 = distribution(generator);
            } while (ConfigManager::instance()->is_channel_ignored(c2));
        }
        return Feature(Rect::create_rand(), c1, Rect::create_rand(), c2, ConfigManager::instance());
    }
    assert(0);
    return Feature();
}

Feature Feature::parse(const std::string &line)
{
    std::stringstream strm(line);
    int type;
    int x1, x2, y1, y2;
    int c1, c2;
    strm >> type;
    strm >> c1;
    strm >> x1;
    strm >> y1;
    strm >> x2;
    strm >> y2;

    Rect r1(Point(x1,y1), Point(x2,y2));

    if (type == SECOND_ORDER) {
        strm >> c2;
        strm >> x1;
        strm >> y1;
        strm >> x2;
        strm >> y2;

        Rect r2(Point(x1,y1), Point(x2,y2));
        return Feature(r1, c1, r2, c2, ConfigManager::instance());
    } else if (type == FIRST_ORDER) {
        return Feature(r1,c1, ConfigManager::instance());
    } else {
        return Feature(); // leafes
    }
}
