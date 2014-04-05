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
#include <detector/LbpFeature.h>

#include <detector/config.h>
#include <x86intrin.h>


namespace Detector {

LbpFeature::LbpFeature(int chan, const Rect &r, int lbp_start, int lbp_end, int window_width, int window_height)
: _channel(chan), _rect(r), _lbp_start(lbp_start), _lbp_end(lbp_end), 
  _window_width(window_width), _window_height(window_height)
{
}

static inline double compute_lbp_count(const RowVector &row, const Rect &r, int lbp_pattern, int lbp_offset, int width, int height)
{
    int lbp_chan = lbp_pattern / 256;
    lbp_pattern  = lbp_pattern % 256;

    double result = 0.0;
    for (int i = r.p1.y; i < r.p2.y+1; ++i) {
        for (int j = r.p1.x; j < r.p2.x + 1; ++j) {
            int idx = lbp_offset + lbp_chan * (width*height) + i*width + j;
            if (row(0, idx) == lbp_pattern) {
                result += 1.0;
            }
        }
    }
    return result / (width * height);
}

static inline double 
compute_lbp_count(const cv::Mat &ii, int x, int y, const Rect &r, int lbp_pattern, int lbp_offset, int width, int height)
{
    int lbp_chan = lbp_pattern / 256;
    lbp_pattern  = lbp_pattern % 256;

    const double *data = ii.ptr<double>(0,0);
    double result = 0.0;
    const size_t s0 = ii.step[0] / sizeof (double);
    const size_t s1 = ii.step[1] / sizeof (double);
    for (int i = r.p1.y; i < r.p2.y+1; ++i) {
        for (int j = r.p1.x; j < r.p2.x + 1; ++j) {
            const double val = *(data + (y+i) * s0 + (x + j) * s1 + lbp_offset+lbp_chan);
            result += val == lbp_pattern;
        }
    }
    return result / (width * height);
}

static inline double 
compute_lbp_count(const std::vector<cv::Mat> &ii, const int x, const int y, const Rect &r, int lbp_pattern, const int lbp_offset, const int width, const int height)
{
    int lbp_chan = lbp_pattern / 256;
    lbp_pattern  = lbp_pattern % 256;
    const cv::Mat &img = ii[lbp_chan + lbp_offset];

    const double *data = img.ptr<double>(0,0);
    double result = 0.0;
    const size_t s0 = img.step[0] / sizeof (double);
    const size_t s1 = img.step[1] / sizeof (double);
    const int end_y = r.p2.y + 1;
    const int end_x = r.p2.x + 1;
    for (int i = r.p1.y; i < end_y; ++i) {
        for (int j = r.p1.x; j < end_x; ++j) {
            const double val = *(data + (y+i) * s0 + (x + j) * s1);
            result += val == lbp_pattern;
        }
    }
    return result / (width * height);
}

static inline double 
compute_lbp_count_simd(
    const std::vector<cv::Mat> &ii,
    int x, int y, 
    const Rect &r, int lbp_pattern, int lbp_offset, int width, int height)
{
    int lbp_chan = lbp_pattern / 256;
    lbp_pattern  = lbp_pattern % 256;
    const cv::Mat &img = ii[lbp_chan + lbp_offset];

    //const double *data = img.ptr<double>(0,0);
    __m128d pattern = _mm_set1_pd(lbp_pattern);
    __m128d ones    = _mm_set1_pd(1.0);
    constexpr size_t s = sizeof (__m128d) / sizeof (double);
    float single_result = 0.0f;
    const int pad = (r.p2.x + 1 - r.p1.x) % s;
    const int len = ((r.p2.x + 1 - r.p1.x) / s) * s;
    for (int i = r.p1.y; i < r.p2.y+1; ++i) {
        const double* data = img.ptr<double>(i+y, r.p1.x+x);
        __m128d result = _mm_set1_pd(0.0);
        for (int j = 0; j < len; j += s) {
            __m128d dat = _mm_loadu_pd(data + j);
            result = _mm_add_pd(result, _mm_and_pd(ones, _mm_cmp_pd(dat, pattern, _CMP_EQ_OQ)));
        }

        result = _mm_hadd_pd(result, result);
        result = _mm_hadd_pd(result, result);
        double res[s];
        _mm_store_pd(res, result);
        single_result += res[0];
        for (int j = len; j < len + pad; j++) {
            single_result += *(data+j)==lbp_pattern;
        }
    }

    return single_result / (width * height);
}

double LbpFeature::compute(const RowVector &r) const
{
    return compute_lbp_count(r, _rect, _channel - _lbp_start, _lbp_start, _window_width, _window_height);
}

double LbpFeature::compute(const cv::Mat &ii, int x, int y) const
{
    return compute_lbp_count(ii, x, y, _rect, _channel - _lbp_start, _lbp_start, _window_width, _window_height);
}

double LbpFeature::compute(const std::vector<cv::Mat> &ii, int x, int y) const
{
    return compute_lbp_count(ii, x, y, _rect, _channel - _lbp_start, _lbp_start, _window_width, _window_height);
}

std::string LbpFeature::to_string() const
{
    std::stringstream ss;
    ss << FEATURE_LBP       << " " << _id << " " << _channel << " " 
       << _rect.to_string() << " " 
       << _lbp_start << " " << _lbp_end << " " 
       << _window_width << " " << _window_height;
    return ss.str();
}

std::shared_ptr<BaseFeature> LbpFeature::parse(const std::string &line) throw (std::runtime_error)
{
    std::stringstream ss(line);
    int type;
    ss >> type;
    if (type != FEATURE_LBP) 
        throw std::runtime_error("LbpFeature::parse invalid type");

    int id, channel, lbp_channels_start, lbp_channels_end, window_width, window_height;
    Rect rect;
    ss >> id >> channel >> rect 
       >> lbp_channels_start
       >> lbp_channels_end
       >> window_width 
       >> window_height;
    std::shared_ptr<LbpFeature> res(std::make_shared<LbpFeature>(
        channel, rect, 
        lbp_channels_start,
        lbp_channels_end,
        window_width, 
        window_height 
    ));
    res->set_id(id);
    return res;
}
}
