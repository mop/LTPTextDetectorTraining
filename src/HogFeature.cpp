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
#include <detector/HogFeature.h>

#include <detector/IntegralImageUtils.h>

#include <detector/config.h>

namespace Detector {

HogFeature::HogFeature(
    int channel, const Rect &r, 
    int hog_start, int hog_end,
    int feature_width, int feature_height, 
    int norm_channel, bool l2_norm)
: _channel(channel), _rect(r), 
  _hog_channels_start(hog_start), 
  _hog_channels_end(hog_end),
  _window_width(feature_width), 
  _window_height(feature_height), 
  _norm_channel(norm_channel), _l2_norm(l2_norm)
{
}

double HogFeature::compute(const RowVector &r) const
{
    if (_l2_norm)
        return query_and_norm_l2_clamp_ii(r, 
            _channel, _rect, _hog_channels_start, 
            _hog_channels_end, _window_width, _window_height);

    double val = query_ii(r, _channel, _rect, _window_width, _window_height);
    if (_norm_channel < 0)
        return val;

    const Rect whole_patch(Point(0,0), Point(_window_width-1, _window_height-1));
    double norm = query_ii(r, _norm_channel, whole_patch, _window_width, _window_height);
    norm = std::max(norm, val);
    return (val < 1e-5 || norm < 1e-5) ? 0.0 : sqrt(val / (norm + 1e-5));
}

double HogFeature::compute(const cv::Mat &ii, int x, int y) const
{
    if (_l2_norm)
        return query_and_norm_l2_clamp_ii(ii, x, y, 
            _channel, _rect, _hog_channels_start, _hog_channels_end, _window_width, _window_height);

    double val = query_ii(ii, x, y, _channel, _rect);
    if (_norm_channel < 0)
        return val;

    const Rect whole_patch(Point(0,0), Point(_window_width-1, _window_height-1));
    double norm = std::max(val, query_ii(ii, x, y, _norm_channel, whole_patch));
    return (val < 1e-5 || norm < 1e-5) ? 0.0 : sqrt(val / (norm + 1e-5));
}

double HogFeature::compute(const std::vector<cv::Mat> &ii, int x, int y) const
{
    if (_l2_norm)
        return query_and_norm_l2_clamp_ii(ii, x, y, 
            _channel, _rect, _hog_channels_start, _hog_channels_end, _window_width, _window_height);

    double val = query_ii(ii, x, y, _channel, _rect);
    if (_norm_channel < 0)
        return val;

    const Rect whole_patch(Point(0,0), Point(_window_width-1, _window_height-1));
    double norm = std::max(val, query_ii(ii, x, y, _norm_channel, whole_patch));
    return (val < 1e-5 || norm < 1e-5) ? 0.0 : sqrt(val / (norm + 1e-5));
}

std::string HogFeature::to_string() const
{
    std::stringstream ss;
    ss << FEATURE_HOG       << " " << _id << " " << _channel << " " 
       << _rect.to_string() << " " << _hog_channels_start << " " 
       << _hog_channels_end << " " << _window_width << " " 
       << _window_height    << " " << _norm_channel << " " 
       << _l2_norm;
    return ss.str();
}

std::shared_ptr<BaseFeature> HogFeature::parse(const std::string &line) throw (std::runtime_error)
{
    std::stringstream ss(line);
    int type;
    ss >> type;
    if (type != FEATURE_HOG) 
        throw std::runtime_error("HogFeature::parse invalid type");

    int id, channel, hog_channels_start, hog_channels_end, window_width, window_height, norm_channel;
    bool l2_norm;
    Rect rect;
    ss >> id >> channel >> rect >> hog_channels_start
       >> hog_channels_end >> window_width 
       >> window_height    >> norm_channel
       >> l2_norm;
    std::shared_ptr<HogFeature> res(std::make_shared<HogFeature>(
        channel, rect, hog_channels_start, hog_channels_end, window_width, 
        window_height, norm_channel, l2_norm
    ));
    res->set_id(id);
    return res;
}
    
} /* namespace Detector */
