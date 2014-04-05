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
#include <detector/IntegralFeature.h>

#include <detector/config.h>

Detector::IntegralFeature::IntegralFeature(int chan, const Rect &r, int w, int h)
: _channel(chan), _rect(r), _feature_width(w), _feature_height(h)
{
}

double Detector::IntegralFeature::compute(const RowVector &r) const
{
    const double norm = _feature_width * _feature_height;
    return query_ii(r, _channel, _rect, _feature_width, _feature_height) / norm;
}

double Detector::IntegralFeature::compute(const cv::Mat &ii, int x, int y) const
{
    const double norm = _feature_width * _feature_height;
    return query_ii(ii, x, y, _channel, _rect) / norm;
}

double Detector::IntegralFeature::compute(const std::vector<cv::Mat> &ii, int x, int y) const
{
    const double norm = _feature_width * _feature_height;
    return query_ii(ii, x, y, _channel, _rect) / norm;
}

std::string Detector::IntegralFeature::to_string() const
{
    std::stringstream ss;
    ss << FEATURE_INT       << " " << _id << " " << _channel << " " 
       << _rect.to_string() << " " << _feature_width << " " 
       << _feature_height;
    return ss.str();
}

std::shared_ptr<Detector::BaseFeature> Detector::IntegralFeature::parse(const std::string &line) throw(std::runtime_error)
{
    std::stringstream ss(line);
    int type;
    ss >> type;
    if (type != FEATURE_INT) 
        throw std::runtime_error("IntegralFeature::parse invalid type");

    int id, channel, window_width, window_height;
    Rect rect;
    ss >> id >> channel >> rect 
       >> window_width 
       >> window_height;
    std::shared_ptr<Detector::IntegralFeature> res(std::make_shared<Detector::IntegralFeature>(
        channel, rect, 
        window_width, 
        window_height 
    ));
    res->set_id(id);
    return res;
}

