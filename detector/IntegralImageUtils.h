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
#ifndef INTEGRALIMAGEUTILS_H
#define INTEGRALIMAGEUTILS_H

#include "DetectorCommon.h"

#include "ConfigManager.h"

#include <Eigen/Core>
#include <boost/archive/xml_oarchive.hpp>
#include <opencv2/core/core.hpp>
#include <sstream>


namespace Detector {

/**
 * This class represents a point with a x and y coordinate
 */
class Point 
{
public:
    Point(int px, int py): x(px), y(py) {}
    Point(): x(0), y(0) {}
    ~Point() {}

    bool operator == (const Point &rhs) const { return x == rhs.x && y == rhs.y; }
    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(x);
        ar & BOOST_SERIALIZATION_NVP(y);
    }

    std::string to_string() const 
    { 
        std::stringstream ss; 
        ss << x << " " << y;
        return ss.str();
    }


    int x, y;
};

/**
 * This class represents the rectangular block cell.
 */
class Rect 
{
public:
    static Rect create_rand(); 
    Rect(const Point &pt1, const Point &pt2): p1(pt1), p2(pt2) {}
    Rect() {}
    ~Rect() {};

    inline bool operator == (const Rect &rhs) const 
    {
        return p1 == rhs.p1 && p2 == rhs.p2;
    }

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(p1);
        ar & BOOST_SERIALIZATION_NVP(p2);
    }

    const std::string to_string() const { return p1.to_string() + " " + p2.to_string(); }
    
    Point p1;
    Point p2;
};

std::istream& operator >> (std::istream &is, Point &p);
std::ostream& operator << (std::ostream &os, const Point &p);
std::istream& operator >> (std::istream &is, Rect &r);
std::ostream& operator << (std::ostream &os, const Rect &r);

static inline double 
norm_channels(double *result, 
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


inline
double query_ii(const RowVector &features, const int channel, const Rect &rect, const int feature_width, const int feature_height)
{
    const int ii_w = feature_width + 1;
    const int ii_h = feature_height + 1;
    int pic_start = channel * (ii_w) * (ii_h);
    int idx1 = pic_start + (rect.p2.y+1)*(ii_w) + rect.p2.x+1;
    int idx2 = pic_start + (rect.p1.y)*(ii_w) + rect.p1.x;
    int idx3 = pic_start + (rect.p2.y+1)*(ii_w) + rect.p1.x;
    int idx4 = pic_start + (rect.p1.y)*(ii_w) + rect.p2.x+1;
    return features(idx1) - features(idx3) - features(idx4) + features(idx2);
}

inline
double query_ii(const cv::Mat &ii, const int x, const int y, const int channel, const Rect &rect)
{
    const double *ptr = ii.ptr<double>(0,0);
    const int s0 = ii.step[0] / sizeof (double);
    const int s1 = ii.step[1] / sizeof (double);
    const double v1 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p2.x + 1) + channel);
    const double v2 = *(ptr + s0 * (y + rect.p1.y)   + s1 * (x + rect.p2.x + 1) + channel);
    const double v3 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p1.x) + channel);
    const double v4 = *(ptr + s0 * (y + rect.p1.y)   + s1 * (x + rect.p1.x) + channel);

    return v1 - v2 - v3 + v4;
}

inline
double query_ii(const std::vector<cv::Mat> &imgs, const int x, const int y, const int channel, const Rect &rect)
{
    const cv::Mat &ii = imgs[channel];
    const double *ptr = ii.ptr<double>(0,0);
    const int s0 = ii.step[0] / sizeof (double);
    const int s1 = ii.step[1] / sizeof (double);
    const double v1 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p2.x + 1));
    const double v2 = *(ptr + s0 * (y + rect.p1.y)   + s1 * (x + rect.p2.x + 1));
    const double v3 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p1.x));
    const double v4 = *(ptr + s0 * (y + rect.p1.y)   + s1 * (x + rect.p1.x));

    return v1 - v2 - v3 + v4;
}

inline double 
query_and_norm_l2_clamp_ii(
    const std::vector<cv::Mat> &img, 
    const int x, const int y, 
    const int channel, 
    const Rect &rect, 
    const int hog_start,
    const int hog_end,
    const int ii_w,
    const int ii_h,
    const double clamp_value = 0.2)
{
    const int hog_channels = hog_end - hog_start;
    std::vector<double> result(hog_channels);

    for (int i = 0; i < hog_channels; i++) {
        const cv::Mat &ii = img[i + hog_start];
        const size_t s0 = ii.step[0] / sizeof (double);
        const size_t s1 = ii.step[1] / sizeof (double);

        const double *ptr = ii.ptr<double>(0,0);
        const double v1 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p2.x + 1));
        const double v2 = *(ptr + s0 * (y + rect.p1.y)   + s1 * (x + rect.p2.x + 1));
        const double v3 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p1.x));
        const double v4 = *(ptr + s0 * (y + rect.p1.y)   + s1 * (x + rect.p1.x));
        result[i] = v1 - v2 - v3 + v4;
    }

    return norm_channels(&result[0], channel, hog_channels, clamp_value);
}

inline double 
query_and_norm_l2_clamp_ii(
    const cv::Mat &ii, 
    const int x, const int y, 
    const int channel, 
    const Rect &rect, 
    const int hog_start,
    const int hog_end,
    const int ii_w,
    const int ii_h,
    const double clamp_value = 0.2)
{
    const double *ptr = ii.ptr<double>(0,0);
    const int hog_channels = hog_end - hog_start;
    std::vector<double> result(hog_channels);
    const size_t s0 = ii.step[0] / sizeof (double);
    const size_t s1 = ii.step[1] / sizeof (double);

    for (int i = 0; i < hog_channels; i++) {
        const double v1 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p2.x + 1) + i + hog_start);
        const double v2 = *(ptr + s0 * (y + rect.p1.y)   + s1 * (x + rect.p2.x + 1) + i + hog_start);
        const double v3 = *(ptr + s0 * (y + rect.p2.y+1) + s1 * (x + rect.p1.x) + i + hog_start);
        const double v4 = *(ptr + s0 * (y + rect.p1.y)   + s1 * (x + rect.p1.x) + i + hog_start);
        result[i] = v1 - v2 - v3 + v4;
    }

    return norm_channels(&result[0], channel, hog_channels, clamp_value);
}

inline double 
query_and_norm_l2_clamp_ii(
    const RowVector &features, 
    const int channel, 
    const Rect &rect,
    const int hog_start,
    const int hog_end,
    const int ii_w, 
    const int ii_h,
    const double clamp_value = 0.2)
{
    const int hog_channels = hog_end - hog_start;

    std::vector<double> result(hog_channels);
    for (int i = 0; i < hog_channels; i++) {
        int pic_start = (i + hog_start) * (ii_w) * (ii_h);
        int idx1 = pic_start + (rect.p2.y+1)*(ii_w) + rect.p2.x+1;
        int idx2 = pic_start + (rect.p1.y)*(ii_w) + rect.p1.x;
        int idx3 = pic_start + (rect.p2.y+1)*(ii_w) + rect.p1.x;
        int idx4 = pic_start + (rect.p1.y)*(ii_w) + rect.p2.x+1;
        result[i] = features(idx1) - features(idx3) - features(idx4) + features(idx2);
    }
    return norm_channels(&result[0], channel, hog_channels, clamp_value);
}

} // namespace Detector

#endif /* end of include guard: INTEGRALIMAGEUTILS_H */
