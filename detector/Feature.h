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
#ifndef MY_FEATURE_H
#define MY_FEATURE_H

#include <string>
#include <Eigen/Core>
#include "DetectorCommon.h"
#include "ConfigManager.h"

#include "IntegralImageUtils.h"

#include <boost/archive/xml_oarchive.hpp>

#include <opencv2/core/core.hpp>

namespace Detector {
/**
 *  This class represents either a first order or second order feature.
 *  Deprecated...
 */
class Feature 
{
public:
    enum Type {
        FIRST_ORDER = 0,
        SECOND_ORDER,
        NONE
    };
    static Feature parse(const std::string &line);
    Feature(): _rect1(Rect()), _rect2(Rect()), _channel1(-1), _channel2(-1), _type(NONE), _id(-1) {}
    Feature(const Rect &r, int c, std::shared_ptr<ConfigManager> mgr);
    Feature(const Rect &r1, int c1, const Rect &r2, int c2, std::shared_ptr<ConfigManager> mgr);
    ~Feature();

    double compute(const RowVector &r) const;
    double compute(const cv::Mat &ii, int x, int y) const;
    double compute_hog(const RowVector &r) const;
    double compute_hog(const cv::Mat &ii, int x, int y) const;
    double compute_int(const RowVector &r, int offset) const;
    double compute_int(const cv::Mat &ii, int x, int y, int offset) const;
    double compute_binary_pattern(const RowVector &r, int channel_offset, int feature_vector_offset) const;
    double compute_binary_pattern(const cv::Mat &ii, int x, int y, int channel_offset, int feature_vector_offset) const;

    int getType() const { return _type; }
    Rect getRect1() const { return _rect1; }
    Rect getRect2() const { return _rect2; }
    int getChannel1() const { return _channel1; }
    int getChannel2() const { return _channel2; }

    std::string to_string() const;

    int get_id() const { return _id; }
    void set_id(int id) { _id = id; }

    static Feature create_rand();

    bool operator == (const Feature &rhs) const { 
        return _rect1 == rhs._rect1 &&
               _rect2 == rhs._rect2 &&
               _channel1 == rhs._channel1 &&
               _channel2 == rhs._channel2 &&
               _type == rhs._type;
    }

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version)
    {
        ar & BOOST_SERIALIZATION_NVP(_rect1);
        ar & BOOST_SERIALIZATION_NVP(_rect2);
        ar & BOOST_SERIALIZATION_NVP(_channel1);
        ar & BOOST_SERIALIZATION_NVP(_channel2);
        ar & BOOST_SERIALIZATION_NVP(_type);
        ar & BOOST_SERIALIZATION_NVP(_id);
        ar & BOOST_SERIALIZATION_NVP(_config);
    }

    void set_config(std::shared_ptr<ConfigManager> mgr)
    {
        _config = mgr;
    }

private:
    Rect _rect1;
    Rect _rect2;
    int _channel1;
    int _channel2;

    int _type;

    // Feature id
    int _id;

    std::shared_ptr<ConfigManager> _config;
};
}

#endif /* end of include guard: MY_FEATURE_H */
