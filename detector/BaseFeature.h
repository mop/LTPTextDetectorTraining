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
#ifndef BASEFEATURE_H

#define BASEFEATURE_H

#include "DetectorCommon.h"

#include <opencv2/core/core.hpp>
#include <string>
#include <exception>
#include <memory>
#include <iostream>
#include <vector>

namespace Detector {

/**
 * This class represents a feature which is evaluated on an integral image 
 * on a specific position.
 */
class BaseFeature 
{
public:
    BaseFeature(): _id(-1) {}
    virtual ~BaseFeature() {}

    virtual double compute(const RowVector &r) const = 0;
    virtual double compute(const cv::Mat &ii, int x, int y) const = 0;
    virtual double compute(const std::vector<cv::Mat> &ii, int x, int y) const = 0;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & _id;
    }

    void set_id(int i) { _id = i; }
    int get_id() const { return _id; }

    //! This method creates a human readable csv-like representation of the object. 
    //! This is useful for numpy/scipy/matlab analysis of generated feature pools
    virtual std::string to_string() const = 0;

    static std::shared_ptr<BaseFeature> parse(const std::string &line) throw (std::runtime_error);
protected:
    //! The ID of the feature
    int _id;
};

} /* namespace Detector */


#endif /* end of include guard: BASEFEATURE_H */
