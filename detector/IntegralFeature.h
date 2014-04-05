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
#ifndef INTEGRALFEATURE_H
#define INTEGRALFEATURE_H

#include "BaseFeature.h"
#include "IntegralImageUtils.h"

namespace Detector {

class IntegralFeature : public BaseFeature
{
public:
    IntegralFeature() {}
    IntegralFeature(int channel, const Rect &r, int feature_width, int feature_height);
    virtual ~IntegralFeature() {}

    //! @see BaseFeature
    virtual double compute(const RowVector &r) const;
    //! @see BaseFeature
    virtual double compute(const cv::Mat &ii, int x, int y) const;
    //! @see BaseFeature
    virtual double compute(const std::vector<cv::Mat> &ii, int x, int y) const;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int version) 
    { 
        ar & boost::serialization::base_object<BaseFeature>(*this);
        ar & _channel;
        ar & _rect;
        ar & _feature_width;
        ar & _feature_height;
    }

    virtual std::string to_string() const;
    static std::shared_ptr<BaseFeature> parse(const std::string &line) throw(std::runtime_error);
private:
    //! The channel of the feature
    int _channel;
    //! The position of the feature
    Rect _rect;
    //! The width of the II feature
    int _feature_width;
    //! The height of the II feature
    int _feature_height;
};

} /* namespace Detector */


#endif /* end of include guard: INTEGRALFEATURE_H */
