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
#ifndef HOGFEATURE_H
#define HOGFEATURE_H

#include "BaseFeature.h"
#include "IntegralImageUtils.h"

#include <memory>
#include <stdexcept>


namespace Detector {

class ConfigManager;

/**
 * This class represents a single channel HOG feature.
 */
class HogFeature : public BaseFeature
{
public:
    HogFeature() {}
    /**
     *  @param channel is the absolute index into the image channel
     */
    HogFeature(
        int channel, 
        const Rect &r, 
        int hog_channels_start,
        int hog_channels_end,   // last HOG channel excluding normalization
        int window_width,
        int window_height,
        int norm_channel = -1,  // no normalization
        bool l2_norm = false);
    virtual ~HogFeature() {}

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
        ar & _hog_channels_start;
        ar & _hog_channels_end;
        ar & _window_width;
        ar & _window_height;
        ar & _norm_channel;
        ar & _l2_norm;
    }

    virtual std::string to_string() const;

    static std::shared_ptr<BaseFeature> parse(const std::string &line) throw (std::runtime_error);

private:
    //! The HOG channel
    int _channel;
    //! The position of the feature
    Rect _rect;

    //! First HOG channel index
    int _hog_channels_start;
    //! Last HOG channel index
    int _hog_channels_end;
    //! The width of the window
    int _window_width;
    //! The height of the window
    int _window_height;

    //! The normalization channel
    int _norm_channel;
    //! Flag which indicates whether this is a l2 norm
    bool _l2_norm;
};

} // namespace Detector

#endif /* end of include guard: HOGFEATURE_H */
