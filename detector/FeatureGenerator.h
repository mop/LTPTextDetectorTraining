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
#ifndef FEATUREGENERATOR_H

#define FEATUREGENERATOR_H

#include <vector>
#include <memory>
#include "BaseFeature.h"

/**
 *  This class is responsible for generating the feature sets
 */
namespace Detector {
class FeatureGenerator 
{
public:
    FeatureGenerator() {}
    ~FeatureGenerator() {}

    void generate_std_hog_features(std::vector<std::shared_ptr<BaseFeature> > &features);
    void generate_std_lbp_features(std::vector<std::shared_ptr<BaseFeature> > &features);
    void generate_std_ltp_features(std::vector<std::shared_ptr<BaseFeature> > &features);
    void generate_random_hog_features(std::vector<std::shared_ptr<BaseFeature> > &features);

private:
};
}

#endif /* end of include guard: FEATUREGENERATOR_H */
