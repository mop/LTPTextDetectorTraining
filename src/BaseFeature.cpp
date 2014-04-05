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
#include <detector/BaseFeature.h>

#include <detector/LbpFeature.h>
#include <detector/HogFeature.h>
#include <detector/IntegralFeature.h>
#include <detector/config.h>

#include <sstream>

namespace Detector {

static std::shared_ptr<BaseFeature> BaseFeature::parse(const std::string &line) throw (std::runtime_error)
{
    std::stringstream strm(line);
    int type;
    strm >> type;
    if (!(type >= FEATURE_HOG && type <= FEATURE_LBP)) 
        throw std::runtime_error("Feature in unsupported range");

    typedef std::shared_ptr<BaseFeature> (*parse_function)(const std::string &);
    parse_function funcs[] = {
        HogFeature::parse,
        IntegralFeature::parse,
        LbpFeature::parse,
    };

    return funcs[type](line);
}
    
} /* namespace Detector */
