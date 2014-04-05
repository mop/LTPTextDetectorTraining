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
#include <detector/IntegralImageUtils.h>

#include <random>

namespace Detector {

Rect Rect::create_rand() 
{
    // first generate the width and height of the rect
    int max_w = ConfigManager::instance()->max_w();
    int max_h = ConfigManager::instance()->max_h();
    int min_w = ConfigManager::instance()->min_w();
    int min_h = ConfigManager::instance()->min_h();
    int feature_width  = ConfigManager::instance()->feature_width();
    int feature_height = ConfigManager::instance()->feature_height();

    static std::default_random_engine generator;
    std::uniform_int_distribution<int> w_distr(min_w, max_w-1);
    std::uniform_int_distribution<int> h_distr(min_h, max_h-1);
    int w = w_distr(generator);
    int h = h_distr(generator);

    std::uniform_int_distribution<int> x_distr(0, feature_width - w + 1 - 1);
    std::uniform_int_distribution<int> y_distr(0, feature_height - h + 1 - 1);

    // then generate the 'starting point'. We need the point to 
    // be smaller than 24-w, 12-h respectively
    // and bigger than 1+radius and 1+radius
    int cx = x_distr(generator);
    int cy = y_distr(generator);

    assert(cx < feature_width);
    assert(cy < feature_height);
    assert((cx+w-1) < feature_width);
    assert((cy+h-1) < feature_height);
    // top left
    return Rect(
        Point(cx, cy),
        Point(cx+w-1, cy+h-1)
    );
}

std::istream& operator >> (std::istream &is, Point &p)
{
    is >> p.x >> p.y;
    return is;
}

std::ostream& operator << (std::ostream &os, const Point &p)
{
    os << p.x << " " << p.y;
    return os;
}

std::istream& operator >> (std::istream &is, Rect &r)
{
    is >> r.p1 >> r.p2;
    return is;
}

std::ostream& operator << (std::ostream &os, const Rect &r)
{
    os << r.p1 << " " << r.p2;
    return os;
}

}
