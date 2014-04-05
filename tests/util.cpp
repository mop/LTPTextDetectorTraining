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
#include "util.h"

#include <sstream>
#include <iostream>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>

namespace Detector {
void readMatrixDimensions(const std::string &filename, int &rows, int &cols)
{
    rows = 0;
    cols = 0;
    std::ifstream is(filename.c_str());
    boost::iostreams::filtering_istream ifs;
    if (filename.substr(filename.size() - 3) == ".gz") {
        ifs.push(boost::iostreams::gzip_decompressor());
    }
    ifs.push(is);
    bool once = true;
    std::string line;
    while (std::getline(ifs, line)) {
        ++rows;
        if (once) {
            std::istringstream stream(line);
            std::string part;
            while (std::getline(stream, part, ',')) {
                ++cols;
            }
            once = false;
        }
    }
}

Matrix readMatrix(const std::string &filename)
{
    int rows, cols; readMatrixDimensions(filename, rows, cols);
    Matrix result(rows, cols);

    std::ifstream is(filename.c_str());
    boost::iostreams::filtering_istream ifs;
    if (filename.substr(filename.size() - 3) == ".gz") {
        ifs.push(boost::iostreams::gzip_decompressor());
    }
    ifs.push(is);
    std::string line;
    int i = 0, j = 0;
    try {
        while (std::getline(ifs, line)) {
            std::istringstream stream(line);
            std::string part;
            j = 0;
            while (std::getline(stream, part, ',')) {
                double val;
                std::istringstream(part) >> val;
                result(i,j) = val;
                j++;
            }
            i++;
        }
    } catch (...) {
        std::cout << "Error in line: " << i << " col: " << j << std::endl;
        throw;
    }

    return result;
}
}
