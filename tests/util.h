#ifndef TEST_UTIL_H
#define TEST_UTIL_H

#include <string>
#include <stdio.h>

#include <detector/Adaboost.h>
#include <detector/ConfigManager.h>


namespace Detector {
    void readMatrixDimensions(const std::string &filename, int &rows, int &cols);
    Matrix readMatrix(const std::string &filename);
}

#endif /* end of include guard: TEST_ */
