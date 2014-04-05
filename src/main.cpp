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
#include <detector/DetectorCommon.h>
#include <detector/Adaboost.h>
#include <detector/ConfigManager.h>

#include <iostream>
#include <fstream>
#include <signal.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>

#include <detector/Node.h>
#include <detector/FeatureGenerator.h>

namespace po = boost::program_options;
namespace fs = boost::filesystem;
using namespace Detector;

void exit_handler(int s) {
    exit(1);
}

void readMatrixDimensions(const std::string &filename, int &rows, int &cols)
{
    rows = 0;
    cols = 0;
    std::ifstream is(filename.c_str());
    boost::iostreams::filtering_istream ifs; //(is);

    if (filename.substr(filename.size()-3) == ".gz") {
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

static inline 
Matrix compute_iis(const Matrix &features)
{
    std::shared_ptr<ConfigManager> cfg(ConfigManager::instance());
    const int fvec_size = cfg->hog_size() + cfg->integral_size() + cfg->lbp_size() + cfg->ltp_size();
    const int ii_rows = cfg->ii_feature_height();
    const int ii_cols = cfg->ii_feature_width();
    const int rows = cfg->feature_height();
    const int cols = cfg->feature_width();
    
    Matrix result(features.rows(), fvec_size);
    for (int f = 0; f < features.rows(); ++f) {
        int c = 0;
        // -------------------- HOG PART ------------------------------------
        for (c = 0; c < cfg->hog_channels(); ++c) {
            Matrix channel = Matrix::Zero(ii_rows,ii_cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    //float factor = c != cfg->hog_channels() ? 1.0 : 8.0;
                    channel(i+1,j+1) = features(f, c*rows*cols+i*cols+j);// * factor;
                }
            }

            // create the integral image
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    channel(i+1,j+1) = channel(i+1,j+1) + channel(i,j+1) + channel(i+1,j) - channel(i,j);
                }
            }

            // write it into the result matrix
            for (int i = 0; i < ii_rows; ++i) {
                for (int j = 0; j < ii_cols; ++j) {
                    result(f, c*ii_cols*ii_rows+i*ii_cols+j) = channel(i,j);
                }
            }
        }

        // --------------------- Integral Channel Part ----------------------
        for (int ic_chan = 0; ic_chan < cfg->integral_channels(); ++ic_chan) {
            Matrix channel = Matrix::Zero(ii_rows,ii_cols);
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    channel(i+1,j+1) = features(f, c*rows*cols+i*cols+j);
                }
            }

            // create the integral image
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    channel(i+1,j+1) = channel(i+1,j+1) + channel(i,j+1) + channel(i+1,j) - channel(i,j);
                }
            }
            
            // write it into the result matrix
            for (int i = 0; i < ii_rows; ++i) {
                for (int j = 0; j < ii_cols; ++j) {
                    result(f, c*ii_cols*ii_rows+i*ii_cols+j) = channel(i,j);
                }
            }
            ++c;
        }
        // -------------------- LBP PART ------------------------------------
        for (int lbp_chan = 0; lbp_chan < cfg->lbp_histograms(); ++lbp_chan) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int result_idx = cfg->lbp_channel_offset() + rows*cols*lbp_chan+cols*i+j;
                    result(f,result_idx) = features(f, c*rows*cols+i*cols+j);
                }
            }
            ++c;
        }
        // -------------------- LTP PART ------------------------------------
        for (int ltp_chan = 0; ltp_chan < cfg->ltp_histograms(); ++ltp_chan) {
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    int result_idx = cfg->ltp_channel_offset() + rows*cols*ltp_chan+cols*i+j;
                    result(f, result_idx) = features(f, c*rows*cols+i*cols+j);
                }
            }
            ++c;
        }
    }
    return result;
}


void readMatrixAndComputeIIs(const std::string &filename, Matrix &result, Matrix &labels)
{
    int rows, cols;
    std::shared_ptr<ConfigManager> cfg(ConfigManager::instance());
    const int fvec_size = cfg->hog_size() + cfg->integral_size() + 
        cfg->lbp_size() + cfg->ltp_size();
    readMatrixDimensions(filename, rows, cols);
    result = Matrix(rows, fvec_size);
    labels = Matrix(rows, 1);

    std::ifstream is(filename.c_str());
    boost::iostreams::filtering_istream ifs;
    if (filename.substr(filename.size() - 3) == ".gz") {
        ifs.push(boost::iostreams::gzip_decompressor());
    }
    ifs.push(is);
    //ifs.rdbuf()->pubsetbuf(buf_in, 1);
    std::string line;
    int i = 0, j = 0;
    try {
        while (std::getline(ifs, line)) {
            std::istringstream stream(line);
            std::string part;
            j = 0;
            Matrix feature_row(1,cols - 1);
            float label = -1.0f;
            while (std::getline(stream, part, ',')) {
                double val = ::atof(part.c_str());
                //std::istringstream(part) >> val;
                if (j > 0) {
                    feature_row(0,j-1) = val;
                } else {
                    label = val;
                }
                j++;
            }

            result.row(i) = compute_iis(feature_row);
            labels(i,0) = label;
            i++;
        }
    } catch (...) {
        std::cout << "Error in line: " << i << " col: " << j << std::endl;
        throw;
    }
}

void train_and_validate(Adaboost &boost, int min_sample_count)
{
    std::vector<std::shared_ptr<BaseFeature> > features;
    if (ConfigManager::instance()->feature_file() == "" || 
        !fs::exists(ConfigManager::instance()->feature_file())) {

        if (ConfigManager::instance()->generate_std_hog()) {
            FeatureGenerator().generate_std_hog_features(features);
        }
        if (ConfigManager::instance()->generate_std_lbp()) {
            FeatureGenerator().generate_std_lbp_features(features);
        }
        if (ConfigManager::instance()->generate_std_ltp()) {
            FeatureGenerator().generate_std_ltp_features(features);
        }
        if (ConfigManager::instance()->generate_random_hog()) {
            FeatureGenerator().generate_random_hog_features(features);
        }
        for (size_t i = 0; i < features.size(); i++) {
            features[i]->set_id(int(i));
        }
    }
    boost.set_features(features);
    if (ConfigManager::instance()->feature_file() != "" && !fs::exists(ConfigManager::instance()->feature_file())) {
        std::string output_file = ConfigManager::instance()->feature_file();
        boost.dump_features(output_file);
    } else if (fs::exists(ConfigManager::instance()->feature_file())) {
        boost.load_features(ConfigManager::instance()->feature_file());
    }
    
    // read the data
    Matrix data, labels, featuresTest, labelsTest;
    std::cout << "Reading data..." << std::endl;
    readMatrixAndComputeIIs(ConfigManager::instance()->train_file(), data, labels);
    readMatrixAndComputeIIs(ConfigManager::instance()->validation_file(), 
            featuresTest, labelsTest);

    std::cout << "Training..." << std::endl;
    boost.train(
        data,
        labels,
        min_sample_count,
        featuresTest,
        labelsTest);
    std::cout << "Trained!" << std::endl;

    std::ofstream ofs(ConfigManager::instance()->output_validation());

    // test on the data and save results
    int errs = 0;
    int fp = 0;
    int fn = 0;
    int total_pos = 0;
    int total_neg = 0;
    int true_pos = 0;
    int pos_classified = 0;
    for (int s = 0; s < featuresTest.rows(); ++s) {
        //float result = (boost.predict(featuresTest.row(s))) > 0 ? 1 : -1;
        float raw = (boost.predict(featuresTest.row(s), Adaboost::LAZY));
        //float result = (boost.predict(featuresTest.row(s))) > 0.5 ? 1 : -1;
        float result = raw > 0 ? 1 : -1;
        if (result > 0)
            pos_classified++;
        if (labelsTest(s,0) > 0) {
            total_pos++;
            if (result > 0) {
                true_pos++;
            }
        } else {
            total_neg++;
        }
        if (result != labelsTest(s,0)) {
            ++errs;
            if (result <= 0.0) 
                fn++;
            else
                fp++;
        }

        ofs << raw << "," << labelsTest(s,0) << std::endl;
    }
    std::cout << "PRED FIN" << std::endl;
    //std::cout << "Train ERR: " << (float)trainErrs/labels.rows() << " cases: " << labels.rows() << std::endl;
    std::cout << "Test ERR: " << (float)errs/labelsTest.rows() << " cases: " << featuresTest.rows() << " fp: " << fp << " fn: " << fn << std::endl;
    std::cout << "Recall: " << float (true_pos) / float (total_pos) << " Precision: " << float (true_pos) / float (pos_classified) << std::endl;

    {
        std::ofstream ofs_model(ConfigManager::instance()->output());
        boost::archive::text_oarchive oa(ofs_model);
        oa << boost::serialization::make_nvp("model", boost);
    }
}

int main(int argc, char* argv[]) {
    
    po::options_description desc("Allowed options");
    desc.add_options()
        ("help,h", "print this help message")
        ("config,c", po::value<std::string>()->required(), 
         "Configuration of the feature pool")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
        std::cout << desc << std::endl;
        return 0;
    }
    po::notify(vm);

    ConfigManager::instance().reset(new ConfigManager(vm["config"].as<std::string>()));

    int n_trees = 200;
    int max_depth = 5;
    int min_sample_count = 0;

    if (ConfigManager::instance()->num_trees() > 0) 
        n_trees = ConfigManager::instance()->num_trees();
    if (ConfigManager::instance()->max_depth() > 0) 
        max_depth = ConfigManager::instance()->max_depth();
    if (ConfigManager::instance()->min_sample_count() > 0) 
        min_sample_count = ConfigManager::instance()->min_sample_count();

    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);

    Adaboost boost(Adaboost::GENTLE, n_trees, max_depth);
    train_and_validate(boost, min_sample_count);
    std::cout << "finish" << std::endl;
    return 0;
} 

