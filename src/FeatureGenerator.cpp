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
#include <detector/FeatureGenerator.h>
#include <detector/config.h>
#include <detector/ConfigManager.h>

#include <detector/BaseFeature.h>
#include <detector/HogFeature.h>
#include <detector/LbpFeature.h>
#include <detector/IntegralFeature.h>


void Detector::FeatureGenerator::generate_std_hog_features(std::vector<std::shared_ptr<BaseFeature> > &features)
{
    std::shared_ptr<ConfigManager> mgr(ConfigManager::instance());
    const int hog_start     = mgr->hog_channels_start();
    const int hog_end       = mgr->hog_channels_end();
    const int window_width  = mgr->feature_width();
    const int window_height = mgr->feature_height();
    const int norm_channel  = mgr->hog_norm_channel_index();
    const bool l2_norm      = mgr->is_l2_norm();
    for (int i = 0; i < 8; ++i) {
        features.emplace_back(new HogFeature(i, Rect(Point(0, 0), Point(23,1)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(0, 2), Point(23,4)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(0, 5), Point(23,6)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(0, 7), Point(23,9)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(0, 10), Point(23,11)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));

        features.emplace_back(new HogFeature(i, Rect(Point(0, 0),  Point(23,1)),  hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(0, 2),  Point(23,9)),  hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(0, 10), Point(23,11)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));

        features.emplace_back(new HogFeature(i, Rect(Point(0, 0), Point(23,11)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));

        features.emplace_back(new HogFeature(i, Rect(Point(0, 0), Point(11,11)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(12, 0), Point(23,11)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));

        features.emplace_back(new HogFeature(i, Rect(Point(0, 0), Point(7,11)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(8, 0), Point(15,11)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
        features.emplace_back(new HogFeature(i, Rect(Point(16, 0), Point(23,11)), hog_start, hog_end, window_width, window_height, norm_channel, l2_norm));
    }
}


void Detector::FeatureGenerator::generate_std_lbp_features(std::vector<std::shared_ptr<BaseFeature> > &features)
{
    std::shared_ptr<ConfigManager> mgr(ConfigManager::instance());
    for (int i = ConfigManager::instance()->lbp_channels_start(); i < ConfigManager::instance()->lbp_channels_end(); ++i) {
        //features.push_back(Feature(Rect(Point(0, 0), Point(23,1)), i));
        //features.push_back(Feature(Rect(Point(0, 2), Point(23,4)), i));
        //features.push_back(Feature(Rect(Point(0, 5), Point(23,6)), i));
        //features.push_back(Feature(Rect(Point(0, 7), Point(23,9)), i));
        //features.push_back(Feature(Rect(Point(0, 10), Point(23,11)), i));

        //features.push_back(Feature(Rect(Point(0, 0),  Point(23,1)), i));
        //features.push_back(Feature(Rect(Point(0, 2),  Point(23,9)), i));
        //features.push_back(Feature(Rect(Point(0, 10), Point(23,11)), i));

        features.emplace_back(new LbpFeature(i, Rect(Point(0, 0), Point(23,11)), 
            mgr->lbp_channels_start(), mgr->lbp_channels_end(), mgr->feature_width(), mgr->feature_height()));

        //features.push_back(Feature(Rect(Point(0, 0), Point(11,11)), i));
        //features.push_back(Feature(Rect(Point(12, 0), Point(23,11)), i));

        //features.push_back(Feature(Rect(Point(0, 0), Point(7,11)), i));
        //features.push_back(Feature(Rect(Point(8, 0), Point(15,11)), i));
        //features.push_back(Feature(Rect(Point(16, 0), Point(23,11)), i));
    }
}

void Detector::FeatureGenerator::generate_std_ltp_features(std::vector<std::shared_ptr<BaseFeature> > &features)
{
    std::shared_ptr<ConfigManager> mgr(ConfigManager::instance());
    const int lbp_start     = mgr->ltp_channels_start();
    const int lbp_end       = mgr->ltp_channels_end();
    const int window_width  = mgr->feature_width();
    const int window_height = mgr->feature_height();
    // we don't calculate everything here -> RAM problems
    for (int i = ConfigManager::instance()->ltp_channels_start(); i < ConfigManager::instance()->ltp_channels_end(); ++i) {
        features.emplace_back(new LbpFeature(i, Rect(Point(0, 0), Point(23,1)), lbp_start, lbp_end, window_width, window_height));
        features.emplace_back(new LbpFeature(i, Rect(Point(0, 2), Point(23,4)), lbp_start, lbp_end, window_width, window_height));
        features.emplace_back(new LbpFeature(i, Rect(Point(0, 5), Point(23,6)), lbp_start, lbp_end, window_width, window_height));
        features.emplace_back(new LbpFeature(i, Rect(Point(0, 7), Point(23,9)), lbp_start, lbp_end, window_width, window_height));
        features.emplace_back(new LbpFeature(i, Rect(Point(0, 10), Point(23,11)), lbp_start, lbp_end, window_width, window_height));

        //features.push_back(Feature(Rect(Point(0, 0),  Point(23,1)), i));
        //features.push_back(Feature(Rect(Point(0, 2),  Point(23,9)), i));
        //features.push_back(Feature(Rect(Point(0, 10), Point(23,11)), i));

        features.emplace_back(new LbpFeature(i, Rect(Point(0, 0), Point(23,11)), lbp_start, lbp_end, window_width, window_height));

        //features.push_back(Feature(Rect(Point(0, 0), Point(11,11)), i));
        //features.push_back(Feature(Rect(Point(12, 0), Point(23,11)), i));

        //features.push_back(Feature(Rect(Point(0, 0), Point(7,11)), i));
        //features.push_back(Feature(Rect(Point(8, 0), Point(15,11)), i));
        //features.push_back(Feature(Rect(Point(16, 0), Point(23,11)), i));
    }
}

static std::shared_ptr<Detector::BaseFeature> create_rand_hog()
{
    static std::default_random_engine generator;
    auto mgr = Detector::ConfigManager::instance();
    const int nchans = mgr->hog_channels() + mgr->integral_channels();
    std::uniform_int_distribution<int> distribution(0, nchans-1);
    Detector::Rect r = Detector::Rect::create_rand();
    int c;
    do {
        c = distribution(generator);
    } while (Detector::ConfigManager::instance()->is_channel_ignored(c));
    if (c >= mgr->hog_channels_start() && c < mgr->hog_channels_end()) {
        return std::make_shared<Detector::HogFeature>(c, r, mgr->hog_channels_start(), 
            mgr->hog_channels_end(), mgr->feature_width(), mgr->feature_height(), mgr->hog_norm_channel_index(), mgr->is_l2_norm());
    } else {
        return std::make_shared<Detector::IntegralFeature>(c, r, 
            mgr->feature_width(), mgr->feature_height());
    }
}

void Detector::FeatureGenerator::generate_random_hog_features(std::vector<std::shared_ptr<BaseFeature> > &features)
{
    features.reserve(ConfigManager::instance()->n_rand_features());
    for (int i = 0; i < ConfigManager::instance()->n_rand_features(); ++i) {
        features.push_back(create_rand_hog());
    }
}

