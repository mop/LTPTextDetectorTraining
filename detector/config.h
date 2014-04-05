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
#ifndef RF_CONFIG_H

#define RF_CONFIG_H

/*
#define FEATURE_W 24
#define FEATURE_H 12

#define II_FEATURE_W 25
#define II_FEATURE_H 13

#define MAX_W 24
#define MAX_H 12
#define MIN_W 4
#define MIN_H 2
#define N_RAND_RECTS 20000
#define FRAC_SECOND_ORDER 0.5


#define HOG_CHANNELS 8
#define NORM_CHANNEL_IDX 8

#define LBP_CHANNELS 8
#define LBP_CHANNEL_IDX 9
#define LBP_OFFSET ((HOG_CHANNELS + 1)*II_FEATURE_W*II_FEATURE_H)
#define NUM_LBP_CHANNELS (LBP_CHANNELS * 256)

#define LTP_CHANNELS 16
#define LTP_CHANNEL_IDX 9
#define LTP_OFFSET ((HOG_CHANNELS + 1)*II_FEATURE_W*II_FEATURE_H)
#define NUM_LTP_CHANNELS (LTP_CHANNELS * 256)

#define NUM_CHANNELS_LBP (HOG_CHANNELS + NUM_LBP_CHANELS)
#define NUM_CHANNELS_LTP (HOG_CHANNELS + NUM_LTP_CHANELS)
//#define NUM_CHANNELS (HOG_CHANNELS)
*/

enum FeatureId {
    FEATURE_HOG = 0,
    FEATURE_INT,
    FEATURE_LBP
};

#endif /* end of include guard: CONFIG_H */
