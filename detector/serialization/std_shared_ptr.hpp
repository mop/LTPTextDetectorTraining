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
#ifndef SERIALIZATION_STD_SHARED_PTR_H

#define SERIALIZATION_STD_SHARED_PTR_H

#include <boost/serialization/shared_ptr.hpp>
#include <boost/unordered_map.hpp>
#include <memory>

namespace boost {
namespace serialization {

template<class Archive, class T>
inline void save(
    Archive & ar,
    const std::shared_ptr< T > &t,
    const unsigned int /* file_version */
){
    // The most common cause of trapping here would be serializing
    // something like shared_ptr<int>.  This occurs because int
    // is never tracked by default.  Wrap int in a trackable type
    BOOST_STATIC_ASSERT((tracking_level< T >::value != track_never));
    const T * t_ptr = t.get();
    ar << boost::serialization::make_nvp("px", t_ptr);
}

template<class Archive, class T>
inline void load(
    Archive & ar,
    std::shared_ptr< T > &value,
    const unsigned int /*file_version*/
){
    // The most common cause of trapping here would be serializing
    // something like shared_ptr<int>.  This occurs because int
    // is never tracked by default.  Wrap int in a trackable type
    BOOST_STATIC_ASSERT((tracking_level< T >::value != track_never));
    T* r;
    ar >> boost::serialization::make_nvp("px", r);
    static boost::unordered_map<void*, std::weak_ptr<T> > hash;
    if (hash[r].expired()) {
        value = std::shared_ptr<T>(r);
        hash[r] = value;
    } else {
        value = hash[r].lock();
    }
}

template <class Archive, class T>
inline void serialize(
    Archive & ar,
    std::shared_ptr< T > &t,
    const unsigned int file_version
){
    // correct shared_ptr serialization depends upon object tracking
    // being used.
    BOOST_STATIC_ASSERT(
        boost::serialization::tracking_level< T >::value
        != boost::serialization::track_never
    );
    boost::serialization::split_free(ar, t, file_version);
}
} // boost::serialization
} // boost


#endif /* end of include guard: SERIALIZATION_STD_SHARED_PTR_H */
