/*************************************************************************
* Copyright (C) 2020, Bruce Canovas, Amaury Negre, GIPSA-lab
* This file is part of https://github.com/BruceCanovas/supersurfel_fusion

* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <https://www.gnu.org/licenses/>.
*************************************************************************/


#ifndef FERNS_TYPES_HPP
#define FERNS_TYPES_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <supersurfel_fusion/matrix_types.h>
#include <thrust/host_vector.h>
#include <algorithm>
#include <opencv2/core.hpp>
#include <vector>


namespace supersurfel_fusion
{

struct Fern
{
    uint2 pos;
    uchar3 rgb;
    float depth;
};

struct KeyFrame
{
    int fernId;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    std::vector<cv::Point3f> keypoints3D;
    thrust::host_vector<float3> positions;
    thrust::host_vector<float3> colors;
    thrust::host_vector<Mat33> orientations;
};

struct StampedPose
{
    Mat33 R;
    float3 t;
    int stamp;
};

} // supersurfel_fusion

#endif // FERNS_TYPES_HPP
