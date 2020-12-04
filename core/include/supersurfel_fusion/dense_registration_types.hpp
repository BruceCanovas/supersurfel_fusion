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


#ifndef DENSE_REGISTRATION_TYPES_HPP
#define DENSE_REGISTRATION_TYPES_HPP

#include <cuda.h>
#include <cuda_runtime.h>


namespace supersurfel_fusion
{

struct add_float3
{
    __device__ float3 operator()(const float3& a, const float3& b) const
    {
        return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
    }
};

struct dist2_to_center // sum of squared distances to center
{
    const float3 center;

    dist2_to_center(float3 c)
        : center(c)
    {}

    __device__ float operator()(const float3& a) const
    {
        float3 diff = make_float3(a.x - center.x, a.y - center.y, a.z - center.z);

        return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
    }

};

struct MotionTrackingData
{
    float JtJ[21];
    float Jtr[6];
    float r, inliers;
};

} // namespace supersurfel_fusion

#endif // DENSE_REGISTRATION_TYPES_HPP
