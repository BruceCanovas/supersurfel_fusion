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


#ifndef SUPERSURFELS_HPP
#define SUPERSURFELS_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <supersurfel_fusion/matrix_types.h>


namespace supersurfel_fusion
{

struct Supersurfels
{
    thrust::device_vector<float3> positions; // position
    thrust::device_vector<float3> colors; // color
    thrust::device_vector<int2> stamps; // timestamps init + current
    thrust::device_vector<Mat33> orientations; // patches orientations
    thrust::device_vector<Cov3> shapes; // shapes of point cloud segments
    thrust::device_vector<float2> dims; // lengths, widths
    thrust::device_vector<float> confidences;

    typedef thrust::zip_iterator<thrust::tuple<thrust::device_vector<float3>::iterator,
                                               thrust::device_vector<float3>::iterator,
                                               thrust::device_vector<int2>::iterator,
                                               thrust::device_vector<Mat33>::iterator,
                                               thrust::device_vector<Cov3>::iterator,
                                               thrust::device_vector<float2>::iterator,
                                               thrust::device_vector<float>::iterator>> iterator;

    iterator begin()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(positions.begin(),
                                                            colors.begin(),
                                                            stamps.begin(),
                                                            orientations.begin(),
                                                            shapes.begin(),
                                                            dims.begin(),
                                                            confidences.begin()));
    }

    iterator end()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(positions.end(),
                                                            colors.end(),
                                                            stamps.end(),
                                                            orientations.end(),
                                                            shapes.end(),
                                                            dims.end(),
                                                            confidences.end()));
    }

    void clear()
    {
        positions.clear();
        colors.clear();
        stamps.clear();
        orientations.clear();
        shapes.clear();
        dims.clear();
        confidences.clear();
    }

    void memset(size_t length)
    {
        cudaMemset(thrust::raw_pointer_cast(&positions[0]), 0.0f, length*sizeof(float3));
        cudaMemset(thrust::raw_pointer_cast(&colors[0]), 0.0f, length*sizeof(float3));
        cudaMemset(thrust::raw_pointer_cast(&stamps[0]), 0, length*sizeof(int2));
        cudaMemset(thrust::raw_pointer_cast(&orientations[0]), 0.0f, length*sizeof(Mat33));
        cudaMemset(thrust::raw_pointer_cast(&shapes[0]), 0.0f, length*sizeof(Cov3));
        cudaMemset(thrust::raw_pointer_cast(&dims[0]), 0.0f, length*sizeof(float2));
        cudaMemset(thrust::raw_pointer_cast(&confidences[0]), 0.0f, length*sizeof(float));
    }
}; // struct Supersurfels

} // namespace supersurfel_fusion

#endif // SUPERSURFELS_HPP
