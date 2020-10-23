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


#ifndef DEFORMATION_GRAPH_TYPES_HPP
#define DEFORMATION_GRAPH_TYPES_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <supersurfel_fusion/matrix_types.h>
#include <thrust/host_vector.h>
#include <algorithm>


namespace supersurfel_fusion
{

struct GraphNodes
{
    thrust::host_vector<int> stamps;
    thrust::host_vector<float3> positions;
    thrust::host_vector<float3> translations;
    thrust::host_vector<Mat33> rotations;
    thrust::host_vector<int4> neighbours;
    thrust::host_vector<bool> enabled;

    void push_back(int t,
                   float3 const& pos,
                   float3 const& trans,
                   Mat33 const& rot,
                   int4 const& neigh,
                   bool en)
    {
        stamps.push_back(t);
        positions.push_back(pos);
        translations.push_back(trans);
        rotations.push_back(rot);
        neighbours.push_back(neigh);
        enabled.push_back(en);
    }

    void clear()
    {
        positions.clear();
        stamps.clear();
        translations.clear();
        rotations.clear();
        neighbours.clear();
        enabled.clear();
    }

    typedef thrust::zip_iterator<thrust::tuple<thrust::host_vector<int>::iterator,
                                               thrust::host_vector<float3>::iterator,
                                               thrust::host_vector<float3>::iterator,
                                               thrust::host_vector<Mat33>::iterator,
                                               thrust::host_vector<int4>::iterator,
                                               thrust::host_vector<bool>::iterator>> iterator;

    iterator begin()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(stamps.begin(),
                                                            positions.begin(),
                                                            translations.begin(),
                                                            rotations.begin(),
                                                            neighbours.begin(),
                                                            enabled.begin()));
    }

    iterator end()
    {
        return thrust::make_zip_iterator(thrust::make_tuple(stamps.end(),
                                                            positions.end(),
                                                            translations.end(),
                                                            rotations.end(),
                                                            neighbours.end(),
                                                            enabled.end()));
    }
};

struct Constraints
{
    thrust::host_vector<float3> sources;
    thrust::host_vector<float3> targets;
    thrust::host_vector<int> sourceStamps;
    thrust::host_vector<int> targetStamps;
    thrust::host_vector<int> idx;
    thrust::host_vector<bool> pin;

    void clear()
    {
        sources.clear();
        targets.clear();
        sourceStamps.clear();
        targetStamps.clear();
        idx.clear();
        pin.clear();
    }
};

struct VerticesWeights
{
    thrust::host_vector<float4> weights;
    thrust::host_vector<int4>  nodes;

    void sort(int vertex_id)
    {
        int n[4] = {nodes[vertex_id].x, nodes[vertex_id].y, nodes[vertex_id].z, nodes[vertex_id].w};
        float w[4] = {weights[vertex_id].x, weights[vertex_id].y, weights[vertex_id].z, weights[vertex_id].w};

        bool done = false;

        int size = 4;

        while(!done)
        {
            done = true;

            for(int i = 0; i < size - 1; i++)
            {
                if(n[i] > n[i + 1])
                {
                    done = false;
                    std::swap(n[i], n[i + 1]);
                    std::swap(w[i], w[i + 1]);
                }
            }

            size--;
        }

        weights[vertex_id] = make_float4(w[0], w[1], w[2], w[3]);
        nodes[vertex_id] = make_int4(n[0], n[1], n[2], n[3]);
    }

    void clear()
    {
        weights.clear();
        nodes.clear();
    }
};

} // supersurfel_fusion

#endif // DEFORMATION_GRAPH_TYPES_HPP
