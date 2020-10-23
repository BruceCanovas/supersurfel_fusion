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


#include <supersurfel_fusion/deformation_graph_kernels.cuh>
#include "supersurfel_fusion/matrix_math.cuh"


namespace supersurfel_fusion
{

__global__ void applyDeformation(float3* model_positions,
                                 Mat33* model_orientations,
                                 Cov3* model_shapes,
                                 const float3* nodes_positions,
                                 const Mat33* nodes_rotations,
                                 const float3* nodes_translations,
                                 const float4* neighbours_weights,
                                 const int4* neighbours_idx,
                                 int model_size)
{
    int i = blockIdx.x *blockDim.x + threadIdx.x;

    if(i >= model_size)
        return;

    float4 weights = neighbours_weights[i];
    int4 nn_idx = neighbours_idx[i];

    float3 pi, po;
    pi = model_positions[i];
    po = make_float3(0.0f, 0.0f, 0.0f);

    float4 b_quat = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for(int k = 0; k < 4; k++)
    {
        int node_id = get_coord(nn_idx, k);
        float wk = get_coord(weights, k);

        float3 gk = nodes_positions[node_id];
        Mat33 Rk = nodes_rotations[node_id];
        float3 tk = nodes_translations[node_id];

        float4 qk = wk * rotMatToQuat(Rk);

        po += wk * (Rk * (pi - gk) + gk +tk);

        b_quat += qk;
    }

    b_quat /= length(b_quat);

    Mat33 av_rot = quatToRotMat(b_quat);

    model_orientations[i] = model_orientations[i] * transpose(av_rot);
    model_shapes[i] = mult_ABAt(av_rot, model_shapes[i]);
    model_positions[i] = po;
}

} // supersurfel_fusion
