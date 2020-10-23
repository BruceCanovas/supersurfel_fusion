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


#ifndef DEFORMATION_GRAPH_KERNELS_CUH
#define DEFORMATION_GRAPH_KERNELS_CUH

#include <supersurfel_fusion/deformation_graph_types.hpp>


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
                                 int model_size);

} // supersurfel_fusion

#endif // DEFORMATION_GRAPH_KERNELS_CUH
