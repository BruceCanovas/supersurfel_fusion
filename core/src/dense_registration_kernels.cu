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


#include <supersurfel_fusion/dense_registration_kernels.cuh>


namespace supersurfel_fusion
{

__global__ void makeCorrespondences(const float3* source_positions,
                                    const float3* source_colors,
                                    const Mat33* source_orientations,
                                    const float* source_confidences,
                                    const float3* target_positions,
                                    const float3* target_colors,
                                    const Mat33* target_orientations,
                                    const float* target_confidences,
                                    float3* matched_source_normals,
                                    float3* matched_source_positions,
                                    float3* matched_target_normals,
                                    float3* matched_target_positions,
                                    bool* valids,
                                    Mat33 R,
                                    float3 t,
                                    float fx,
                                    float fy,
                                    float cx,
                                    float cy,
                                    cudaTextureObject_t tex_index,
                                    cudaTextureObject_t tex_depth,
                                    int width,
                                    int height,
                                    int nb_supersurfels)
{
    int source_id = blockIdx.x * blockDim.x + threadIdx.x;

    if(source_id < nb_supersurfels)
    {
        valids[source_id] = false;

        if(source_confidences[source_id] > 0.0f)
        {
            float3 p_view = R * source_positions[source_id] + t;

            int u = lroundf(p_view.x * fx / p_view.z + cx);
            int v = lroundf(p_view.y * fy / p_view.z + cy);

            if(u < 0 || u >= width || v < 0 || v >= height)
                return;

            int target_id = tex2D<int>(tex_index, u, v);

            if(target_confidences[target_id] > 0.0f)
            {
                float dist_color = length(rgbToLab(source_colors[source_id]) - rgbToLab(target_colors[target_id]));

                float t_depth = tex2D<float>(tex_depth, u, v);

                if(isfinite(t_depth))
                {
                    float3 s_normal = normalize(source_orientations[source_id].rows[2]);
                    s_normal = normalize(R * s_normal);
                    float3 t_normal = normalize(target_orientations[target_id].rows[2]);
                    //float3 t_position = target_positions[target_id];
                    float3 t_position = make_float3(t_depth * (u - cx) / fx, t_depth * (v - cy) / fy, t_depth);

                    if(dist_color < 20.0f &&
                       length(p_view - t_position) < 0.05f  &&
                       acosf(dot(s_normal, t_normal)) * 180.0f / PI < 30.0f)
                    {
                        matched_source_positions[source_id] = p_view;
                        matched_source_normals[source_id] = s_normal;

                        //matched_target_positions[source_id] = make_float3(t_depth * (u - cx) / fx, t_depth * (v - cy) / fy, t_depth);
                        matched_target_positions[source_id] = t_position;
                        matched_target_normals[source_id] = t_normal;

                        valids[source_id] = true;
                    }
                }


            }
        }
    }
}

}
