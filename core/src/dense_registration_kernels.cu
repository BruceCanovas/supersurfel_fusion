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
                       length(p_view - t_position) < 0.1f  &&
                       fabsf(dot(s_normal, t_normal)) > 0.8f)
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

__global__ void buildSymmetricPoint2PlaneSystem2(MotionTrackingData* system,
                                                 const float3* source_positions,
                                                 const float3* source_normals,
                                                 const float3* target_positions,
                                                 const float3* target_normals,
                                                 float3 source_centroid,
                                                 float3 target_centroid,
                                                 float scale,
                                                 int nb_pairs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    MotionTrackingData mtd_sum;
    #pragma unroll
    for(int k = 0; k < 21; k++)
        mtd_sum.JtJ[k] = 0.0f;

    #pragma unroll
    for(int k = 0; k < 6; k++)
        mtd_sum.Jtr[k] = 0.0f;

    if(i < nb_pairs)
    {
        float3 ps = scale * (source_positions[i] - source_centroid);
        float3 pt = scale * (target_positions[i] - target_centroid);
        float3 ns = normalize(source_normals[i]);
        float3 nt = normalize(target_normals[i]);

        float3 d = pt - ps;
        float3 c1 = cross(pt, ns);
        float3 c2 = cross(ps, nt);
        float dn1 = dot(d, ns);
        float dn2 = dot(d, nt);

        float x1[6] = {c1.x, c1.y, c1.z, ns.x, ns.y, ns.z};
        float x2[6] = {c2.x, c2.y, c2.z, nt.x, nt.y, nt.z};

        for(int j = 0; j < 6; j++)
            mtd_sum.Jtr[j] = dn1 * x1[j] + dn2 * x2[j];

        mtd_sum.JtJ[0] = x1[0] * x1[0] + x2[0] * x2[0];
        mtd_sum.JtJ[1] = x1[0] * x1[1] + x2[0] * x2[1];
        mtd_sum.JtJ[2] = x1[0] * x1[2] + x2[0] * x2[2];
        mtd_sum.JtJ[3] = x1[0] * x1[3] + x2[0] * x2[3];
        mtd_sum.JtJ[4] = x1[0] * x1[4] + x2[0] * x2[4];
        mtd_sum.JtJ[5] = x1[0] * x1[5] + x2[0] * x2[5];

        mtd_sum.JtJ[6] = x1[1] * x1[1] + x2[1] * x2[1];
        mtd_sum.JtJ[7] = x1[1] * x1[2] + x2[1] * x2[2];
        mtd_sum.JtJ[8] = x1[1] * x1[3] + x2[1] * x2[3];
        mtd_sum.JtJ[9] = x1[1] * x1[4] + x2[1] * x2[4];
        mtd_sum.JtJ[10] = x1[1] * x1[5] + x2[1] * x2[5];

        mtd_sum.JtJ[11] = x1[2] * x1[2] + x2[2] * x2[2];
        mtd_sum.JtJ[12] = x1[2] * x1[3] + x2[2] * x2[3];
        mtd_sum.JtJ[13] = x1[2] * x1[4] + x2[2] * x2[4];
        mtd_sum.JtJ[14] = x1[2] * x1[5] + x2[2] * x2[5];

        mtd_sum.JtJ[15] = x1[3] * x1[3] + x2[3] * x2[3];
        mtd_sum.JtJ[16] = x1[3] * x1[4] + x2[3] * x2[4];
        mtd_sum.JtJ[17] = x1[3] * x1[5] + x2[3] * x2[5];

        mtd_sum.JtJ[18] = x1[4] * x1[4] + x2[4] * x2[4];
        mtd_sum.JtJ[19] = x1[4] * x1[5] + x2[4] * x2[5];

        mtd_sum.JtJ[20] = x1[5] * x1[5] + x2[5] * x2[5];
    }

    mtd_sum = blockReduceSum(mtd_sum);

    if(threadIdx.x == 0)
        atomicAdd(system, mtd_sum);
}

}
