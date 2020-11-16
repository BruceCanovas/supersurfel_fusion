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


#include <supersurfel_fusion/supersurfel_fusion_kernels.cuh>


namespace supersurfel_fusion
{

__device__ void maxEigenVec(const Cov3& A, float3& eigenVec, int n)
{
    Cov3 Ai = A/trace(A);
    for(int i=0; i<n; ++i)
    {
        Ai=square(Ai);
        Ai/=trace(Ai);
    }

    float vmax = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(Ai.xx, Ai.xy), Ai.xz), Ai.yy), Ai.yz), Ai.zz);
    if(Ai.xx==vmax || Ai.xy==vmax || Ai.xz==vmax)
    {
        eigenVec = normalize(make_float3(Ai.xx, Ai.xy, Ai.xz));
    }else if(Ai.yy==vmax || Ai.yz==vmax)
    {
        eigenVec = normalize(make_float3(Ai.xy, Ai.yy, Ai.yz));
    }else{
        eigenVec = normalize(make_float3(Ai.xz, Ai.yz, Ai.zz));
    }

}

__device__ void eigenDecomposition(const Cov3& A, Mat33& eigenVecs, float3& eigenVals, int n)
{
    Cov3 Ai = A/trace(A);
    Cov3 Bi = make_cov3(1.f-Ai.xx, -Ai.xy, -Ai.xz, 1.f-Ai.yy, -Ai.yz, 1.f-Ai.zz);

    for(int i=0; i<n; ++i)
    {
        Ai=square(Ai);
        Ai/=trace(Ai);

        Bi=square(Bi);
        Bi/=trace(Bi);
    }

    float vmax = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(Ai.xx, Ai.xy), Ai.xz), Ai.yy), Ai.yz), Ai.zz);
    if(Ai.xx==vmax || Ai.xy==vmax || Ai.xz==vmax)
    {
        eigenVecs.rows[0]=normalize(make_float3(Ai.xx, Ai.xy, Ai.xz));
    }else if(Ai.yy==vmax || Ai.yz==vmax)
    {
        eigenVecs.rows[0]=normalize(make_float3(Ai.xy, Ai.yy, Ai.yz));
    }else{
        eigenVecs.rows[0]=normalize(make_float3(Ai.xz, Ai.yz, Ai.zz));
    }

    vmax = fmaxf(fmaxf(fmaxf(fmaxf(fmaxf(Bi.xx, Bi.xy), Bi.xz), Bi.yy), Bi.yz), Bi.zz);
    if(Bi.xx==vmax || Bi.xy==vmax || Bi.xz==vmax)
    {
        eigenVecs.rows[2]=normalize(make_float3(Bi.xx, Bi.xy, Bi.xz));
    }else if(Bi.yy==vmax || Bi.yz==vmax)
    {
        eigenVecs.rows[2]=normalize(make_float3(Bi.xy, Bi.yy, Bi.yz));
    }else{
        eigenVecs.rows[2]=normalize(make_float3(Bi.xz, Bi.yz, Bi.zz));
    }

    eigenVecs.rows[1] = cross(eigenVecs.rows[2], eigenVecs.rows[0]);

    float emax = fmaxf(fmaxf(eigenVecs.rows[0].x, eigenVecs.rows[0].y), eigenVecs.rows[0].z);
    if(eigenVecs.rows[0].x==emax)
        eigenVals.x = (A.xx*eigenVecs.rows[0].x+A.xy*eigenVecs.rows[0].y+A.xz*eigenVecs.rows[0].z) / eigenVecs.rows[0].x;
    else if(eigenVecs.rows[0].y==emax)
        eigenVals.x = (A.xy*eigenVecs.rows[0].x+A.yy*eigenVecs.rows[0].y+A.yz*eigenVecs.rows[0].z) / eigenVecs.rows[0].y;
    else
        eigenVals.x = (A.xz*eigenVecs.rows[0].x+A.yz*eigenVecs.rows[0].y+A.zz*eigenVecs.rows[0].z) / eigenVecs.rows[0].z;


    emax = fmaxf(fmaxf(eigenVecs.rows[1].x, eigenVecs.rows[1].y), eigenVecs.rows[1].z);
    if(eigenVecs.rows[1].x==emax)
        eigenVals.y = (A.xx*eigenVecs.rows[1].x+A.xy*eigenVecs.rows[1].y+A.xz*eigenVecs.rows[1].z) / eigenVecs.rows[1].x;
    else if(eigenVecs.rows[1].y==emax)
        eigenVals.y = (A.xy*eigenVecs.rows[1].x+A.yy*eigenVecs.rows[1].y+A.yz*eigenVecs.rows[1].z) / eigenVecs.rows[1].y;
    else
        eigenVals.y = (A.xz*eigenVecs.rows[1].x+A.yz*eigenVecs.rows[1].y+A.zz*eigenVecs.rows[1].z) / eigenVecs.rows[1].z;

    emax = fmaxf(fmaxf(eigenVecs.rows[2].x, eigenVecs.rows[2].y), eigenVecs.rows[2].z);
    if(eigenVecs.rows[2].x==emax)
        eigenVals.z = (A.xx*eigenVecs.rows[2].x+A.xy*eigenVecs.rows[2].y+A.xz*eigenVecs.rows[2].z) / eigenVecs.rows[2].x;
    else if(eigenVecs.rows[2].y==emax)
        eigenVals.z = (A.xy*eigenVecs.rows[2].x+A.yy*eigenVecs.rows[2].y+A.yz*eigenVecs.rows[2].z) / eigenVecs.rows[2].y;
    else
        eigenVals.z = (A.xz*eigenVecs.rows[2].x+A.yz*eigenVecs.rows[2].y+A.zz*eigenVecs.rows[2].z) / eigenVecs.rows[2].z;

}

__global__ void computeSupersurfelCoeffs(float3* positions,
                                         Cov3* shapes,
                                         float3* colors,
                                         float* confidences,
                                         cudaTextureObject_t texImg,
                                         cudaTextureObject_t texDepth,
                                         cudaTextureObject_t texIndex,
                                         cudaTextureObject_t texInliers,
                                         cudaTextureObject_t texBound,
                                         int width,
                                         int height,
                                         float fx,
                                         float fy,
                                         float cx,
                                         float cy)
{
    int x = blockIdx.x *blockDim.x + threadIdx.x;
    int y = blockIdx.y *blockDim.y + threadIdx.y;

    if(x >= width || y >= height)
        return;

    if(!tex2D<unsigned char>(texInliers, x, y))
        return;

    int index = tex2D<int>(texIndex, x, y);

    int bound = tex2D<int>(texBound, x, y);

    uchar4 color = tex2D<uchar4>(texImg, x, y);
    float depth = tex2D<float>(texDepth, x, y);

    if(isfinite(depth) && depth > 0.0f && bound == 0)
    {
        float3 pos = make_float3((x-cx) * depth / fx, (y-cy) * depth / fy, depth);
        float3 lab = rgbToLab(make_float3((float)color.x, (float)color.y, (float)color.z));

        atomicAdd(&positions[index].x, pos.x);
        atomicAdd(&positions[index].y, pos.y);
        atomicAdd(&positions[index].z, pos.z);
        atomicAdd(&colors[index].x, lab.x);
        atomicAdd(&colors[index].y, lab.y);
        atomicAdd(&colors[index].z, lab.z);

        Cov3 cov = outer_product(pos);

        atomicAdd(&shapes[index].xx, cov.xx);
        atomicAdd(&shapes[index].xy, cov.xy);
        atomicAdd(&shapes[index].xz, cov.xz);
        atomicAdd(&shapes[index].yy, cov.yy);
        atomicAdd(&shapes[index].yz, cov.yz);
        atomicAdd(&shapes[index].zz, cov.zz);
        atomicAdd(&confidences[index], 1.f);
    }
}

__global__ void computeSupersurfels(float3* positions,
                                    float3* colors,
                                    int2* stamps,
                                    Mat33* orientations,
                                    Cov3* shapes,
                                    float2* dims,
                                    float* confidences,
                                    float z_min,
                                    float z_max,
                                    int stamp,
                                    int nb_supersurfels)
{
    int k = blockIdx.x *blockDim.x + threadIdx.x;

    if(k >= nb_supersurfels)
        return;

    float z = positions[k].z / confidences[k];

    if(isfinite(z) && confidences[k] > 100.0f && z > z_min && z < z_max)
    {
        positions[k].x /= confidences[k];
        positions[k].y /= confidences[k];
        positions[k].z = z;

        colors[k].x /= confidences[k];
        colors[k].y /= confidences[k];
        colors[k].z /= confidences[k];

        float3 rgb = labToRgb(colors[k]);

        colors[k] =  rgb;

        float3 position = positions[k];

        shapes[k] = shapes[k] / confidences[k] - outer_product(position);

        float3 eigen_vals;
        Mat33 eigen_vecs;

        eigenDecomposition(shapes[k], eigen_vecs, eigen_vals, 10);

        orientations[k] = eigen_vecs;

        dims[k].x = eigen_vals.x;
        dims[k].y = eigen_vals.y;

        stamps[k].x = stamp;
        stamps[k].y = stamp;

        float theta = acosf(fabs(dot(normalize(position), eigen_vecs.rows[2])));

        //printf("%f ", eigen_vals.x / eigen_vals.y);

        if(/*theta * 180.0f / PI > 80.0f || */eigen_vals.x / eigen_vals.y > 50.0f)
            confidences[k] = -1.0f;

        //if(eigen_vals.x / eigen_vals.y > 4.0f)
        //    dims[k].y = 4.0f * eigen_vals.x;

    }
    else
        confidences[k] = -1.0f;
}

__global__ void fuseSupersurfels(const float3* frame_positions,
                                 const float3* frame_colors,
                                 const Mat33* frame_orientations,
                                 const Cov3* frame_shapes,
                                 const float2* frame_dims,
                                 const float* frame_confidences,
                                 float3* model_positions,
                                 float3* model_colors,
                                 int2* model_stamps,
                                 Mat33* model_orientations,
                                 Cov3* model_shapes,
                                 float2* model_dims,
                                 float* model_confidences,
                                 bool* matched,
                                 cudaTextureObject_t tex_index,
                                 Mat33 R,
                                 float3 t,
                                 float fx,
                                 float fy,
                                 float cx,
                                 float cy,
                                 float z_min,
                                 float z_max,
                                 int width,
                                 int height,
                                 int stamp,
                                 int* nb_match,
                                 int model_length)
{
    int model_id =  blockIdx.x * blockDim.x + threadIdx.x;

    if(model_id >= model_length)
        return;

    if(model_confidences[model_id] > 0.0f)
    {
        Mat33 Rview = transpose(R);
        float3 tview = - Rview * t;

        float3 model_position = model_positions[model_id];
        float3 pm_view = Rview * model_position + tview;
        int2 pm_proj = make_int2(lroundf(pm_view.x * fx / pm_view.z + cx), lroundf(pm_view.y * fy / pm_view.z + cy));

        if(pm_view.z > z_min && pm_view.z < z_max && pm_proj.x >= 0 && pm_proj.x < width && pm_proj.y >= 0 && pm_proj.y < height)
        {
            int frame_id = tex2D<int>(tex_index, pm_proj.x, pm_proj.y);

            if(frame_confidences[frame_id] > 0.0f)
            {
                float3 frame_position = R * frame_positions[frame_id] + t;
                Cov3 frame_shape = mult_ABAt(R, frame_shapes[frame_id]);
                Mat33 frame_rot = frame_orientations[frame_id] * transpose(R);

                float3 frame_normal = normalize(frame_rot.rows[2]);
                float3 model_normal = normalize(model_orientations[model_id].rows[2]);
                float3 frame_lab = rgbToLab(frame_colors[frame_id]);
                float3 model_lab = rgbToLab(model_colors[model_id]);

                //float dist = length(model_position - frame_position);
                float lab_dist = length(model_lab - frame_lab);
                float delta_norm = fabsf(dot(model_normal, frame_normal));
                float dist_to_plane = 0.5f * (fabsf(dot(frame_position, model_normal) - dot(model_position, model_normal)) +
                                              fabsf(dot(model_position, frame_normal) - dot(frame_position, frame_normal)));

                if(lab_dist < 20.0f && delta_norm > 0.8f && dist_to_plane < 0.05f)
                {
                    matched[frame_id] = true;

                    atomicAggInc(nb_match);

                    float m_conf = model_confidences[model_id];
                    float f_conf = frame_confidences[frame_id];
                    float ratio = 1.0f / (m_conf + f_conf);

                    model_stamps[model_id].y = stamp;

                    float3 fused_color;
                    fused_color = labToRgb(ratio * (f_conf * frame_lab + m_conf * model_lab));
                    //fused_color = ratio * (f_conf * frame_colors[frame_id] + m_conf * model_colors[model_id]);

                    Cov3 f_shape_1, m_shape_1, fused_shape, fused_shape_1;
                    float3 fused_position;

                    float w = ratio * f_conf;

                    if(inverse(frame_shape, f_shape_1) && inverse(model_shapes[model_id], m_shape_1))
                    {
                        fused_shape_1 = w * f_shape_1 + (1.0f - w) * m_shape_1;

                        if(inverse(fused_shape_1, fused_shape))
                            fused_position = fused_shape * (w * f_shape_1 * frame_position + (1.0f - w) * m_shape_1 * model_position);
                        else
                        {
                            fused_shape = ratio * (f_conf * frame_shape + m_conf * model_shapes[model_id]);
                            fused_position = ratio * (f_conf * frame_position + m_conf * model_position);
                        }
                    }
                    else
                    {
                        fused_shape = ratio * (f_conf * frame_shape + m_conf * model_shapes[model_id]);
                        fused_position = ratio * (f_conf * frame_position + m_conf * model_position);
                    }

                    model_positions[model_id] = fused_position;
                    model_confidences[model_id] = m_conf + f_conf;
                    model_shapes[model_id] = fused_shape;

                    Mat33 eigen_vecs;
                    float3 eigen_vals;

                    eigenDecomposition(fused_shape, eigen_vecs, eigen_vals, 10);

                    model_orientations[model_id] = eigen_vecs;

                    model_colors[model_id] = fused_color;

                    model_dims[model_id].x = eigen_vals.x;
                    model_dims[model_id].y = eigen_vals.y;

                    //if(eigen_vals.x / eigen_vals.y > 5.0f)
                    //    model_dims[model_id].y = 5.0f * eigen_vals.x;
                }
            }
        }
    }
}

__global__ void insertSupersurfels(const float3* frame_positions,
                                   const float3* frame_colors,
                                   const Mat33* frame_orientations,
                                   const Cov3* frame_shapes,
                                   const float2* frame_dims,
                                   const float* frame_confidences,
                                   float3* model_positions,
                                   float3* model_colors,
                                   int2* model_stamps,
                                   Mat33* model_orientations,
                                   Cov3* model_shapes,
                                   float2* model_dims,
                                   float* model_confidences,
                                   Mat33 R,
                                   float3 t,
                                   const int stamp,
                                   bool* matched,
                                   int* model_length,
                                   int frame_length,
                                   int max_model_length)
{
    int frame_id =  blockIdx.x * blockDim.x + threadIdx.x;

    if(frame_id >= frame_length)
        return;

    if(frame_confidences[frame_id] > 0.0f && !matched[frame_id])
    {
        int k = atomicAggInc(model_length);

        if(k < max_model_length)
        {
            float3 frame_position = R * frame_positions[frame_id] + t;
            Cov3 frame_shape = mult_ABAt(R, frame_shapes[frame_id]);
            Mat33 frame_rot = frame_orientations[frame_id] * transpose(R);

            model_positions[k] = frame_position;
            model_confidences[k] = frame_confidences[frame_id];
            model_colors[k] = frame_colors[frame_id];
            model_stamps[k] = make_int2(stamp, stamp);
            model_dims[k] = frame_dims[frame_id];
            model_orientations[k] = frame_rot;
            model_shapes[k] = frame_shape;
        }
        else
            atomicAggDec(model_length);
    }
}

__global__ void filterModel(float3* positions,
                            int2* stamps,
                            float* confidences,
                            int* states,
                            int current_stamp,
                            int deltaT,
                            float conf_thresh,
                            cudaTextureObject_t texDepth,
                            int* nbRemovedSurfelDev,
                            int* nbActive,
                            Mat33 R,
                            float3 t,
                            float fx,
                            float fy,
                            float cx,
                            float cy,
                            float z_min,
                            float z_max,
                            int width,
                            int height,
                            int nb_supersurfels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= nb_supersurfels)
        return;

    int state = 0;

    if((current_stamp - stamps[idx].y > deltaT && confidences[idx] < conf_thresh && current_stamp > deltaT)
            || confidences[idx] <= 0.0f)
    {
        confidences[idx] = -1.0f;
        state = 2;
    }
    else
    {
        //float3 pos = positions[idx];

        //Mat33 Rt = transpose(R);

        //float3 p = Rt * (pos - t);

        float3 p =  R * positions[idx] + t;

        if(p.z > z_min && p.z < z_max)
        {
            float u = fx * p.x / p.z + cx;
            float v = fy * p.y / p.z + cy;

            if(u >= 0.0f && u < width && v >= 0.0f && v < height)
            {
                float z = tex2D<float>(texDepth, u, v);

                if(p.z < 0.8f * z)
                {
                    confidences[idx] = -1.0f;
                    state = 2;
                }
            }
            else state = 1;
        }
        else
            state = 1;
    }

    if(state == 0)
        atomicAggInc(nbActive);
    if(state == 2)
        atomicAggInc(nbRemovedSurfelDev);

    states[idx] = state;
}

__global__ void applyTransformSuperSurfel(float3* positions,
                                          Mat33* orientations,
                                          Cov3* shapes,
                                          float* confidences,
                                          Mat33 R,
                                          float3 t,
                                          int nb_supersurfels)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k >= nb_supersurfels)
        return;

    if(confidences[k] <= 0.0f)
        return;

    positions[k] = R * positions[k] + t;
    orientations[k] = orientations[k] * transpose(R);
    shapes[k] = mult_ABAt(R, shapes[k]);
}

__global__ void extractLocalPointCloudKernel(const float3* model_positions,
                                             const Mat33* model_orientations,
                                             const float* model_confidences,
                                             float3* cloud_positions,
                                             float3* cloud_normals,
                                             float conf_thresh,
                                             Mat33 R,
                                             float3 t,
                                             float radius,
                                             int* nb_points,
                                             int nb_supersurfels)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    if(k >= nb_supersurfels)
        return;

    if(model_confidences[k] >= conf_thresh)
    {
        float3 p = R * model_positions[k] + t;

        if(length(p) < radius)
        {
            int id = atomicAggInc(nb_points);

            cloud_positions[id] = p;
            cloud_normals[id] = normalize(R * model_orientations[k].rows[2]);
        }
    }

}

__global__ void findBestMatches(const float3* frame_positions,
                                const float3* frame_colors,
                                const Mat33* frame_orientations,
                                //const Cov3* frame_shapes,
                                //const float2* frame_dims,
                                const float* frame_confidences,
                                float3* model_positions,
                                float3* model_colors,
                                //int2* model_stamps,
                                Mat33* model_orientations,
                                //Cov3* model_shapes,
                                //float2* model_dims,
                                float* model_confidences,
                                bool* matched,
                                float2* idx_scores,
                                cudaTextureObject_t tex_index,
                                Mat33 R,
                                float3 t,
                                float fx,
                                float fy,
                                float cx,
                                float cy,
                                float z_min,
                                float z_max,
                                int width,
                                int height,
                                //int stamp,
                                //int* nb_match,
                                int model_length)
{
    int model_id =  blockIdx.x * blockDim.x + threadIdx.x;

    if(model_id >= model_length)
        return;

    if(model_confidences[model_id] > 0.0f)
    {
        Mat33 Rview = transpose(R);
        float3 tview = - Rview * t;

        float3 model_position = model_positions[model_id];
        float3 pm_view = Rview * model_position + tview;
        int2 pm_proj = make_int2(lroundf(pm_view.x * fx / pm_view.z + cx), lroundf(pm_view.y * fy / pm_view.z + cy));

        if(pm_view.z > z_min && pm_view.z < z_max && pm_proj.x >= 0 && pm_proj.x < width && pm_proj.y >= 0 && pm_proj.y < height)
        {
            int frame_id = tex2D<int>(tex_index, pm_proj.x, pm_proj.y);

            if(frame_confidences[frame_id] > 0.0f)
            {
                float3 frame_position = R * frame_positions[frame_id] + t;
                //Cov3 frame_shape = mult_ABAt(R, frame_shapes[frame_id]);
                Mat33 frame_rot = frame_orientations[frame_id] * transpose(R);

                float3 frame_normal = normalize(frame_rot.rows[2]);
                float3 model_normal = normalize(model_orientations[model_id].rows[2]);
                float3 frame_lab = rgbToLab(frame_colors[frame_id]);
                float3 model_lab = rgbToLab(model_colors[model_id]);

                float dist = length(model_position - frame_position);
                float lab_dist = length(model_lab - frame_lab);
                float delta_norm = fabsf(dot(model_normal, frame_normal));
                //float dist_to_plane = 0.5f * (fabsf(dot(frame_position, model_normal) - dot(model_position, model_normal)) +
                //                              fabsf(dot(model_position, frame_normal) - dot(frame_position, frame_normal)));

                if(lab_dist < 20.0f && delta_norm > 0.8f && dist < 0.05/* && dist_to_plane < 0.05f*/)
                {
                    matched[frame_id] = true;

                    //atomicAggInc(nb_match);

                    if(dist < idx_scores[frame_id].y)
                    {
                        atomicExch(&idx_scores[frame_id].y, dist);
                        atomicExch(&idx_scores[frame_id].x, float(model_id));

                    }
                }
            }
        }
    }
}

__global__ void updateSupersurfels(const float3* frame_positions,
                                   const float3* frame_colors,
                                   const Cov3* frame_shapes,
                                   const float* frame_confidences,
                                   float3* model_positions,
                                   float3* model_colors,
                                   int2* model_stamps,
                                   Mat33* model_orientations,
                                   Cov3* model_shapes,
                                   float2* model_dims,
                                   float* model_confidences,
                                   bool* matched,
                                   float2* idx_scores,
                                   Mat33 R,
                                   float3 t,
                                   int stamp,
                                   int frame_length)
{
    int frame_id =  blockIdx.x * blockDim.x + threadIdx.x;

    if(frame_id >= frame_length)
        return;

    if(matched[frame_id])
    {
        int model_id = int(idx_scores[frame_id].x);

        float3 model_position = model_positions[model_id];
        float3 frame_position = R * frame_positions[frame_id] + t;
        Cov3 frame_shape = mult_ABAt(R, frame_shapes[frame_id]);
        float3 frame_lab = rgbToLab(frame_colors[frame_id]);
        float3 model_lab = rgbToLab(model_colors[model_id]);

        float m_conf = model_confidences[model_id];
        float f_conf = frame_confidences[frame_id];
        float ratio = 1.0f / (m_conf + f_conf);

        model_stamps[model_id].y = stamp;

        float3 fused_color;
        fused_color = labToRgb(ratio * (f_conf * frame_lab + m_conf * model_lab));
        //fused_color = ratio * (f_conf * frame_colors[frame_id] + m_conf * model_colors[model_id]);

        Cov3 f_shape_1, m_shape_1, fused_shape, fused_shape_1;
        float3 fused_position;

        float w = ratio * f_conf;

        if(inverse(frame_shape, f_shape_1) && inverse(model_shapes[model_id], m_shape_1))
        {
            fused_shape_1 = w * f_shape_1 + (1.0f - w) * m_shape_1;

            if(inverse(fused_shape_1, fused_shape))
                fused_position = fused_shape * (w * f_shape_1 * frame_position + (1.0f - w) * m_shape_1 * model_position);
            else
            {
                fused_shape = ratio * (f_conf * frame_shape + m_conf * model_shapes[model_id]);
                fused_position = ratio * (f_conf * frame_position + m_conf * model_position);
            }
        }
        else
        {
            fused_shape = ratio * (f_conf * frame_shape + m_conf * model_shapes[model_id]);
            fused_position = ratio * (f_conf * frame_position + m_conf * model_position);
        }

        model_positions[model_id] = fused_position;
        model_confidences[model_id] = m_conf + f_conf;
        model_shapes[model_id] = fused_shape;

        Mat33 eigen_vecs;
        float3 eigen_vals;

        eigenDecomposition(fused_shape, eigen_vecs, eigen_vals, 10);

        model_orientations[model_id] = eigen_vecs;

        model_colors[model_id] = fused_color;

        model_dims[model_id].x = eigen_vals.x;
        model_dims[model_id].y = eigen_vals.y;
    }
}

} // namespace supersurfel_fusion
