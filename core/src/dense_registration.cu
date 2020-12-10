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


#include <supersurfel_fusion/dense_registration.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include <supersurfel_fusion/dense_registration_kernels.cuh>
#include <supersurfel_fusion/cuda_error_check.h>
#include <Eigen/Geometry>
#include <Eigen/Dense>


namespace supersurfel_fusion
{

DenseRegistration::DenseRegistration()
    : nbIter(10),
      covThresh(0.04)
{
    cudaMallocManaged(&mtd, sizeof(MotionTrackingData));
}

DenseRegistration::DenseRegistration(int nb_iter,
                                     double cov_thresh)
    : nbIter(nb_iter),
      covThresh(cov_thresh)
{
    cudaMallocManaged(&mtd, sizeof(MotionTrackingData));
}

DenseRegistration::~DenseRegistration()
{
    cudaFree(mtd);
}

bool DenseRegistration::align(const thrust::device_vector<float3>& source_positions,
                              const thrust::device_vector<float3>& source_colors,
                              const thrust::device_vector<Mat33>& source_orientations,
                              const thrust::device_vector<float>& source_confidences,
                              const thrust::device_vector<float3>& target_positions,
                              const thrust::device_vector<float3>& target_colors,
                              const thrust::device_vector<Mat33>& target_orientations,
                              const thrust::device_vector<float>& target_confidences,
                              int source_size,
                              const cv::Ptr<Texture<float>>& tex_depth,
                              const cv::Ptr<Texture<int>>& tex_index,
                              const Mat33& R_init,
                              const float3& t_init,
                              const CamParam& cam,
                              Mat33& R,
                              float3& t)
{
    bool valid = true;

    int iter = 0;

    Eigen::Matrix4d tf_inc = Eigen::Matrix4d::Identity();

    R = make_mat33(1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f);
    t = make_float3(0.0f, 0.0f, 0.0f);
    Mat33 R_corres;
    float3 t_corres;
    Mat33 R_inc;
    float3 t_inc;

    dim3 dim_block(64);
    dim3 dim_grid((source_size + dim_block.x - 1) / dim_block.x);

    Eigen::Matrix<double, 6, 6> pose_covariance;

    while(iter++ < nbIter)
    {
        Eigen::Matrix3f basis = tf_inc.block<3,3>(0,0).cast<float>();
        R_inc = make_mat33(basis(0,0), basis(0,1), basis(0,2),
                           basis(1,0), basis(1,1), basis(1,2),
                           basis(2,0), basis(2,1), basis(2,2));
        Eigen::Vector3f origin = tf_inc.block<3,1>(0,3).cast<float>();
        t_inc = make_float3(origin(0), origin(1), origin(2));

        R_corres = R_inc * R_init;
        t_corres = R_inc * t_init + t_inc;

        cudaMemset(mtd, 0, sizeof(MotionTrackingData));

        float3 target_centroid = make_float3(0.0f, 0.0f, 0.0f), source_centroid = target_centroid;
        float scale = 1.0f;
        int nb_pairs = 0.0f;

        thrust::device_vector<float3> matched_source_normals(source_size), matched_source_positions(source_size),
                                      matched_target_normals(source_size), matched_target_positions(source_size);
        thrust::device_vector<bool> valids(source_size, false);

        makeCorrespondences<<<dim_grid, dim_block>>>(thrust::raw_pointer_cast(source_positions.data()),
                                                     thrust::raw_pointer_cast(source_colors.data()),
                                                     thrust::raw_pointer_cast(source_orientations.data()),
                                                     thrust::raw_pointer_cast(source_confidences.data()),
                                                     thrust::raw_pointer_cast(target_colors.data()),
                                                     thrust::raw_pointer_cast(target_orientations.data()),
                                                     thrust::raw_pointer_cast(target_confidences.data()),
                                                     thrust::raw_pointer_cast(matched_source_normals.data()),
                                                     thrust::raw_pointer_cast(matched_source_positions.data()),
                                                     thrust::raw_pointer_cast(matched_target_normals.data()),
                                                     thrust::raw_pointer_cast(matched_target_positions.data()),
                                                     thrust::raw_pointer_cast(valids.data()),
                                                     R_corres,
                                                     t_corres,
                                                     cam.fx,
                                                     cam.fy,
                                                     cam.cx,
                                                     cam.cy,
                                                     tex_index->getTextureObject(),
                                                     tex_depth->getTextureObject(),
                                                     cam.width,
                                                     cam.height,
                                                     source_size);
        cudaDeviceSynchronize();
        CudaCheckError();

        // Filtered out unvalid data and compute scale and centroïds of extracted valid ones
        thrust::remove_if(thrust::make_zip_iterator(thrust::make_tuple(matched_source_positions.begin(), matched_source_normals.begin(), matched_target_positions.begin(), matched_target_normals.begin())),
                          thrust::make_zip_iterator(thrust::make_tuple(matched_source_positions.end(), matched_source_normals.end(), matched_target_positions.end(), matched_target_normals.end())),
                          valids.begin(),
                          thrust::logical_not<bool>());
        nb_pairs = thrust::count_if(valids.begin(), valids.end(), thrust::identity<bool>());

        if(nb_pairs < 100)
        {
            std::cout<<" In ICP, number of matches is too small!   nb matches = "<<nb_pairs<<std::endl;
            valid = false;
            break;
        }

        matched_source_normals.resize(nb_pairs);
        matched_source_positions.resize(nb_pairs);
        matched_target_normals.resize(nb_pairs);
        matched_target_positions.resize(nb_pairs);

        target_centroid = thrust::reduce(matched_target_positions.begin(), matched_target_positions.end(), make_float3(0.0f, 0.0f, 0.0f), add_float3());
        target_centroid /= float(nb_pairs);
        source_centroid = thrust::reduce(matched_source_positions.begin(), matched_source_positions.end(), make_float3(0.0f, 0.0f, 0.0f), add_float3());
        source_centroid /= float(nb_pairs);

        thrust::device_vector<float> t_dist2_to_center(matched_target_positions.size()), s_dist2_to_center(matched_source_positions.size());
        thrust::transform(matched_target_positions.begin(), matched_target_positions.end(), t_dist2_to_center.begin(), dist2_to_center(target_centroid));
        thrust::transform(matched_source_positions.begin(), matched_source_positions.end(), s_dist2_to_center.begin(), dist2_to_center(source_centroid));
        scale = thrust::reduce(t_dist2_to_center.begin(), t_dist2_to_center.end(), 0.0f, thrust::plus<float>());
        scale += thrust::reduce(s_dist2_to_center.begin(), s_dist2_to_center.end(), 0.0f, thrust::plus<float>());
        scale = std::sqrt(scale / (2.0f * nb_pairs));
        scale = 1.0f / scale;

        buildSymmetricPoint2PlaneSystem<128><<<(nb_pairs + 128 - 1) / 128, 128>>>(mtd,
                                                                                  thrust::raw_pointer_cast(matched_source_positions.data()),
                                                                                  thrust::raw_pointer_cast(matched_source_normals.data()),
                                                                                  thrust::raw_pointer_cast(matched_target_positions.data()),
                                                                                  thrust::raw_pointer_cast(matched_target_normals.data()),
                                                                                  source_centroid,
                                                                                  target_centroid,
                                                                                  scale,
                                                                                  nb_pairs);
        cudaDeviceSynchronize();
        CudaCheckError();

        MotionTrackingData* d = mtd;

        Eigen::Matrix<double, 6, 6> JtJ;
        JtJ << double(d->JtJ[0]), double(d->JtJ[1]), double(d->JtJ[2]), double(d->JtJ[3]), double(d->JtJ[4]), double(d->JtJ[5]),
               double(d->JtJ[1]), double(d->JtJ[6]), double(d->JtJ[7]), double(d->JtJ[8]), double(d->JtJ[9]), double(d->JtJ[10]),
               double(d->JtJ[2]), double(d->JtJ[7]), double(d->JtJ[11]), double(d->JtJ[12]), double(d->JtJ[13]), double(d->JtJ[14]),
               double(d->JtJ[3]), double(d->JtJ[8]), double(d->JtJ[12]), double(d->JtJ[15]), double(d->JtJ[16]), double(d->JtJ[17]),
               double(d->JtJ[4]), double(d->JtJ[9]), double(d->JtJ[13]), double(d->JtJ[16]), double(d->JtJ[18]), double(d->JtJ[19]),
               double(d->JtJ[5]), double(d->JtJ[10]), double(d->JtJ[14]), double(d->JtJ[17]), double(d->JtJ[19]), double(d->JtJ[20]);
        Eigen::Matrix<double, 6, 1> Jtr;
        Jtr << double(d->Jtr[0]), double(d->Jtr[1]), double(d->Jtr[2]), double(d->Jtr[3]), double(d->Jtr[4]), double(d->Jtr[5]);

        pose_covariance = JtJ;
        Eigen::Matrix<double, 6, 1> Xp = JtJ.ldlt().solve(Jtr);

        Eigen::Vector3d tran(Xp(3), Xp(4), Xp(5));
        Eigen::Vector3d rot_axis(Xp(0), Xp(1), Xp(2));

        double rot_axis_norm = rot_axis.norm();
        double rot_angle = 0.5f * std::atan(rot_axis_norm);
        rot_axis /= rot_axis_norm;
        tran /= scale;
        tran *= std::cos(rot_angle);

        Eigen::Isometry3d iso_rot(Eigen::AngleAxisd(rot_angle, rot_axis));
        Eigen::Isometry3d iso_iter = Eigen::Isometry3d(Eigen::Translation3d(double(target_centroid.x), double(target_centroid.y), double(target_centroid.z))) *
                                     iso_rot * Eigen::Isometry3d(Eigen::Translation3d(tran)) * iso_rot *
                                     Eigen::Isometry3d(Eigen::Translation3d(-1.0 * double(source_centroid.x), -1.0 * double(source_centroid.y), -1.0 * double(source_centroid.z)));

        Eigen::Matrix4d tf_iter;
        tf_iter << iso_iter.matrix()(0, 0), iso_iter.matrix()(0, 1), iso_iter.matrix()(0, 2), iso_iter.matrix()(0, 3),
                   iso_iter.matrix()(1, 0), iso_iter.matrix()(1, 1), iso_iter.matrix()(1, 2), iso_iter.matrix()(1, 3),
                   iso_iter.matrix()(2, 0), iso_iter.matrix()(2, 1), iso_iter.matrix()(2, 2), iso_iter.matrix()(2, 3),
                 0.0f, 0.0f, 0.0f, 1.0f;
        tf_iter.block<3,3>(0,0) = Eigen::Quaterniond(tf_iter.block<3,3>(0,0)).normalized().toRotationMatrix();

        tf_inc = tf_iter * tf_inc;
    }

    pose_covariance = pose_covariance.cast<double>().lu().inverse();

    for(unsigned int i = 0; i < 6; i++)
    {
        if(pose_covariance(i, i) > covThresh)
        {
            valid = false;
            break;
        }
    }

    if(valid)
    {
        if(length(t_inc) > 0.3f)
            valid = false;
        else
        {
            R = transpose(R_inc);
            t = -R * t_inc;
        }
    }

    return valid;
}

bool DenseRegistration::featureConstrainedSymmetricICP(const thrust::device_vector<float3>& source_positions,
                                                       const thrust::device_vector<float3>& source_colors,
                                                       const thrust::device_vector<Mat33>& source_orientations,
                                                       const thrust::device_vector<float3>& target_colors,
                                                       const thrust::device_vector<Mat33>& target_orientations,
                                                       const thrust::device_vector<float>& target_confidences,
                                                       const thrust::host_vector<float3>& source_features3D,
                                                       const thrust::host_vector<float3>& target_features3D,
                                                       int source_size,
                                                       const cv::Ptr<Texture<float>>& tex_depth,
                                                       const cv::Ptr<Texture<int>>& tex_index,
                                                       const Mat33& R_init,
                                                       const float3& t_init,
                                                       const CamParam& cam,
                                                       Mat33& R,
                                                       float3& t)
{
    bool valid = true;

    int iter = 0;

    Eigen::Matrix4d tf_inc = Eigen::Matrix4d::Identity();

    R = make_mat33(1.0f, 0.0f, 0.0f,
                   0.0f, 1.0f, 0.0f,
                   0.0f, 0.0f, 1.0f);
    t = make_float3(0.0f, 0.0f, 0.0f);
    Mat33 R_corres;
    float3 t_corres;
    Mat33 R_inc;
    float3 t_inc;

    dim3 dim_block(128);
    dim3 dim_grid((source_size + dim_block.x - 1) / dim_block.x);

    Eigen::Matrix<double, 6, 1> Xp;
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtr;
    Eigen::Matrix<double, 3, 6> J_features;

    Eigen::Matrix4d tf_iter;

    double prev_error = std::numeric_limits<double>::max();

    while(iter++ < nbIter)
    {
        Eigen::Matrix3f basis = tf_inc.block<3,3>(0,0).cast<float>();
        R_inc = make_mat33(basis(0,0), basis(0,1), basis(0,2),
                           basis(1,0), basis(1,1), basis(1,2),
                           basis(2,0), basis(2,1), basis(2,2));
        Eigen::Vector3f origin = tf_inc.block<3,1>(0,3).cast<float>();
        t_inc = make_float3(origin(0), origin(1), origin(2));

        R_corres = R_inc * R_init;
        t_corres = R_inc * t_init + t_inc;

        cudaMemset(mtd, 0, sizeof(MotionTrackingData));

        computeSymmetricICPSystem<128><<<dim_grid, dim_block>>>(mtd,
                                                                thrust::raw_pointer_cast(source_positions.data()),
                                                                thrust::raw_pointer_cast(source_colors.data()),
                                                                thrust::raw_pointer_cast(source_orientations.data()),
                                                                thrust::raw_pointer_cast(target_colors.data()),
                                                                thrust::raw_pointer_cast(target_orientations.data()),
                                                                thrust::raw_pointer_cast(target_confidences.data()),
                                                                R_corres,
                                                                t_corres,
                                                                cam.fx,
                                                                cam.fy,
                                                                cam.cx,
                                                                cam.cy,
                                                                tex_index->getTextureObject(),
                                                                tex_depth->getTextureObject(),
                                                                cam.width,
                                                                cam.height,
                                                                source_size);
        cudaDeviceSynchronize();
        CudaCheckError();

        MotionTrackingData* d = mtd;

        JtJ << double(d->JtJ[0]), double(d->JtJ[1]), double(d->JtJ[2]), double(d->JtJ[3]), double(d->JtJ[4]), double(d->JtJ[5]),
               double(d->JtJ[1]), double(d->JtJ[6]), double(d->JtJ[7]), double(d->JtJ[8]), double(d->JtJ[9]), double(d->JtJ[10]),
               double(d->JtJ[2]), double(d->JtJ[7]), double(d->JtJ[11]), double(d->JtJ[12]), double(d->JtJ[13]), double(d->JtJ[14]),
               double(d->JtJ[3]), double(d->JtJ[8]), double(d->JtJ[12]), double(d->JtJ[15]), double(d->JtJ[16]), double(d->JtJ[17]),
               double(d->JtJ[4]), double(d->JtJ[9]), double(d->JtJ[13]), double(d->JtJ[16]), double(d->JtJ[18]), double(d->JtJ[19]),
               double(d->JtJ[5]), double(d->JtJ[10]), double(d->JtJ[14]), double(d->JtJ[17]), double(d->JtJ[19]), double(d->JtJ[20]);
        Jtr << double(d->Jtr[0]), double(d->Jtr[1]), double(d->Jtr[2]), double(d->Jtr[3]), double(d->Jtr[4]), double(d->Jtr[5]);
        double error = double(d->r);
        float inliers = d->inliers;

        if(inliers < 100.0f)
        {
            std::cout<<" In ICP, number of matches is too small!   nb matches = "<<inliers<<std::endl;
            valid = false;
            break;
        }

        for(size_t i = 0; i < source_features3D.size(); i++)
        {
            float3 ps = R_corres * source_features3D[i] + t_corres;
            float3 pt = target_features3D[i];
            float3 diff = pt - ps;
            float dist = length(diff);

            if(dist < 0.01f)
            {
                float3 sum = pt + ps;

                float w = 1.0f;

                J_features << 1.0, 0.0, 0.0, 0.0f, double(-sum.z), double(sum.y),
                              0.0, 1.0, 0.0, double(ps.z), 0.0f, double(-sum.x),
                              0.0, 0.0, 1.0, double(-sum.y), double(sum.x), 0.0f;
                Jtr += w * J_features.transpose() * Eigen::Vector3d(double(diff.x), double(diff.y), double(diff.z));
                JtJ += w * J_features.transpose() * J_features;
            }
        }

        Xp = JtJ.ldlt().solve(Jtr);

        Eigen::Vector3d tran(Xp(3), Xp(4), Xp(5));
        Eigen::Vector3d rot_axis(Xp(0), Xp(1), Xp(2));

        double rot_axis_norm = rot_axis.norm();
        double rot_angle = 0.5f * std::atan(rot_axis_norm);
        rot_axis /= rot_axis_norm;
        tran *= std::cos(rot_angle);

        Eigen::Isometry3d iso_rot(Eigen::AngleAxisd(rot_angle, rot_axis));
        Eigen::Isometry3d iso_iter = iso_rot * Eigen::Isometry3d(Eigen::Translation3d(tran)) * iso_rot;

        tf_iter << iso_iter.matrix()(0, 0), iso_iter.matrix()(0, 1), iso_iter.matrix()(0, 2), iso_iter.matrix()(0, 3),
                   iso_iter.matrix()(1, 0), iso_iter.matrix()(1, 1), iso_iter.matrix()(1, 2), iso_iter.matrix()(1, 3),
                   iso_iter.matrix()(2, 0), iso_iter.matrix()(2, 1), iso_iter.matrix()(2, 2), iso_iter.matrix()(2, 3),
                   0.0f, 0.0f, 0.0f, 1.0f;
        tf_iter.block<3,3>(0,0) = Eigen::Quaterniond(tf_iter.block<3,3>(0,0)).normalized().toRotationMatrix();

        tf_inc = tf_iter * tf_inc;

        if(error / prev_error > 0.9999)
            break;

        prev_error = error;
    }

    Eigen::MatrixXd pose_covariance = JtJ.cast<double>().lu().inverse();

    for(unsigned int i = 0; i < 6; i++)
    {
        if(pose_covariance(i, i) > covThresh)
        {
            valid = false;
            break;
        }
    }

    if(valid)
    {
        if(length(t_inc) > 0.3f)
            valid = false;
        else
        {
            Eigen::Matrix3f basis = tf_inc.block<3,3>(0,0).cast<float>();
            R_inc = make_mat33(basis(0,0), basis(0,1), basis(0,2),
                               basis(1,0), basis(1,1), basis(1,2),
                               basis(2,0), basis(2,1), basis(2,2));
            Eigen::Vector3f origin = tf_inc.block<3,1>(0,3).cast<float>();
            t_inc = make_float3(origin(0), origin(1), origin(2));

            R = transpose(R_inc);
            t = -R * t_inc;
        }
    }

    return valid;
}

} // supersurfel_fusion
