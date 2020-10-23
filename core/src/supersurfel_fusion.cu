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


#include <supersurfel_fusion/supersurfel_fusion.hpp>
#include <supersurfel_fusion/supersurfel_fusion_kernels.cuh>
#include <thrust/sort.h>
#include <iostream>
#include <supersurfel_fusion/matrix_math.cuh>
#include <chrono>
#include <supersurfel_fusion/cuda_error_check.h>
#include <thread>
#include "gms_matcher.h"
#include <Eigen/StdVector>


namespace supersurfel_fusion
{

SupersurfelFusion::SupersurfelFusion()
    : initialized(false)
{}

SupersurfelFusion::~SupersurfelFusion()
{
    cudaFree(nbSupersurfelsDev);
    cudaFree(nbRemovedDev);
    delete(tps);
    delete(icp);
    delete(vo);
    delete(ferns);
}

void SupersurfelFusion::initialize(const CamParam& cam_param,
                                   int cell_size,
                                   float lambda_pos,
                                   float lambda_bound,
                                   float lambda_size,
                                   float lambda_disp,
                                   float thresh_disp,
                                   int seg_iter,
                                   bool seg_use_ransac,
                                   int nb_samples,
                                   int filter_iter,
                                   float filter_alpha,
                                   float filter_beta,
                                   float filter_threshold,
                                   float range_min,
                                   float range_max,
                                   int delta_t,
                                   float conf_thresh,
                                   int nb_supersurfels_max,
                                   int icp_iter,
                                   double icp_cov_thresh,
                                   int nb_features,
                                   float features_scale_factor,
                                   int features_nb_levels,
                                   int ini_th_fast,
                                   int min_th_fast,
                                   int untracked_threshold,
                                   bool enable_loop_closure,
                                   bool enable_mod)
{
    cam = cam_param;
    confThresh = conf_thresh;
    deltaT = delta_t;
    rangeMin = range_min;
    rangeMax = range_max;

    dimBlockIm.x = 32;
    dimBlockIm.y = 32;
    dimGridIm.x = (cam.width + dimBlockIm.x - 1) / dimBlockIm.x;
    dimGridIm.y = (cam.height + dimBlockIm.y - 1) / dimBlockIm.y;

    int grid_size_X = (cam.width + cell_size - 1) / cell_size;
    int grid_size_Y = (cam.height + cell_size - 1) / cell_size;

    nbSuperpixels = grid_size_X * grid_size_Y;

    dimBlockList.x = 128;
    dimGridList.x = (nbSuperpixels + dimBlockList.x - 1) / dimBlockList.x;

    filteredDepth.create(cam_param.height, cam_param.width, CV_32FC1);
    texDepth = new Texture<float>(filteredDepth);

    tps = new TPS_RGBD(cell_size,
                       lambda_pos,
                       lambda_bound,
                       lambda_size,
                       lambda_disp,
                       thresh_disp,
                       seg_iter,
                       seg_use_ransac,
                       nb_samples,
                       filter_iter,
                       filter_alpha,
                       filter_beta,
                       filter_threshold);

    nbSupersurfelsMax = nb_supersurfels_max;
    nbActive = 0;
    nbSupersurfels = 0;

    model.positions.resize(nbSupersurfelsMax);
    model.colors.resize(nbSupersurfelsMax);
    model.stamps.resize(nbSupersurfelsMax);
    model.orientations.resize(nbSupersurfelsMax);
    model.shapes.resize(nbSupersurfelsMax);
    model.dims.resize(nbSupersurfelsMax);
    model.confidences.resize(nbSupersurfelsMax);
    model.memset(nbSupersurfelsMax);

    cudaMalloc(&nbSupersurfelsDev, sizeof(int));
    cudaMalloc(&nbRemovedDev, sizeof(int));
    cudaMemset(nbSupersurfelsDev, 0, sizeof(int));
    cudaMemset(nbRemovedDev, 0, sizeof(int));

    pose.R = make_mat33(1.f, 0.f, 0.f,
                        0.f, 1.f, 0.f,
                        0.f, 0.f, 1.f);
    pose.t = make_float3(0.f, 0.f, 0.f);

    icp = new DenseRegistration(icp_iter, icp_cov_thresh);

    vo = new SparseVO(nb_features,
                      features_scale_factor,
                      features_nb_levels,
                      ini_th_fast,
                      min_th_fast,
                      cam_param,
                      range_min,
                      range_max,
                      untracked_threshold);

    enableMod = enable_mod;

    ferns = new Ferns(500, 5, range_max);
    stampLastLC = 0;
    previousFernId = 0;
    enableLoopClosure = enable_loop_closure;
    LCdone = false;

    modelSize = 0.0;
    runtime = 0.0;

    stamp = 0;

    initialized = true;
}

void SupersurfelFusion::processFrame(const cv::Mat& rgb_h,
                                     const cv::Mat& depth_h)
{
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    cv::Mat gray_h;
    cv::cuda::GpuMat gray;
    rgb.upload(rgb_h);
    depth.upload(depth_h);
    cv::cuda::cvtColor(rgb, gray, CV_RGB2GRAY);
    gray.convertTo(gray, CV_8UC1);
    gray.download(gray_h);

    /***** Filter depth *****/
    cv::cuda::bilateralFilter(depth, depth, -1, 0.03/* depth_sigma*/, 4.5/* space_sigma*/);
    depth.download(depth_h);

    vo->setFrame(rgb_h, depth_h, gray_h);

    //vo->extractFeatures();
    std::thread t1(&SparseVO::extractFeatures,
                   vo);


    /***** Segment in superpixels *****/
    tps->compute(rgb, depth);
    tps->filter();
    tps->computeDepthImage(filteredDepth);
    //depth.copyTo(filteredDepth);

    /***** Generation of supersurfels *****/
    generateSupersurfels();

    t1.join();

    if(enableMod)
    {
        std::vector<cv::KeyPoint> static_keypoints = vo->getKeypoints();
        cv::Mat static_descriptors = vo->getDescriptors().clone();

        cv::Mat filtered_depth_h;
        filteredDepth.download(filtered_depth_h);

        mod.detectMotion(rgb_h,
                         gray_h,
                         filtered_depth_h/*depth_h*/,
                         static_keypoints,
                         static_descriptors,
                         frame.positions,
                         frame.orientations,
                         frame.confidences,
                         tps->getSuperpixels(),
                         tps->getIndexImage());

        vo->setKeypoints() = static_keypoints;
        vo->setDescriptors() = static_descriptors.clone();
    }

    /***** Camera pose tracking *****/

    /***** Sparse feature-based odometry *****/

    vo->matchFeatures();
    bool vo_success = vo->track();

    pose = eigenToTransform3(vo->getPose());

    /***** Dense frame to model registration *****/

    if(nbActive > 0)
    {
        Mat33 R_view = transpose(pose.R);
        float3 t_view = -R_view * pose.t;

        Mat33 R_rel = make_mat33(1.0f, 0.0f, 0.0f,
                                 0.0f, 1.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f);
        float3 t_rel = make_float3(0.0f, 0.0f, 0.0f);

        bool icp_success = icp->align(model.positions,
                                      model.colors,
                                      model.orientations,
                                      model.confidences,
                                      frame.positions,
                                      frame.colors,
                                      frame.orientations,
                                      frame.confidences,
                                      nbActive,
                                      texDepth,
                                      tps->getTexIndex(),
                                      R_view,
                                      t_view,
                                      cam,
                                      R_rel,
                                      t_rel);

        if(icp_success)
        {
            std::cout<<"ICP success"<<std::endl;

            pose.t = pose.R * t_rel + pose.t;
            pose.R = pose.R * R_rel;

            Eigen::Matrix3f eig_R;
            eig_R << pose.R.rows[0].x, pose.R.rows[0].y, pose.R.rows[0].z,
                     pose.R.rows[1].x, pose.R.rows[1].y, pose.R.rows[1].z,
                     pose.R.rows[2].x, pose.R.rows[2].y, pose.R.rows[2].z;
            eig_R = Eigen::Quaternionf(eig_R).normalized().toRotationMatrix();
            pose.R = make_mat33(eig_R(0,0),eig_R(0,1), eig_R(0,2),
                           eig_R(1,0),eig_R(1,1), eig_R(1,2),
                           eig_R(2,0),eig_R(2,1), eig_R(2,2));
        }
    }

    /***** Loop Closure *****/
    ferns->processFerns(rgb, depth, 0.3095f/*0.35f*/);

    if(enableLoopClosure &&
       !ferns->isNewFrame() &&
       ferns->getBestKeyFrameId() != previousFernId  &&
       stamp - stampLastLC > 100 &&
       stamp - ferns->getBestPose().stamp > 50)
    {
        closeLoop(vo->getKeypoints(),
                  vo->getDescriptors());
    }
    else
        LCdone = false;

    if(LCdone)
        vo->reset(getPoseEigen());
    else
    {
        vo->setPose() = getPoseEigen();
        if(enableMod)
        {
            cv::Mat index_mat;
            tps->getIndexImage().download(index_mat);

            vo->updateLocalMapMOD(index_mat, mod.getIsStatic()/*, mod.getMask()*/);
        }
        else
            vo->updateLocalMap();
    }

    previousFernId = ferns->getBestKeyFrameId();

    /***** Update and filter model *****/
    if(nbSupersurfels > 0)
    {
        cudaMemset(nbRemovedDev, 0, sizeof(int));
        thrust::device_vector<bool> matched(nbSuperpixels, false);

        if(nbActive > 0)
        {
            thrust::device_vector<float2> idx_scores(nbSuperpixels, make_float2(-1.0f, 0.05f));

            findBestMatches<<<(nbActive + 127) / 128, 128>>>(thrust::raw_pointer_cast(&frame.positions[0]),
                                                             thrust::raw_pointer_cast(&frame.colors[0]),
                                                             thrust::raw_pointer_cast(&frame.orientations[0]),
                                                             thrust::raw_pointer_cast(&frame.confidences[0]),
                                                             thrust::raw_pointer_cast(&model.positions[0]),
                                                             thrust::raw_pointer_cast(&model.colors[0]),
                                                             thrust::raw_pointer_cast(&model.orientations[0]),
                                                             thrust::raw_pointer_cast(&model.confidences[0]),
                                                             thrust::raw_pointer_cast(&matched[0]),
                                                             thrust::raw_pointer_cast(&idx_scores[0]),
                                                             tps->getTexIndex()->getTextureObject(),
                                                             pose.R,
                                                             pose.t,
                                                             cam.fx,
                                                             cam.fy,
                                                             cam.cx,
                                                             cam.cy,
                                                             rangeMin,
                                                             rangeMax,
                                                             cam.width,
                                                             cam.height,
                                                             nbActive);
            cudaDeviceSynchronize();
            CudaCheckError();


            updateSupersurfels<<<dimGridList, dimBlockList>>>(thrust::raw_pointer_cast(&frame.positions[0]),
                                                             thrust::raw_pointer_cast(&frame.colors[0]),
                                                             thrust::raw_pointer_cast(&frame.shapes[0]),
                                                             thrust::raw_pointer_cast(&frame.confidences[0]),
                                                             thrust::raw_pointer_cast(&model.positions[0]),
                                                             thrust::raw_pointer_cast(&model.colors[0]),
                                                             thrust::raw_pointer_cast(&model.stamps[0]),
                                                             thrust::raw_pointer_cast(&model.orientations[0]),
                                                             thrust::raw_pointer_cast(&model.shapes[0]),
                                                             thrust::raw_pointer_cast(&model.dims[0]),
                                                             thrust::raw_pointer_cast(&model.confidences[0]),
                                                             thrust::raw_pointer_cast(&matched[0]),
                                                             thrust::raw_pointer_cast(&idx_scores[0]),
                                                             pose.R,
                                                             pose.t,
                                                             stamp,
                                                             nbSuperpixels);
            cudaDeviceSynchronize();
            CudaCheckError();
        }

        insertSupersurfels<<<dimGridList, dimBlockList>>>(thrust::raw_pointer_cast(&frame.positions[0]),
                                                          thrust::raw_pointer_cast(&frame.colors[0]),
                                                          thrust::raw_pointer_cast(&frame.orientations[0]),
                                                          thrust::raw_pointer_cast(&frame.shapes[0]),
                                                          thrust::raw_pointer_cast(&frame.dims[0]),
                                                          thrust::raw_pointer_cast(&frame.confidences[0]),
                                                          thrust::raw_pointer_cast(&model.positions[0]),
                                                          thrust::raw_pointer_cast(&model.colors[0]),
                                                          thrust::raw_pointer_cast(&model.stamps[0]),
                                                          thrust::raw_pointer_cast(&model.orientations[0]),
                                                          thrust::raw_pointer_cast(&model.shapes[0]),
                                                          thrust::raw_pointer_cast(&model.dims[0]),
                                                          thrust::raw_pointer_cast(&model.confidences[0]),
                                                          pose.R,
                                                          pose.t,
                                                          stamp,
                                                          thrust::raw_pointer_cast(&matched[0]),
                                                          nbSupersurfelsDev,
                                                          nbSuperpixels,
                                                          nbSupersurfelsMax);
        cudaDeviceSynchronize();
        CudaCheckError();

        cudaMemcpy(&nbSupersurfels, nbSupersurfelsDev, sizeof(int), cudaMemcpyDeviceToHost);

        thrust::device_vector<int> states(nbSupersurfels, 0); // 0: active, 1: inactive, 2: invalid
        nbActive = 0;
        int* nb_active_d;
        cudaMalloc((void**)&nb_active_d, sizeof(int));
        cudaMemcpy(nb_active_d, &nbActive, sizeof(int), cudaMemcpyHostToDevice);

        Mat33 R_view = transpose(pose.R);
        float3 t_view = -R_view * pose.t;

        filterModel<<<(nbSupersurfels + 127) / 128, 128>>>(thrust::raw_pointer_cast(&model.positions[0]),
                                                           thrust::raw_pointer_cast(&model.stamps[0]),
                                                           thrust::raw_pointer_cast(&model.confidences[0]),
                                                           thrust::raw_pointer_cast(&states[0]),
                                                           stamp,
                                                           deltaT,
                                                           confThresh,
                                                           texDepth->getTextureObject(),
                                                           nbRemovedDev,
                                                           nb_active_d,
                                                           R_view,
                                                           t_view,
                                                           cam.fx,
                                                           cam.fy,
                                                           cam.cx,
                                                           cam.cy,
                                                           rangeMin,
                                                           rangeMax,
                                                           cam.width,
                                                           cam.height,
                                                           nbSupersurfels);
        cudaDeviceSynchronize();
        CudaCheckError();

        cudaMemcpy(&nbRemoved, nbRemovedDev, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&nbActive, nb_active_d, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(nb_active_d);

        thrust::sort_by_key(thrust::cuda::par(allocator),
                            states.begin(),
                            states.end(),
                            model.begin()); // model supersurfels sorted from active, inactive, invalid

        nbSupersurfels -= nbRemoved;
        cudaMemcpy(nbSupersurfelsDev, &nbSupersurfels, sizeof(int), cudaMemcpyHostToDevice);
    }
    else
    {
        thrust::copy(frame.begin(), frame.end(), model.begin());
        nbSupersurfels = nbSuperpixels;
        nbActive = nbSupersurfels;
        cudaMemcpy(nbSupersurfelsDev, &nbSupersurfels, sizeof(int), cudaMemcpyHostToDevice);
    }

    if(ferns->isNewFrame())
    {
        thrust::host_vector<float3> positions(frame.positions);
        thrust::host_vector<float3> colors(frame.colors);
        thrust::host_vector<Mat33> orientations(frame.orientations);
        thrust::host_vector<float> confidences(frame.confidences);

        std::vector<cv::KeyPoint> filtered_keypoints;
        cv::Mat filtered_descriptors;
        std::vector<cv::Point3f> points3D;

        vo->computeFilteredKeypoints3D(filtered_keypoints,
                                       filtered_descriptors,
                                       points3D,
                                       rangeMin,
                                       rangeMax);

        ferns->addKeyFrame(pose.R,
                           pose.t,
                           stamp,
                           filtered_keypoints,
                           filtered_descriptors,
                           points3D,
                           positions,
                           colors,
                           orientations,
                           confidences);
    }

    std::chrono::steady_clock::time_point stop = std::chrono::steady_clock::now();

    std::cout<<"Total time = "<<std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count()<<" ms"<<std::endl;

    std::cout<<"nb supersurfels: "<<nbSupersurfels<<std::endl;
    std::cout<<"stamp: "<<stamp<<std::endl;
    stamp++;

    modelSize = std::max(modelSize, double(nbSupersurfels * (24 * sizeof(float) + sizeof(int2))) / (1024.0 * 1024.0));
    std::cout<<"Max model size = "<<modelSize<<" MB"<<std::endl;

    runtime += double(std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count());
    std::cout<<"Mean runtime = "<<runtime / double(stamp)<<std::endl;
    std::cout<<"\n";
}

void SupersurfelFusion::generateSupersurfels()
{
    frame.positions.resize(nbSuperpixels);
    frame.colors.resize(nbSuperpixels);
    frame.stamps.resize(nbSuperpixels);
    frame.orientations.resize(nbSuperpixels);
    frame.shapes.resize(nbSuperpixels);
    frame.dims.resize(nbSuperpixels);
    frame.confidences.resize(nbSuperpixels);
    frame.memset(nbSuperpixels);

    computeSupersurfelCoeffs<<<dimGridIm, dimBlockIm>>>(thrust::raw_pointer_cast(&frame.positions[0]),
                                                        thrust::raw_pointer_cast(&frame.shapes[0]),
                                                        thrust::raw_pointer_cast(&frame.colors[0]),
                                                        thrust::raw_pointer_cast(&frame.confidences[0]),
                                                        tps->getTexRGBA()->getTextureObject(),
                                                        texDepth->getTextureObject(),
                                                        tps->getTexIndex()->getTextureObject(),
                                                        tps->getTexInliers()->getTextureObject(),
                                                        tps->getTexBound()->getTextureObject(),
                                                        cam.width,
                                                        cam.height,
                                                        cam.fx,
                                                        cam.fy,
                                                        cam.cx,
                                                        cam.cy);
    cudaDeviceSynchronize();
    CudaCheckError();

    computeSupersurfels<<<dimGridList, dimBlockList>>>(thrust::raw_pointer_cast(&frame.positions[0]),
                                                       thrust::raw_pointer_cast(&frame.colors[0]),
                                                       thrust::raw_pointer_cast(&frame.stamps[0]),
                                                       thrust::raw_pointer_cast(&frame.orientations[0]),
                                                       thrust::raw_pointer_cast(&frame.shapes[0]),
                                                       thrust::raw_pointer_cast(&frame.dims[0]),
                                                       thrust::raw_pointer_cast(&frame.confidences[0]),
                                                       rangeMin,
                                                       rangeMax,
                                                       stamp,
                                                       nbSuperpixels);
    cudaDeviceSynchronize();
    CudaCheckError();
}

void SupersurfelFusion::exportModel(std::string filename)
{
    std::cout<<"Save model"<<std::endl;

    std::ofstream file;
    file.open(filename, std::ofstream::out);

    if(file.good() != true)
    {
        std::cerr<<"Can't open file"<<std::endl;
        return;
    }

    thrust::host_vector<float3> positions(model.positions.begin(), model.positions.begin() + nbSupersurfels);
    thrust::host_vector<Mat33> orientations(model.orientations.begin(), model.orientations.begin() + nbSupersurfels);
    thrust::host_vector<Cov3> shapes(model.shapes.begin(), model.shapes.begin() + nbSupersurfels);
    thrust::host_vector<float2> dims(model.dims.begin(), model.dims.begin() + nbSupersurfels);
    thrust::host_vector<float> confidences(model.confidences.begin(), model.confidences.begin() + nbSupersurfels);
    thrust::host_vector<float3> colors(model.colors.begin(), model.colors.begin() + nbSupersurfels);
    thrust::host_vector<int2> stamps(model.stamps.begin(), model.stamps.begin() + nbSupersurfels);

    for(int i = 0; i < nbSupersurfels; i++)
    {
        if(confidences[i] > confThresh)
        {
            file<<std::to_string(stamps[i].x)<<" "<<std::to_string(stamps[i].y)<<" "<<std::to_string(confidences[i])<<"\n"; // t0 t conf
            file<<std::to_string(positions[i].x)<<" "<<std::to_string(positions[i].y)<<" "<<std::to_string(positions[i].z)<<"\n"; // position
            file<<std::to_string(colors[i].x)<<" "<<std::to_string(colors[i].y)<<" "<<std::to_string(colors[i].z)<<"\n"; // color
            file<<std::to_string(dims[i].x)<<" "<<std::to_string(dims[i].y)<<"\n"; //dims
            file<<std::to_string(orientations[i].rows[0].x)<<" "<<std::to_string(orientations[i].rows[0].y)<<" "<<std::to_string(orientations[i].rows[0].z)<<" "
                <<std::to_string(orientations[i].rows[1].x)<<" "<<std::to_string(orientations[i].rows[1].y)<<" "<<std::to_string(orientations[i].rows[1].z)<<" "
                <<std::to_string(orientations[i].rows[2].x)<<" "<<std::to_string(orientations[i].rows[2].y)<<" "<<std::to_string(orientations[i].rows[2].z)<<"\n"; //orientation
            file<<std::to_string(shapes[i].xx)<<" "<<std::to_string(shapes[i].xy)<<" "<<std::to_string(shapes[i].xz)<<" "<<std::to_string(shapes[i].yy)<<" "<<std::to_string(shapes[i].yz)<<" "<<std::to_string(shapes[i].zz)<<"\n"; // shape
            file<<"\n";
        }
    }

    file.close();
}

void SupersurfelFusion::computeSuperpixelSegIm(cv::Mat& seg_im)
{
    cv::cuda::GpuMat seg_im_d;
    tps->computePreviewImage(seg_im_d);
    seg_im_d.download(seg_im);
}

void SupersurfelFusion::computeSlantedPlaneIm(cv::Mat& slanted_plane_im)
{
    cv::cuda::GpuMat slanted_plane_im_d;
    tps->computeDepthImage(slanted_plane_im_d);
    slanted_plane_im_d.download(slanted_plane_im);
}

Eigen::Isometry3f SupersurfelFusion::getPoseEigen() const
{
    Eigen::Isometry3f iso;
    iso.setIdentity();

    iso.translation() = Eigen::Vector3f(pose.t.x, pose.t.y, pose.t.z);
    Eigen::Matrix3f m;
    m << pose.R.rows[0].x, pose.R.rows[0].y, pose.R.rows[0].z,
         pose.R.rows[1].x, pose.R.rows[1].y, pose.R.rows[1].z,
         pose.R.rows[2].x, pose.R.rows[2].y, pose.R.rows[2].z;
    iso.linear() = m;

    return iso;
}

Transform3 SupersurfelFusion::eigenToTransform3(const Eigen::Isometry3f& iso) const
{
    Transform3 tf;

    tf.R = make_mat33(iso.matrix()(0,0), iso.matrix()(0,1), iso.matrix()(0,2),
                      iso.matrix()(1,0), iso.matrix()(1,1), iso.matrix()(1,2),
                      iso.matrix()(2,0), iso.matrix()(2,1), iso.matrix()(2,2));
    tf.t = make_float3(iso.matrix()(0,3), iso.matrix()(1, 3), iso.matrix()(2,3));

    return tf;
}

void SupersurfelFusion::closeLoop(const std::vector<cv::KeyPoint>& curr_kpts,
                                  const cv::Mat& curr_desc)
{
    std::cout<<"Loop closure detected"<<std::endl;

    Mat33 R_init = make_mat33(1.0f, 0.0f, 0.0f,
                              0.0f, 1.0f, 0.0f,
                              0.0f, 0.0f, 1.0f);
    float3 t_init = make_float3(0.0f, 0.0f, 0.0f);

    bool sparse_registration_ok = false;

    KeyFrame kf = ferns->getBestMatch();
    StampedPose stamped_pose =  ferns->getBestPose();

    std::vector<cv::KeyPoint> kf_kpts = kf.keypoints;
    cv::Mat kf_desc = kf.descriptors;
    std::vector<cv::Point3f> kf_pts3D = kf.keypoints3D;

    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    std::vector<cv::DMatch> matches_all;

    cv::cuda::GpuMat curr_desc_d(curr_desc), kf_desc_d(kf_desc);
    matcher->match(kf_desc_d, curr_desc_d, matches_all);

    std::vector<bool> inlier_states;
    cv::Size img_size(cam.width, cam.height);
    gms_matcher gms(kf_kpts, img_size, curr_kpts, img_size, matches_all);

    int nb_inliers = gms.GetInlierMask(inlier_states, true, true);

    if(nb_inliers > 30)
    {
        std::vector<cv::Point2d> matched_curr_pts2D;
        std::vector<cv::Point3d> matched_kf_pts3D;

        for(size_t i = 0; i < inlier_states.size(); i++)
        {
            if(inlier_states[i])
            {
                matched_kf_pts3D.push_back(cv::Point3d((double)kf_pts3D[matches_all[i].queryIdx].x,
                                                       (double)kf_pts3D[matches_all[i].queryIdx].y,
                                                       (double)kf_pts3D[matches_all[i].queryIdx].z));
                matched_curr_pts2D.push_back(cv::Point2d((double)curr_kpts[matches_all[i].trainIdx].pt.x,
                                                         (double)curr_kpts[matches_all[i].trainIdx].pt.y));
            }
        }

        cv::Mat rvec, tvec;
        std::vector<int> ransac_inliers_idx;
        cv::Mat K = cv::Mat::zeros(3, 3, CV_64FC1);
        K.at<double>(0,0) = double(cam.fx);
        K.at<double>(0,2) = double(cam.cx);
        K.at<double>(1,1) = double(cam.fy);
        K.at<double>(1,2) = double(cam.cy);
        K.at<double>(2,2) = 1.0;

        if(cv::solvePnPRansac(matched_kf_pts3D,
                              matched_curr_pts2D,
                              K,
                              cv::noArray(),
                              rvec,
                              tvec,
                              false,
                              200,
                              5.0,
                              0.99,
                              ransac_inliers_idx,
                              cv::SOLVEPNP_EPNP))
        {
            if(float(ransac_inliers_idx.size()) > 0.3f * float(matched_kf_pts3D.size()))
            {
                cv::Mat cv_rot, cv_trans;
                cv::Rodrigues(rvec, cv_rot);
                cv_rot.convertTo(cv_rot, CV_32FC1);
                tvec.convertTo(cv_trans, CV_32FC1);

                R_init = make_mat33(cv_rot.at<float>(0,0), cv_rot.at<float>(0,1), cv_rot.at<float>(0,2),
                                    cv_rot.at<float>(1,0), cv_rot.at<float>(1,1), cv_rot.at<float>(1,2),
                                    cv_rot.at<float>(2,0), cv_rot.at<float>(2,1), cv_rot.at<float>(2,2));
                t_init = make_float3(cv_trans.at<float>(0), cv_trans.at<float>(1), cv_trans.at<float>(2));


                sparse_registration_ok = true;
            }
            else
                std::cout<<"Nb inliers of cv::solvePnPRansac is too small"<<std::endl;
        }
        else
            std::cout<<"cv::solvePnPRansac failed"<<std::endl;
    }
    else
        std::cout<<"Not enough matches, nb matches = "<<matches_all.size()<<std::endl;

    Mat33 R_rel = make_mat33(1.0f, 0.0f, 0.0f,
                             0.0f, 1.0f, 0.0f,
                             0.0f, 0.0f, 1.0f);
    float3 t_rel = make_float3(0.0f, 0.0f, 0.0f);

    thrust::device_vector<float3> kf_positions(kf.positions);
    thrust::device_vector<float3> kf_colors(kf.colors);
    thrust::device_vector<Mat33> kf_orientations(kf.orientations);
    thrust::device_vector<float> kf_confidences(kf.positions.size(), 1.0f);

    bool icp_pose_ok = icp->align(kf_positions,
                                  kf_colors,
                                  kf_orientations,
                                  kf_confidences,
                                  frame.positions,
                                  frame.colors,
                                  frame.orientations,
                                  frame.confidences,
                                  kf.positions.size(),
                                  texDepth,
                                  tps->getTexIndex(),
                                  R_init,
                                  t_init,
                                  cam,
                                  R_rel,
                                  t_rel);

    if(sparse_registration_ok)
    {
        R_init = transpose(R_init);
        t_init = -1.0f * R_init * t_init;
    }

    R_rel = R_init * R_rel;
    t_rel =  R_init * t_rel + t_init;

    Eigen::Matrix3f eig_R;
    eig_R << R_rel.rows[0].x, R_rel.rows[0].y, R_rel.rows[0].z,
             R_rel.rows[1].x, R_rel.rows[1].y, R_rel.rows[1].z,
             R_rel.rows[2].x, R_rel.rows[2].y, R_rel.rows[2].z;
    eig_R = Eigen::Quaternionf(eig_R).normalized().toRotationMatrix();
    R_rel = make_mat33(eig_R(0,0),eig_R(0,1), eig_R(0,2),
                       eig_R(1,0),eig_R(1,1), eig_R(1,2),
                       eig_R(2,0),eig_R(2,1), eig_R(2,2));

    if(!icp_pose_ok && !sparse_registration_ok)
    {
        std::cout<<"Loop closure rejected: pose estimation failed"<<std::endl;
        return;
    }

    float3 t_LC = stamped_pose.R * t_rel + stamped_pose.t;
    Mat33 R_LC = stamped_pose.R * R_rel;

    eig_R << R_LC.rows[0].x, R_LC.rows[0].y, R_LC.rows[0].z,
             R_LC.rows[1].x, R_LC.rows[1].y, R_LC.rows[1].z,
             R_LC.rows[2].x, R_LC.rows[2].y, R_LC.rows[2].z;
    eig_R = Eigen::Quaternionf(eig_R).normalized().toRotationMatrix();
    R_LC = make_mat33(eig_R(0,0),eig_R(0,1), eig_R(0,2),
                      eig_R(1,0),eig_R(1,1), eig_R(1,2),
                      eig_R(2,0),eig_R(2,1), eig_R(2,2));

    // Generate constraints
    Constraints constraints;

    thrust::host_vector<float3> frame_positions(frame.positions);
    thrust::host_vector<float> frame_confidences(frame.confidences);

    for(int i = 0; i < nbSuperpixels; i += nbSuperpixels / 50 )
    {
        if(frame_confidences[i] > 0.0f)
        {
            float3 src = pose.R * frame_positions[i] + pose.t;
            float3 target = R_LC * frame_positions[i] + t_LC;

            constraints.sources.push_back(src);
            constraints.targets.push_back(target);
            constraints.sourceStamps.push_back(stamp);
            constraints.targetStamps.push_back(stamped_pose.stamp);
            constraints.idx.push_back(0);
            constraints.pin.push_back(false);

            // pin constraint
            constraints.sources.push_back(target);
            constraints.targets.push_back(target);
            constraints.sourceStamps.push_back(stamped_pose.stamp);
            constraints.targetStamps.push_back(stamped_pose.stamp);
            constraints.idx.push_back(0);
            constraints.pin.push_back(true);
        }
    }

    thrust::host_vector<float3> model_positions(model.positions.begin(), model.positions.begin() + nbSupersurfels);
    thrust::host_vector<int2> model_stamps(model.stamps.begin(), model.stamps.begin() + nbSupersurfels);

    def.buildGraphFromModel(model_positions,
                            model_stamps,
                            50,
                            constraints,
                            ferns->getPoseGraph());

    stampLastLC = stamp;

    if(def.constrain(model.positions, model.orientations, model.shapes, nbSupersurfels, ferns->setPoseGraph(),/* true,*/ stamp))
    {
        std::cout<<"Deformation applied"<<std::endl;

        pose.R = R_LC;
        pose.t = t_LC;

        LCdone = true;
    }
}

void SupersurfelFusion::extractLocalPointCloud(thrust::host_vector<float3>& cloud_positions,
                                               thrust::host_vector<float3>& cloud_normals)
{
    thrust::device_vector<float3> cloud_positions_d(nbSupersurfels);
    thrust::device_vector<float3> cloud_normals_d(nbSupersurfels);

    int nb_points = 0;
    int* nb_points_d;
    cudaMalloc((void**)&nb_points_d, sizeof(int));
    cudaMemcpy(nb_points_d, &nb_points, sizeof(int), cudaMemcpyHostToDevice);

    Mat33 R_view = transpose(pose.R);
    float3 t_view = -R_view * pose.t;

    extractLocalPointCloudKernel<<<(nbSupersurfels + 127) / 128, 128>>>(thrust::raw_pointer_cast(&model.positions[0]),
                                                                        thrust::raw_pointer_cast(&model.orientations[0]),
                                                                        thrust::raw_pointer_cast(&model.confidences[0]),
                                                                        thrust::raw_pointer_cast(&cloud_positions_d[0]),
                                                                        thrust::raw_pointer_cast(&cloud_normals_d[0]),
                                                                        confThresh,
                                                                        R_view,
                                                                        t_view,
                                                                        rangeMax,
                                                                        nb_points_d,
                                                                        nbSupersurfels);
    cudaDeviceSynchronize();
    CudaCheckError();

    cudaMemcpy(&nb_points, nb_points_d, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(nb_points_d);

    cloud_positions_d.resize(nb_points);
    cloud_normals_d.resize(nb_points);

   cloud_positions = cloud_positions_d;
   cloud_normals = cloud_normals_d;
}

void SupersurfelFusion::setYoloDetector(const std::string& yolo_config_file,
                                        const std::string& yolo_weights_file,
                                        const std::string& yolo_names_file)
{
    mod.setYoloDetector(yolo_config_file, yolo_weights_file, yolo_names_file);
}

} // namespace supersurfel_fusion
