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


#include <supersurfel_fusion/motion_detection.hpp>
#include <supersurfel_fusion/motion_detection_kernels.cuh>
#include <supersurfel_fusion/cuda_error_check.h>
#include <supersurfel_fusion/vector_math.cuh>
#include "gms_matcher.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <stack>


namespace supersurfel_fusion
{

// Recursively visit vertices.
static void AdjListDFS(std::vector<std::vector<int>> &adjList, int &vertex, std::vector<bool> &visited, std::vector<int>& cluster)
{
    // Mark the vertex as visited.
    visited[vertex] = true;

    // Outputting vertex+1 because that's the way our graph picture looks.
    //std::cout<<vertex/*+1*/<<" ";

    cluster.push_back(vertex);

    // Look at this vertex's neighbors.
    for(size_t i = 0; i < adjList[vertex].size(); i++)
    {
        int neighbor = adjList[vertex][i];
        // Recursively call DFS on the neighbor, if it wasn't visited.
        if(visited[neighbor] == false)
        {
            AdjListDFS(adjList, neighbor, visited, cluster);
        }
    }
}

// Given an Adjacency List, do a DFS on vertex "start"
static void AdjListDFSInitialize(std::vector<std::vector<int>> &adjList, int start, std::vector<std::vector<int>>& clusters)
{
    //std::cout << "\nDoing a DFS on an adjacency list.\n";

    int n = adjList.size();
    // Create a "visited" array (true or false) to keep track of if we visited a vertex.
    std::vector<bool> visited;

    for(int i = 0; i < n; i++)
    {
        visited.push_back(false);
    }

    for(int i = 0; i < n; i++)
    {
        if(!visited[i])
        {
            std::vector<int> cluster;

            AdjListDFS(adjList, i, visited, cluster);

            clusters.push_back(cluster);

            //std::cout<<std::endl<<std::endl;
        }
    }

    return;
}

static cv::Mat flowToDisplay(const cv::Mat flow)
{
    cv::Mat flow_split[2];
    cv::Mat magnitude, angle;
    cv::Mat hsv_split[3], hsv, rgb;
    cv::split(flow, flow_split);
    cv::cartToPolar(flow_split[0], flow_split[1], magnitude, angle, true);
    cv::normalize(magnitude, magnitude, 0, 1, cv::NORM_MINMAX);
    hsv_split[0] = angle; // already in degrees - no normalization needed
    hsv_split[1] = cv::Mat::ones(angle.size(), angle.type());
    hsv_split[2] = magnitude;
    cv::merge(hsv_split, 3, hsv);
    cv::cvtColor(hsv, rgb, cv::COLOR_HSV2BGR);

    return rgb;
}

static std::vector<std::string> objects_names_from_file(std::string const filename)
{
    std::ifstream file(filename);
    std::vector<std::string> file_lines;
    if (!file.is_open()) return file_lines;
    for(std::string line; getline(file, line);) file_lines.push_back(line);
    std::cout << "object names loaded \n";

    return file_lines;
}

static void draw_boxes(cv::Mat mat_img, std::vector<bbox_t> result_vec, std::vector<std::string> obj_names)
{
    //int const colors[6][3] = { { 1,0,1 },{ 0,0,1 },{ 0,1,1 },{ 0,1,0 },{ 1,1,0 },{ 1,0,0 } };

    for (auto &i : result_vec)
    {
        //if(i.obj_id == 0) // draw only persons
        {
            cv::Scalar color = obj_id_to_color(i.obj_id);
            cv::rectangle(mat_img, cv::Rect(i.x, i.y, i.w, i.h), color, 2);

            if (obj_names.size() > i.obj_id)
            {
                std::string obj_name = obj_names[i.obj_id];
                if (i.track_id > 0) obj_name += " - " + std::to_string(i.track_id);
                cv::Size const text_size = getTextSize(obj_name, cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 2, 0);
                int max_width = (text_size.width > i.w + 2) ? text_size.width : (i.w + 2);
                max_width = std::max(max_width, (int)i.w + 2);
                //max_width = std::max(max_width, 283);

                cv::rectangle(mat_img, cv::Point2f(std::max((int)i.x - 1, 0), std::max((int)i.y - 35, 0)),
                              cv::Point2f(std::min((int)i.x + max_width, mat_img.cols - 1), std::min((int)i.y, mat_img.rows - 1)), color, CV_FILLED, 8, 0);
                putText(mat_img, obj_name, cv::Point2f(i.x, i.y - 16), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, cv::Scalar(0, 0, 0), 2);
            }
        }
    }
}

MotionDetection::MotionDetection()
    : useYolo(false)
{
    dof = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_ULTRAFAST);
    //dof = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_MEDIUM);
    //dof = cv::optflow::createOptFlow_DIS(cv::optflow::DISOpticalFlow::PRESET_FAST);

    matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

    //mask = cv::Mat::zeros(480, 640, CV_8UC1);
}

void MotionDetection::setYoloDetector(const std::string& yolo_config_file,
                                      const std::string& yolo_weights_file,
                                      const std::string& yolo_names_file)
{
    yoloDetector = new Detector(yolo_config_file, yolo_weights_file, 0.2f/* thresh*/);
    objectsNames = objects_names_from_file(yolo_names_file);

    useYolo = true;
}

void MotionDetection::detectYolo(const cv::Mat& rgb)
{
    objects.clear();

    objects = yoloDetector->detect(rgb);

    //cv::Mat im_detect;
    //cv::cvtColor(rgb, im_detect, cv::COLOR_RGB2BGR);
    //draw_boxes(im_detect, objects, objectsNames);

    //cv::imshow("detection", im_detect);
    //cv::waitKey(1);
}

void MotionDetection::computeGeometricClusters(const thrust::host_vector<float3>& positions,
                                               const thrust::host_vector<Mat33>& orientations,
                                               const thrust::host_vector<float>& confidences,
                                               std::vector<std::vector<int>>& clusters,
                                               std::vector<std::vector<int>>& adjacency_list,
                                               std::vector<std::vector<int>>& adjacency_list_all)
{
    int nb_superpixels = positions.size();

    cv::cuda::GpuMat adjacency_d(nb_superpixels, nb_superpixels, CV_8UC1, cv::Scalar(0));

    dim3 dim_block(32, 32);
    dim3 dim_grid((flowMat.cols + dim_block.x - 1) / dim_block.x, (indexMat.rows + dim_block.y - 1) / dim_block.y);

    computeSuperpixelsAdjacency<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
                                                         adjacency_d,
                                                         indexMat.cols,
                                                         indexMat.rows);
    cudaDeviceSynchronize();
    CudaCheckError();

    adjacency_list.resize(nb_superpixels);
    adjacency_list_all.resize(nb_superpixels);
    cv::Mat adjacency;
    adjacency_d.download(adjacency);

    for(int i = 0; i < nb_superpixels; i++)
    {
        for(int j = 0; j < nb_superpixels; j++)
        {
            if(adjacency.at<uchar>(i, j))
            {
                if(confidences[i] > 0.0f && confidences[j] > 0.0f)
                {
                    //float depth_dist = fabsf(positions[i].z - positions[j].z);
                    float3 n_i = normalize(orientations[i].rows[2]);
                    float3 n_j = normalize(orientations[j].rows[2]);
                    //float xyz_dist = lenght(positions[i] - positions[j]);
                    //float normal_dist = acosf(dot(n_i, n_j)) * 180.0f / CV_PI;
                    float3 c_ij =  (positions[i] - positions[j]) / length(positions[i] - positions[j]);
                    float geometric_dist = (length(cross(n_i, n_j)) + fabsf(dot(n_i, c_ij)) + fabsf(dot(n_j, c_ij))) / 3.0f;

                    //if(dot(n_i, c_ij) - dot(n_j, c_ij) >= 0)
                    //    geometric_dist *= 0.5f;

                    if(geometric_dist < 0.2f)
                        //if(depth_dist < 0.05f)
                        adjacency_list[i].push_back(j);
                }

                adjacency_list_all[i].push_back(j);
            }
        }
    }

    AdjListDFSInitialize(adjacency_list, 0, clusters);
}

void MotionDetection::detectMotion(const cv::Mat& rgb,
                                   const cv::Mat& gray,
                                   const cv::Mat& depth,
                                   std::vector<cv::KeyPoint>& keypoints,
                                   cv::Mat& descriptors,
                                   const thrust::device_vector<float3>& supersurfels_positions,
                                   const thrust::device_vector<Mat33>& supersurfels_orientations,
                                   thrust::device_vector<float>& supersurfels_confidences,
                                   const thrust::device_vector<SuperpixelRGBD>& superpixels,
                                   const cv::cuda::GpuMat& index_mat)
{
    if(useYolo)
        detectMotionCombined(rgb,
                             gray,
                             depth,
                             keypoints,
                             descriptors,
                             supersurfels_positions,
                             supersurfels_orientations,
                             supersurfels_confidences,
                             superpixels,
                             index_mat);
//        detectMotionYoloOnly(rgb,
//                             keypoints,
//                             descriptors,
//                             supersurfels_positions,
//                             supersurfels_confidences,
//                             superpixels,
//                             index_mat);
    else
        detectMotionSimple(gray,
                           depth,
                           keypoints,
                           descriptors,
                           supersurfels_positions,
                           supersurfels_orientations,
                           supersurfels_confidences,
                           superpixels,
                           index_mat);

}

void MotionDetection::detectMotionSimple(const cv::Mat& gray,
                                         const cv::Mat& depth,
                                         std::vector<cv::KeyPoint>& keypoints,
                                         cv::Mat& descriptors,
                                         const thrust::device_vector<float3>& supersurfels_positions,
                                         const thrust::device_vector<Mat33>& supersurfels_orientations,
                                         thrust::device_vector<float>& supersurfels_confidences,
                                         const thrust::device_vector<SuperpixelRGBD>& superpixels,
                                         const cv::cuda::GpuMat& index_mat)
{
    int nb_superpixels = superpixels.size();

    isStatic.clear();
    isStatic.resize(nb_superpixels, true);

    //if(prevGrayVec.size() == 0)
    if(prevGray.empty())
    {
        //        prevGrayVec.push_back(gray.clone());
        //        prevDepthVec.push_back(depth.clone());
        //        prevKeypointsVec.push_back(keypoints);
        //        prevDescriptorsVec.push_back(descriptors.clone());

        prevGray = gray.clone();
        prevDepth = depth.clone();
        prevKeypoints = keypoints;
        prevDescriptors = descriptors.clone();

        indexMat.create(gray.size(), CV_32SC1);
        flowMat.create(gray.size(), CV_32FC2);
        depthEstimateMat.create(gray.size(), CV_32FC1);

        texIndex = new Texture<int>(indexMat);
        texFlow = new Texture<float2>(flowMat);
        texDepthEstimate = new Texture<float>(depthEstimateMat);

        mask = cv::Mat::zeros(gray.size(), CV_8UC1);

        return;
    }

    index_mat.copyTo(indexMat);

    thrust::host_vector<float3> positions(supersurfels_positions);
    thrust::host_vector<Mat33> orientations(supersurfels_orientations);
    thrust::host_vector<float> confidences(supersurfels_confidences);

    std::vector<std::vector<int>> adjacency_list;
    std::vector<std::vector<int>> adjacency_list_all;

    std::vector<std::vector<int>> clusters;

    std::vector<cv::Point2f> src_pts, dest_pts;
    std::vector<cv::DMatch> matches_all;
    cv::cuda::GpuMat prev_descriptors_d(prevDescriptors), descriptors_d(descriptors);
    //cv::cuda::GpuMat prev_descriptors_d(prevDescriptorsVec[0]), descriptors_d(descriptors);
    matcher->match(prev_descriptors_d, descriptors_d, matches_all);

    //    computeGeometricClusters(positions,
    //                             orientations,
    //                             confidences,
    //                             clusters,
    //                             adjacency_list,
    //                             adjacency_list_all);
    std::thread t(&MotionDetection::computeGeometricClusters,
                  this,
                  std::cref(positions),
                  std::cref(orientations),
                  std::cref(confidences),
                  std::ref(clusters),
                  std::ref(adjacency_list),
                  std::ref(adjacency_list_all));

    std::vector<bool> inliers_states;
    cv::Size img_size(prevGray.cols, prevGray.rows);
    gms_matcher gms(prevKeypoints, img_size, keypoints, img_size, matches_all);
    //cv::Size img_size(prevGrayVec[0].cols, prevGrayVec[0].rows);
    //gms_matcher gms(prevKeypointsVec[0], img_size, keypoints, img_size, matches_all);
    int nb_inliers = gms.GetInlierMask(inliers_states, false, false);

    for(size_t i = 0; i < inliers_states.size(); i++)
    {
        if(inliers_states[i])
        {
            src_pts.push_back(prevKeypoints[matches_all[i].queryIdx].pt);
            //src_pts.push_back(prevKeypointsVec[0][matches_all[i].queryIdx].pt);
            dest_pts.push_back(keypoints[matches_all[i].trainIdx].pt);
        }
    }

    cv::Mat gray_estimate, depth_estimate;
    cv::Mat H;
    cv::Mat uvflow;
    cv::Mat optflow(gray.size[0], gray.size[1], CV_32FC2);

    if(src_pts.size() > 10)
    {
        H = cv::estimateAffinePartial2D(src_pts, dest_pts,
                                        cv::noArray(),
                                        /*cv::LMEDS*/cv::RANSAC,
                                        4.0/* reproj_thresh*/,
                                        2000/* max_iters*/,
                                        0.96/* conf*/,
                                        10/* refine_iters*/);
        H.convertTo(H, CV_32F);
        cv::warpAffine(prevGray, gray_estimate, H, prevGray.size());
        cv::warpAffine(prevDepth, depth_estimate, H, prevDepth.size());
        //cv::warpAffine(prevGrayVec[0], gray_estimate, H, prevGrayVec[0].size());
        //cv::warpAffine(prevDepthVec[0], depth_estimate, H, prevDepthVec[0].size());

        /***** Compute dense optical flow *****/
        dof->calc(gray, gray_estimate, optflow);
    }

    t.join();

    dim3 dim_block(32, 32);
    dim3 dim_grid((gray.cols + dim_block.x - 1) / dim_block.x, (gray.rows + dim_block.y - 1) / dim_block.y);

    if(src_pts.size() > 10)
    {
        cv::cuda::GpuMat uvflow_d(nb_superpixels, 1, CV_32FC2, cv::Vec2f(0.0f, 0.0f));
        flowMat.upload(optflow);

        initSuperpixelsUVFlow<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
                                                       texFlow->getTextureObject(),
                                                       reinterpret_cast<float2*>(uvflow_d.data),
                                                       flowMat.rows,
                                                       flowMat.cols,
                                                       40);
        cudaDeviceSynchronize();
        CudaCheckError();

        dim3 dim_block_list(128);
        dim3 dim_grid_list((nb_superpixels + dim_block.x - 1) / dim_block.x);


        computeSuperpixelsUVFlow<<<dim_grid_list, dim_block_list>>>(thrust::raw_pointer_cast(superpixels.data()),
                                                                    reinterpret_cast<float2*>(uvflow_d.data),
                                                                    nb_superpixels);
        cudaDeviceSynchronize();
        CudaCheckError();

        uvflow_d.download(uvflow);

        thrust::host_vector<SuperpixelRGBD> superpixels_h(superpixels);
        thrust::host_vector<int> labels(nb_superpixels, -1);

        int nb_clusters = clusters.size();

        std::vector<float> clusters_flow_mag(nb_clusters, 0.0f);
        std::vector<float> clusters_depth_diff(nb_clusters, 0.0f);
        std::vector<float> clusters_depth(nb_clusters, 0.0f);
        std::vector<float> counters(nb_clusters, 0.0f);

        //float flow_mag_max = 0.0f;

        for(int i = 0; i < nb_clusters; i++)
        {
            if(clusters[i].size() > 1)
            {
                for(size_t j = 0; j < clusters[i].size(); j++)
                {
                    labels[clusters[i][j]] = i;

                    //float flow_mag = cv::norm(cv::Mat(uvflow.at<cv::Vec2f>(clusters[i][j])));

                    //if(flow_mag > flow_mag_max)
                    //    flow_mag_max = flow_mag;

                    clusters_flow_mag[i] += cv::norm(cv::Mat(uvflow.at<cv::Vec2f>(clusters[i][j])));
                    //clusters_flow_mag[i] += flow_mag;
                    //counters[i] += superpixels_h[clusters[i][j]].size.x;
                    counters[i]++;


                    float z_diff = 0.0f;
                    float z_estimate = depth_estimate.at<float>(superpixels_h[clusters[i][j]].xy_rg.y, superpixels_h[clusters[i][j]].xy_rg.x);
                    float z = positions[clusters[i][j]].z;

                    if(z_estimate >= 0.2f && z_estimate < 4.0f && z >= 0.2f && z < 4.0f)
                    {
                        z_diff = std::abs(z_estimate - z);

                        clusters_depth[i] += z;
                    }

                    clusters_depth_diff[i] += z_diff;
                }
            }
        }

        for(size_t i = 0; i < labels.size(); i++)
        {
            if(labels[i] < 0)
            {
                if(adjacency_list_all[i].size() > 1)
                {
                    bool change = false;

                    for(size_t k = 0; k < adjacency_list_all[i].size() - 1; k++)
                    {
                        if(labels[adjacency_list_all[i][k]] != labels[adjacency_list_all[i][k+1]])
                        {
                            change = true;
                            break;
                        }
                    }

                    if(!change)
                        labels[i] = labels[adjacency_list_all[i][0]];
                }
            }

            if(labels[i] < 0)
            {
                float dist_min = 20.0f;
                int label_id_min = -1;


                for(size_t k = 0; k < adjacency_list_all[i].size(); k++)
                {
                    if(labels[adjacency_list_all[i][k]] >= 0)
                    {
                        float3 c_i = make_float3(superpixels_h[i].xy_rg.z,
                                                 superpixels_h[i].xy_rg.w,
                                                 superpixels_h[i].theta_b.w);
                        float3 c_k = make_float3(superpixels_h[adjacency_list_all[i][k]].xy_rg.z,
                                                 superpixels_h[adjacency_list_all[i][k]].xy_rg.w,
                                                 superpixels_h[adjacency_list_all[i][k]].theta_b.w);

                        //float dist = length(rgbToLab(colors[i]) - rgbToLab(colors[adjacency_list_all[i][k]]));
                        float dist = length(rgbToLab(c_i) - rgbToLab(c_k));

                        if(dist < dist_min)
                        {
                            dist_min = dist;
                            label_id_min = labels[adjacency_list_all[i][k]];
                        }
                    }
                }

                labels[i] = label_id_min;
            }

            if(labels[i] >= 0)
            {
                //float flow_mag = cv::norm(cv::Mat(uvflow.at<cv::Vec2f>(i)));

                //if(flow_mag > flow_mag_max)
                //    flow_mag_max = flow_mag;

                clusters_flow_mag[labels[i]] += cv::norm(cv::Mat(uvflow.at<cv::Vec2f>(i)));
                //clusters_flow_mag[labels[i]] += flow_mag;
                //counters[labels[i]] += superpixels_h[i].size.x;
                counters[labels[i]]++;

                float z_diff = 0.0f;
                float z_estimate = depth_estimate.at<float>(superpixels_h[i].xy_rg.y, superpixels_h[i].xy_rg.x);
                float z = positions[i].z;

                if(z_estimate >= 0.2f && z_estimate < 4.0f && z >= 0.2f && z < 4.0f)
                {
                    z_diff = std::abs(z_estimate - z);

                    clusters_depth[labels[i]] += z;
                }

                clusters_depth_diff[labels[i]] += z_diff;
            }
        }

        float flow_thresh = 2.5f + 0.5f * std::sqrt(H.at<float>(0, 2) * H.at<float>(0, 2)  + H.at<float>(1, 2) * H.at<float>(1, 2));

        for(int i = 0; i < nb_clusters; i++)
        {
            if(counters[i] < 1.0f)
                counters[i] = 1.0f;

            clusters_flow_mag[i] /= counters[i];
            clusters_depth_diff[i] /= counters[i];
            clusters_depth[i] /= counters[i];
        }

        for(int i = 0; i < nb_superpixels; i++)
        {
            if(labels[i] >= 0)
            {
                float depth_thresh = 12.0f * (0.0012f + 0.0019f * (clusters_depth[labels[i]] - 0.4f) * (clusters_depth[labels[i]] - 0.4f));

                if(clusters_flow_mag[labels[i]] > flow_thresh/* && clusters_flow_mag[labels[i]] > 0.5f * flow_mag_max*//* || clusters_depth_diff[labels[i]] > depth_thresh*/)
                {
                    isStatic[i] = false;
                    confidences[i] = -1.0f;
                }
            }
        }

        supersurfels_confidences = confidences;

        std::vector<cv::KeyPoint> static_keypoints;
        cv::Mat static_descriptors;
        cv::Mat index_mat_h;
        indexMat.download(index_mat_h);

        for(size_t i = 0; i < keypoints.size(); i++)
        {
            int superpixel_id = index_mat_h.at<int>(keypoints[i].pt.y, keypoints[i].pt.x);

            if(isStatic[superpixel_id])
            {
                static_keypoints.push_back(keypoints[i]);
                static_descriptors.push_back(descriptors.row(i).clone());
            }
        }

        keypoints = static_keypoints;
        descriptors = static_descriptors.clone();

        //        thrust::host_vector<int3> rng_colors;
        //        cv::RNG rgen(12345);

        //        for(int i = 0; i < nb_clusters; i++)
        //            rng_colors.push_back(make_int3(rgen.uniform(0,255), rgen.uniform(0,255), rgen.uniform(0,255)));

        //        thrust::device_vector<int> labels_d(labels);
        //        thrust::device_vector<int3> rng_colors_d(rng_colors);

        //        cv::cuda::GpuMat im_label_d(480, 640, CV_32FC3);

        //        computeLabelsImage<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
        //                                                    thrust::raw_pointer_cast(labels_d.data()),
        //                                                    reinterpret_cast<float3*>(im_label_d.data),
        //                                                    thrust::raw_pointer_cast(rng_colors_d.data()),
        //                                                    im_label_d.rows,
        //                                                    im_label_d.cols,
        //                                                    im_label_d.step/sizeof(float3));
        //        cudaDeviceSynchronize();
        //        CudaCheckError();

        //        cv::Mat im_label;
        //        im_label_d.download(im_label);
        //        im_label.convertTo(im_label, CV_8UC3);

        //        cv::imshow("labels", im_label);

        thrust::device_vector<bool> is_static_d(isStatic);

        cv::cuda::GpuMat mask_d(gray.rows, gray.cols, CV_8U);
        computeStaticDynamicImage<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
                                                           thrust::raw_pointer_cast(is_static_d.data()),
                                                           reinterpret_cast<uchar*>(mask_d.data),
                                                           mask_d.rows,
                                                           mask_d.cols,
                                                           mask_d.step/sizeof(uchar));
        cudaDeviceSynchronize();
        CudaCheckError();

        mask_d.download(mask);
        //cv::imshow("Static/Dynamic", mask);

        //        cv::Mat flow_image = flowToDisplay(optflow);
        //        cv::imshow("Computed flow", flow_image);
        //        cv::imshow("Gray estimate", gray_estimate);
        //        cv::imshow("Depth estimate", 0.3f*depth_estimate);
        //        cv::imshow("Gray", gray);
        //        cv::imshow("Previous gray", prevGray);
        //        //cv::imshow("Previous depth", 0.2f*prevDepth[0]);
        //        cv::Mat diff_depth;
        //        cv::subtract(depth, depth_estimate, diff_depth);
        //        cv::imshow("Depth difference", diff_depth);

        //        cv::waitKey(1);
    }

    prevGray = gray.clone();
    prevDepth = depth.clone();
    prevKeypoints = keypoints;
    prevDescriptors = descriptors.clone();

    //    if(prevKeypointsVec.size() < 1)
    //    {
    //        prevKeypointsVec.push_back(keypoints);
    //        prevDescriptorsVec.push_back(descriptors.clone());
    //        prevGrayVec.push_back(gray.clone());
    //        prevDepthVec.push_back(depth.clone());
    //    }
    //    else
    //    {
    //        std::rotate(prevKeypointsVec.begin(), prevKeypointsVec.begin() + 1, prevKeypointsVec.end());
    //        prevKeypointsVec[0] = keypoints;
    //        std::rotate(prevDescriptorsVec.begin(), prevDescriptorsVec.begin() + 1, prevDescriptorsVec.end());
    //        prevDescriptorsVec[0] = descriptors.clone();
    //        std::rotate(prevGrayVec.begin(), prevGrayVec.begin() + 1, prevGrayVec.end());
    //        prevGrayVec[0] = gray.clone();
    //        std::rotate(prevDepthVec.begin(), prevDepthVec.begin() + 1, prevDepthVec.end());
    //        prevDepthVec[0] = depth.clone();
    //    }
}

void MotionDetection::detectMotionCombined(const cv::Mat& rgb,
                                           const cv::Mat& gray,
                                           const cv::Mat& depth,
                                           std::vector<cv::KeyPoint>& keypoints,
                                           cv::Mat& descriptors,
                                           const thrust::device_vector<float3>& supersurfels_positions,
                                           const thrust::device_vector<Mat33>& supersurfels_orientations,
                                           thrust::device_vector<float>& supersurfels_confidences,
                                           const thrust::device_vector<SuperpixelRGBD>& superpixels,
                                           const cv::cuda::GpuMat& index_mat)
{
    int nb_superpixels = superpixels.size();

    isStatic.clear();
    isStatic.resize(nb_superpixels, true);

    if(prevGray.empty())
    {
        prevGray = gray.clone();
        prevDepth = depth.clone();
        prevKeypoints = keypoints;
        prevDescriptors = descriptors.clone();

        indexMat.create(gray.size(), CV_32SC1);
        flowMat.create(gray.size(), CV_32FC2);
        depthEstimateMat.create(gray.size(), CV_32FC1);

        texIndex = new Texture<int>(indexMat);
        texFlow = new Texture<float2>(flowMat);
        texDepthEstimate = new Texture<float>(depthEstimateMat);

        mask = cv::Mat::zeros(gray.size(), CV_8UC1);

        return;
    }

    index_mat.copyTo(indexMat);

    thrust::host_vector<float3> positions(supersurfels_positions);
    thrust::host_vector<Mat33> orientations(supersurfels_orientations);
    thrust::host_vector<float> confidences(supersurfels_confidences);

    std::vector<std::vector<int>> adjacency_list(nb_superpixels);
    std::vector<std::vector<int>> adjacency_list_all(nb_superpixels);

    std::vector<std::vector<int>> clusters;

    cv::cuda::GpuMat adjacency_d(nb_superpixels, nb_superpixels, CV_8UC1, cv::Scalar(0));

    dim3 dim_block(32, 32);
    dim3 dim_grid((flowMat.cols + dim_block.x - 1) / dim_block.x, (indexMat.rows + dim_block.y - 1) / dim_block.y);

    computeSuperpixelsAdjacency<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
                                                         adjacency_d,
                                                         indexMat.cols,
                                                         indexMat.rows);
    cudaDeviceSynchronize();
    CudaCheckError();

    cv::Mat adjacency;
    adjacency_d.download(adjacency);

    /*** Object detection ***/
    std::thread t(&MotionDetection::detectYolo,
                  this,
                  std::cref(rgb));
    //detectYolo(rgb);

    for(int i = 0; i < nb_superpixels; i++)
    {
        for(int j = 0; j < nb_superpixels; j++)
        {
            if(adjacency.at<uchar>(i, j))
            {
                if(confidences[i] > 0.0f && confidences[j] > 0.0f)
                {
                    float3 n_i = normalize(orientations[i].rows[2]);
                    float3 n_j = normalize(orientations[j].rows[2]);
                    float3 c_ij =  (positions[i] - positions[j]) / length(positions[i] - positions[j]);
                    float geometric_dist = (length(cross(n_i, n_j)) + fabsf(dot(n_i, c_ij)) + fabsf(dot(n_j, c_ij))) / 3.0f;

                    if(geometric_dist < 0.2f)
                        adjacency_list[i].push_back(j);
                }

                adjacency_list_all[i].push_back(j);
            }
        }
    }

    AdjListDFSInitialize(adjacency_list, 0, clusters);

    t.join();

    cv::Mat index_mat_h;
    indexMat.download(index_mat_h);
    thrust::host_vector<SuperpixelRGBD> superpixels_h(superpixels);
    std::vector<bool> visited(nb_superpixels, false);

    // Flood filling
    for(auto &obj : objects)
    {
        if(obj.obj_id == 0/* && obj.prob > 0.3f*/) // 0 = human class id
        {
            std::stack<std::vector<int>> ff_stack;

            int i = std::min(int(obj.x + obj.w / 2), index_mat_h.cols - 1);
            int j = std::min(int(obj.y + obj.h / 2), index_mat_h.rows - 1);
            int seed = index_mat_h.at<int>(j, i);
            
            for(size_t n = 0; n < adjacency_list_all[seed].size(); n++)
            {
                if((positions[seed].z > positions[adjacency_list_all[seed][n]].z || confidences[seed] < 0.0f) &&
                   superpixels_h[adjacency_list_all[seed][n]].xy_rg.x >= obj.x &&
                   superpixels_h[adjacency_list_all[seed][n]].xy_rg.x < obj.x + obj.w &&
                   superpixels_h[adjacency_list_all[seed][n]].xy_rg.y >= obj.y &&
                   superpixels_h[adjacency_list_all[seed][n]].xy_rg.y < obj.y + obj.h)
                      seed = adjacency_list_all[seed][n];
            }

//            float3 seed_color = make_float3(superpixels_h[seed].xy_rg.z,
//                                            superpixels_h[seed].xy_rg.w,
//                                            superpixels_h[seed].theta_b.w);
            isStatic[seed] = false;
            visited[seed] = true;
            
            float z_seed = positions[seed].z;

            ff_stack.push(adjacency_list_all[seed]);

            while(!ff_stack.empty())
            {
                std::vector<int> neighbours = ff_stack.top();
                ff_stack.pop();

                for(size_t k = 0; k < neighbours.size(); k++)
                {
                    if(!visited[neighbours[k]] &&
                       superpixels_h[neighbours[k]].xy_rg.x >= obj.x &&
                       superpixels_h[neighbours[k]].xy_rg.x < obj.x + obj.w &&
                       superpixels_h[neighbours[k]].xy_rg.y >= obj.y &&
                       superpixels_h[neighbours[k]].xy_rg.y < obj.y + obj.h)
                    {
//                        float3 neighbour_color = make_float3(superpixels_h[neighbours[k]].xy_rg.z,
//                                                             superpixels_h[neighbours[k]].xy_rg.w,
//                                                             superpixels_h[neighbours[k]].theta_b.w);
                        //float lab_dist = length(rgbToLab(seed_color) - rgbToLab(neighbour_color));

                        float depth_dist = 1000.0f;
                        if(confidences[neighbours[k]])
                            depth_dist = std::abs(z_seed - positions[neighbours[k]].z);

                        if(/*lab_dist < 3.0f || */depth_dist < 0.3f)
                        {
                            seed = neighbours[k];
                            //seed_color = neighbour_color;

                            isStatic[seed] = false;

                            ff_stack.push(adjacency_list_all[seed]);
                        }

                        visited[neighbours[k]] = true;
                    }
                }
            }
        }
    }

    // Filter potentially dynamic keypoints and estimate 2D homography
    std::vector<cv::KeyPoint> filtered_keypoints;
    cv::Mat filtered_descriptors;

    for(size_t i = 0; i < keypoints.size(); i++)
    {
        int superpixel_id = index_mat_h.at<int>(keypoints[i].pt.y, keypoints[i].pt.x);

        if(isStatic[superpixel_id])
        {
            filtered_keypoints.push_back(keypoints[i]);
            filtered_descriptors.push_back(descriptors.row(i).clone());
        }
    }

    keypoints = filtered_keypoints;
    descriptors = filtered_descriptors.clone();

    for(int i = 0; i < nb_superpixels; i++)
    {
        if(!isStatic[i])
            confidences[i] = -1.0f;
    }

//    supersurfels_confidences = confidences;

    std::vector<cv::Point2f> src_pts, dest_pts;
    std::vector<cv::DMatch> matches_all;
    cv::cuda::GpuMat prev_descriptors_d(prevDescriptors), descriptors_d(descriptors);
    matcher->match(prev_descriptors_d, descriptors_d, matches_all);

    std::vector<bool> inliers_states;
    cv::Size img_size(prevGray.cols, prevGray.rows);
    gms_matcher gms(prevKeypoints, img_size, keypoints, img_size, matches_all);
    int nb_inliers = gms.GetInlierMask(inliers_states, false, false);

    for(size_t i = 0; i < inliers_states.size(); i++)
    {
        if(inliers_states[i])
        {
            src_pts.push_back(prevKeypoints[matches_all[i].queryIdx].pt);
            dest_pts.push_back(keypoints[matches_all[i].trainIdx].pt);
        }
    }

    if(src_pts.size() > 10)
    {
        cv::Mat gray_estimate, depth_estimate;
        cv::Mat H;
        cv::Mat uvflow;
        cv::Mat optflow(gray.size[0], gray.size[1], CV_32FC2);

        H = cv::estimateAffinePartial2D(src_pts, dest_pts,
                                        cv::noArray(),
                                        /*cv::LMEDS*/cv::RANSAC,
                                        4.0/* reproj_thresh*/,
                                        2000/* max_iters*/,
                                        0.96/* conf*/,
                                        10/* refine_iters*/);
        H.convertTo(H, CV_32F);
        cv::warpAffine(prevGray, gray_estimate, H, prevGray.size());
        cv::warpAffine(prevDepth, depth_estimate, H, prevDepth.size());

        /***** Compute dense optical flow *****/
        dof->calc(gray, gray_estimate, optflow);

        cv::cuda::GpuMat uvflow_d(nb_superpixels, 1, CV_32FC2, cv::Vec2f(0.0f, 0.0f));
        flowMat.upload(optflow);

        initSuperpixelsUVFlow<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
                                                       texFlow->getTextureObject(),
                                                       reinterpret_cast<float2*>(uvflow_d.data),
                                                       flowMat.rows,
                                                       flowMat.cols,
                                                       40);
        cudaDeviceSynchronize();
        CudaCheckError();

        dim3 dim_block_list(128);
        dim3 dim_grid_list((nb_superpixels + dim_block.x - 1) / dim_block.x);


        computeSuperpixelsUVFlow<<<dim_grid_list, dim_block_list>>>(thrust::raw_pointer_cast(superpixels.data()),
                                                                    reinterpret_cast<float2*>(uvflow_d.data),
                                                                    nb_superpixels);
        cudaDeviceSynchronize();
        CudaCheckError();

        uvflow_d.download(uvflow);

        thrust::host_vector<int> labels(nb_superpixels, -1);

        int nb_clusters = clusters.size();

        std::vector<float> clusters_flow_mag(nb_clusters, 0.0f);
        std::vector<float> clusters_depth_diff(nb_clusters, 0.0f);
        std::vector<float> clusters_depth(nb_clusters, 0.0f);
        std::vector<float> counters(nb_clusters, 0.0f);

        for(int i = 0; i < nb_clusters; i++)
        {
            if(clusters[i].size() > 2)
            {
                for(size_t j = 0; j < clusters[i].size(); j++)
                {
                    if(isStatic[clusters[i][j]])
                        labels[clusters[i][j]] = i;
                    else
                        continue;

                    clusters_flow_mag[i] += cv::norm(cv::Mat(uvflow.at<cv::Vec2f>(clusters[i][j])));
                    counters[i]++;

                    float z_diff = 0.0f;
                    float z_estimate = depth_estimate.at<float>(superpixels_h[clusters[i][j]].xy_rg.y, superpixels_h[clusters[i][j]].xy_rg.x);
                    float z = positions[clusters[i][j]].z;

                    if(z_estimate >= 0.2f && z_estimate < 3.0f && z >= 0.2f && z < 3.0f)
                    {
                        z_diff = std::abs(z_estimate - z);

                        clusters_depth[i] += z;
                    }

                    clusters_depth_diff[i] += z_diff;
                }
            }
        }

        for(size_t i = 0; i < labels.size(); i++)
        {
            if(!isStatic[i])
                continue;

            if(labels[i] < 0)
            {
                if(adjacency_list_all[i].size() > 1)
                {
                    bool change = false;

                    for(size_t k = 0; k < adjacency_list_all[i].size() - 1; k++)
                    {
                        if(labels[adjacency_list_all[i][k]] != labels[adjacency_list_all[i][k+1]])
                        {
                            change = true;
                            break;
                        }
                    }

                    if(!change)
                        labels[i] = labels[adjacency_list_all[i][0]];
                }
            }

            if(labels[i] < 0)
            {
                float dist_min = 20.0f;
                int label_id_min = -1;


                for(size_t k = 0; k < adjacency_list_all[i].size(); k++)
                {
                    if(labels[adjacency_list_all[i][k]] >= 0)
                    {
                        float3 c_i = make_float3(superpixels_h[i].xy_rg.z,
                                                 superpixels_h[i].xy_rg.w,
                                                 superpixels_h[i].theta_b.w);
                        float3 c_k = make_float3(superpixels_h[adjacency_list_all[i][k]].xy_rg.z,
                                                 superpixels_h[adjacency_list_all[i][k]].xy_rg.w,
                                                 superpixels_h[adjacency_list_all[i][k]].theta_b.w);
                        float dist = length(rgbToLab(c_i) - rgbToLab(c_k));

                        if(dist < dist_min)
                        {
                            dist_min = dist;
                            label_id_min = labels[adjacency_list_all[i][k]];
                        }
                    }
                }

                labels[i] = label_id_min;
            }

            if(labels[i] >= 0)
            {
                clusters_flow_mag[labels[i]] += cv::norm(cv::Mat(uvflow.at<cv::Vec2f>(i)));
                counters[labels[i]]++;

                float z_diff = 0.0f;
                float z_estimate = depth_estimate.at<float>(superpixels_h[i].xy_rg.y, superpixels_h[i].xy_rg.x);
                float z = positions[i].z;

                if(z_estimate >= 0.2f && z_estimate < 3.0f && z >= 0.2f && z < 3.0f)
                {
                    z_diff = std::abs(z_estimate - z);

                    clusters_depth[labels[i]] += z;
                }

                clusters_depth_diff[labels[i]] += z_diff;
            }
        }

        float flow_thresh = 4.0f + 0.5f * std::sqrt(H.at<float>(0, 2) * H.at<float>(0, 2)  + H.at<float>(1, 2) * H.at<float>(1, 2));

        for(int i = 0; i < nb_clusters; i++)
        {
            if(counters[i] < 1.0f)
                counters[i] = 1.0f;


            clusters_flow_mag[i] /= counters[i];
            clusters_depth_diff[i] /= counters[i];
            clusters_depth[i] /= counters[i];
        }

        for(int i = 0; i < nb_superpixels; i++)
        {
            if(labels[i] >= 0 && isStatic[i])
            {
                float depth_thresh = 20.0f * (0.0012f + 0.0019f * (clusters_depth[labels[i]] - 0.4f) * (clusters_depth[labels[i]] - 0.4f));

                if(clusters_flow_mag[labels[i]] > flow_thresh/* || clusters_depth_diff[labels[i]] > depth_thresh*/)
                {
                    isStatic[i] = false;
                    confidences[i] = -1.0f;
                }
            }
        }

        //supersurfels_confidences = confidences;

        filtered_keypoints.clear();
        filtered_descriptors.release();

        for(size_t i = 0; i < keypoints.size(); i++)
        {
            int superpixel_id = index_mat_h.at<int>(keypoints[i].pt.y, keypoints[i].pt.x);

            if(isStatic[superpixel_id])
            {
                filtered_keypoints.push_back(keypoints[i]);
                filtered_descriptors.push_back(descriptors.row(i).clone());
            }
        }

        keypoints = filtered_keypoints;
        descriptors = filtered_descriptors.clone();
    }

    supersurfels_confidences = confidences;

    thrust::device_vector<bool> is_static_d(isStatic);

    cv::cuda::GpuMat mask_d(gray.rows, gray.cols, CV_8U);
    computeStaticDynamicImage<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
                                                       thrust::raw_pointer_cast(is_static_d.data()),
                                                       reinterpret_cast<uchar*>(mask_d.data),
                                                       mask_d.rows,
                                                       mask_d.cols,
                                                       mask_d.step/sizeof(uchar));
    cudaDeviceSynchronize();
    CudaCheckError();

    mask_d.download(mask);
    //cv::imshow("Static/Dynamic", mask);
    //cv::waitKey(1);

    prevGray = gray.clone();
    prevDepth = depth.clone();
    prevKeypoints = keypoints;
    prevDescriptors = descriptors.clone();
}

void MotionDetection::detectMotionYoloOnly(const cv::Mat& rgb,
                                           std::vector<cv::KeyPoint>& keypoints,
                                           cv::Mat& descriptors,
                                           const thrust::device_vector<float3>& supersurfels_positions,
                                           thrust::device_vector<float>& supersurfels_confidences,
                                           const thrust::device_vector<SuperpixelRGBD>& superpixels,
                                           const cv::cuda::GpuMat& index_mat)
{
    int nb_superpixels = int(superpixels.size());

    isStatic.clear();
    isStatic.resize(nb_superpixels, true);

    if(indexMat.empty())
    {
        indexMat.create(index_mat.size(), CV_32SC1);
        texIndex = new Texture<int>(indexMat);
        mask = cv::Mat::zeros(index_mat.size(), CV_8UC1);

        return;
    }

    index_mat.copyTo(indexMat);

    /*** Object detection ***/
    detectYolo(rgb);

    /*** Compute superpixels adjacency list ***/
    cv::cuda::GpuMat adjacency_d(nb_superpixels, nb_superpixels, CV_8UC1, cv::Scalar(0));

    dim3 dim_block(32, 32);
    dim3 dim_grid((indexMat.cols + dim_block.x - 1) / dim_block.x, (indexMat.rows + dim_block.y - 1) / dim_block.y);

    computeSuperpixelsAdjacency<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
                                                         adjacency_d,
                                                         indexMat.cols,
                                                         indexMat.rows);
    cudaDeviceSynchronize();
    CudaCheckError();

    std::vector<std::vector<int>> adjacency_list(nb_superpixels);
    cv::Mat adjacency;
    adjacency_d.download(adjacency);

//    #pragma omp parallel for
    for(int i = 0; i < nb_superpixels; i++)
    {
        for(int j = 0; j < nb_superpixels; j++)
        {
            if(adjacency.at<uchar>(i, j))
                adjacency_list[i].push_back(j);

            if(adjacency_list[i].size() > 3)
                break;
        }
    }

    /*** Extract superpixels belonging to potential moving objects ***/
    cv::Mat index_mat_h;
    indexMat.download(index_mat_h);

    thrust::host_vector<SuperpixelRGBD> superpixels_h(superpixels);
    thrust::host_vector<float3> positions(supersurfels_positions);
    thrust::host_vector<float> confidences(supersurfels_confidences);

    std::vector<bool> visited(nb_superpixels, false);

    // Flood filling
    for(auto &obj : objects)
    {
        if(obj.obj_id == 0/* && obj.prob > 0.3f*/) // 0 = human class id
        {
            std::stack<std::vector<int>> ff_stack;

            int i = std::min(int(obj.x + obj.w / 2), index_mat_h.cols - 1);
            int j = std::min(int(obj.y + obj.h / 2), index_mat_h.rows - 1);
            int seed = index_mat_h.at<int>(j, i);

            for(size_t n = 0; n < adjacency_list_all[seed].size(); n++)
            {
                if((positions[seed].z > positions[adjacency_list_all[seed][n]].z || confidences[seed] < 0.0f) &&
                   superpixels_h[adjacency_list_all[seed][n]].xy_rg.x >= obj.x &&
                   superpixels_h[adjacency_list_all[seed][n]].xy_rg.x < obj.x + obj.w &&
                   superpixels_h[adjacency_list_all[seed][n]].xy_rg.y >= obj.y &&
                   superpixels_h[adjacency_list_all[seed][n]].xy_rg.y < obj.y + obj.h)
                      seed = adjacency_list_all[seed][n];
            }
            
            float3 seed_color = make_float3(superpixels_h[seed].xy_rg.z,
                                            superpixels_h[seed].xy_rg.w,
                                            superpixels_h[seed].theta_b.w);
            float z_seed = positions[seed].z;
            
            isStatic[seed] = false;
            visited[seed] = true;

            ff_stack.push(adjacency_list[seed]);

            while(!ff_stack.empty())
            {
                std::vector<int> neighbours = ff_stack.top();
                ff_stack.pop();

                for(size_t k = 0; k < neighbours.size(); k++)
                {
                    if(!visited[neighbours[k]] &&
                       superpixels_h[neighbours[k]].xy_rg.x >= obj.x &&
                       superpixels_h[neighbours[k]].xy_rg.x < obj.x + obj.w &&
                       superpixels_h[neighbours[k]].xy_rg.y >= obj.y &&
                       superpixels_h[neighbours[k]].xy_rg.y < obj.y + obj.h)
                    {
                        float3 neighbour_color = make_float3(superpixels_h[neighbours[k]].xy_rg.z,
                                                             superpixels_h[neighbours[k]].xy_rg.w,
                                                             superpixels_h[neighbours[k]].theta_b.w);
                        float lab_dist = length(rgbToLab(seed_color) - rgbToLab(neighbour_color));

                        float depth_dist = 1000.0f;
                        if(confidences[neighbours[k]])
                            depth_dist = std::abs(z_seed - positions[neighbours[k]].z);

                        if(/*lab_dist < 3.0f || */depth_dist < 0.3f)
                        {
                            seed = neighbours[k];
                            seed_color = neighbour_color;

                            isStatic[seed] = false;

                            ff_stack.push(adjacency_list[seed]);
                        }

                        visited[neighbours[k]] = true;
                    }
                }
            }
        }
    }

    // Filter potentially dynamic keypoints and estimate 2D homography
    std::vector<cv::KeyPoint> filtered_keypoints;
    cv::Mat filtered_descriptors;

    for(size_t i = 0; i < keypoints.size(); i++)
    {
        int superpixel_id = index_mat_h.at<int>(keypoints[i].pt.y, keypoints[i].pt.x);

        if(isStatic[superpixel_id])
        {
            filtered_keypoints.push_back(keypoints[i]);
            filtered_descriptors.push_back(descriptors.row(i).clone());
        }
    }

    keypoints = filtered_keypoints;
    descriptors = filtered_descriptors.clone();

    for(int i = 0; i < nb_superpixels; i++)
    {
        if(!isStatic[i])
            confidences[i] = -1.0f;
    }

    supersurfels_confidences = confidences;

    thrust::device_vector<bool> is_static_d(isStatic);
    cv::cuda::GpuMat mask_d(index_mat.rows, index_mat.cols, CV_8U);

    computeStaticDynamicImage<<<dim_grid, dim_block>>>(texIndex->getTextureObject(),
                                                       thrust::raw_pointer_cast(is_static_d.data()),
                                                       reinterpret_cast<uchar*>(mask_d.data),
                                                       mask_d.rows,
                                                       mask_d.cols,
                                                       mask_d.step/sizeof(uchar));
    cudaDeviceSynchronize();
    CudaCheckError();

    mask_d.download(mask);
    //cv::imshow("Mask", mask);
    //cv::waitKey(1);
}

} // supersurfel_fusion
