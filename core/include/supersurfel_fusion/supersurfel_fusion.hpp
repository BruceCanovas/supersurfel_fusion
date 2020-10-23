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


#ifndef SUPERSURFEL_FUSION_HPP
#define SUPERSURFEL_FUSION_HPP

#include <opencv2/opencv.hpp>
#include <supersurfel_fusion/TPS_RGBD.hpp>
#include <supersurfel_fusion/cached_allocator.hpp>
#include <supersurfel_fusion/supersurfels.hpp>
#include <supersurfel_fusion/dense_registration.hpp>
#include <supersurfel_fusion/sparse_vo.hpp>
#include <supersurfel_fusion/ferns.hpp>
#include <Eigen/Geometry>
#include <supersurfel_fusion/deformation_graph.hpp>
#include <supersurfel_fusion/motion_detection.hpp>


namespace supersurfel_fusion
{

class SupersurfelFusion
{

public:
    SupersurfelFusion();
    ~SupersurfelFusion();
    void initialize(const CamParam& cam_param,
                    int cell_size = 16,
                    float lambda_pos = 50.0f,
                    float lambda_bound = 1000.0f,
                    float lambda_size = 10000.0f,
                    float lambda_disp = 1000000.0f,
                    float thresh_disp = 0.0001f,
                    int seg_iter = 10,
                    bool seg_use_ransac = true,
                    int nb_samples = 16,
                    int filter_iter = 4,
                    float filter_alpha = 0.1f,
                    float filter_beta = 1.0f,
                    float filter_threshold = 0.05f,
                    float range_min = 0.2f,
                    float range_max = 5.0f,
                    int delta_t = 20,
                    float conf_thresh = 2500.0f,
                    int nb_supersurfels_max = 50000,
                    int icp_iter = 10,
                    double icp_cov_thresh = 0.04,
                    int nb_features = 2000,
                    float features_scale_factor = 1.2f,
                    int features_nb_levels = 8,
                    int ini_th_fast = 20,
                    int min_th_fast = 7,
                    int untracked_threshold = 10,
                    bool enable_loop_closure = true,
                    bool enable_mod = true);
    void processFrame(const cv::Mat& rgb_h,
                      const cv::Mat& depth_h);
    void generateSupersurfels();
    void exportModel(std::string filename);
    void computeSuperpixelSegIm(cv::Mat& seg_im);
    void computeSlantedPlaneIm(cv::Mat& slanted_plane_im);
    void closeLoop(const std::vector<cv::KeyPoint>& curr_kpts,
                   const cv::Mat& curr_desc);

    inline bool isInitialized() {return initialized;}
    inline const Supersurfels& getFrame() {return frame;}
    inline const Supersurfels& getModel() {return model;}
    inline int getnbSupersurfels() const {return nbSupersurfels;}
    inline int getStamp() const {return stamp;}
    inline const Transform3& getPose() const {return pose;}
    Eigen::Isometry3f getPoseEigen() const;
    inline thrust::host_vector<float3> const& getNodesPositions() const { return def.getNodesPositions(); }
    inline thrust::host_vector<int4> const& getNodesNeighbours() const { return def.getNodesNeighbours(); }
    inline thrust::host_vector<float3> const& getConstraintsSources() const { return def.getConstraintsSources(); }
    inline thrust::host_vector<float3> const& getConstraintsTargets() const { return def.getConstraintsTargets(); }
    inline bool const& getLCdone() const { return LCdone; }
    inline const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& getLocalMapPoints() const {return vo->getLocalMapPoints();}
    Transform3 eigenToTransform3(const Eigen::Isometry3f& iso) const;
    void extractLocalPointCloud(thrust::host_vector<float3>& positions,
                                thrust::host_vector<float3>& normals);
    inline cv::Mat const& getModMask() const {return mod.getMask();}
    void setYoloDetector(const std::string& yolo_config_file,
                         const std::string& yolo_weights_file,
                         const std::string& yolo_names_file);

private:
    cv::cuda::GpuMat depth, rgb;

    TPS_RGBD *tps;
    int nbSuperpixels;

    dim3 dimBlockIm, dimGridIm, dimBlockList, dimGridList;

    Supersurfels model, frame;
    CachedAllocator allocator;
    int nbSupersurfels, nbSupersurfelsMax, nbRemoved, nbActive;
    int *nbSupersurfelsDev, *nbRemovedDev;

    cv::cuda::GpuMat filteredDepth;
    cv::Ptr<Texture<float>> texDepth;

    SparseVO *vo;
    DenseRegistration *icp;
    Transform3 pose;

    Ferns* ferns;
    bool enableLoopClosure, LCdone;
    int previousFernId, stampLastLC;
    DeformationGraph def;

    MotionDetection mod;
    bool enableMod;

    int stamp;
    float rangeMin, rangeMax;
    bool initialized;
    int deltaT;
    float confThresh;
    CamParam cam;

    double runtime, modelSize;

}; // class SupersurfelFusion

} // namespace supersurfel_fusion

#endif // SUPERSURFEL_FUSION_HPP
