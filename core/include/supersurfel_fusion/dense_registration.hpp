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


#ifndef DENSE_REGISTRATION_HPP
#define DENSE_REGISTRATION_HPP

#include <opencv2/core.hpp>
#include <supersurfel_fusion/dense_registration_types.hpp>
#include <supersurfel_fusion/cam_param.hpp>
#include <thrust/device_vector.h>
#include <supersurfel_fusion/matrix_types.h>
#include <supersurfel_fusion/texture.hpp>


namespace supersurfel_fusion
{

class DenseRegistration
{

public:
    DenseRegistration();
    DenseRegistration(int max_iter, double cov_thresh);
    ~DenseRegistration();
    bool align(const thrust::device_vector<float3>& source_positions,
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
               const CamParam& cam_param,
               Mat33& R,
               float3& t);
    inline int& setNbIter(){return nbIter;}
    inline const int&  getNbIter() const {return nbIter;}
    inline double& setCovThresh(){return covThresh;}
    inline const double&  getCovThresh() const {return covThresh;}

private:
    int nbIter;
    double covThresh;
    MotionTrackingData* mtd;

}; // class DenseRegistration

} // supersurfel_fusion

#endif // DENSE_REGISTRATION_HPP
