/*********************************************************************
 * BSD 3-Clause License
 *
 * Copyright (c) 2018, Rawashdeh Research Group
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **********************************************************************/


#ifndef PNP_SOLVER_HPP
#define PNP_SOLVER_HPP

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/StdVector>


namespace g2o
{

class SparseOptimizer;

}

namespace supersurfel_fusion
{

class PnPSolver
{

  public:
    PnPSolver();
    ~PnPSolver();

    bool computePose(Eigen::Isometry3f& cam_pose,
                     const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points3D,
                     const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& points2D,
                     float fx,
                     float fy,
                     float cx,
                     float cy);
    bool computePoseWithPrior(Eigen::Isometry3f& cam_pose,
                              Eigen::Isometry3f const& cam_pose_prior,
                              const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points3D,
                              const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& points2D,
                              float fx,
                              float fy,
                              float cx,
                              float cy);

    inline const std::vector<int>& getInlierMarks() const {return inlierMarks;}
    //inline const double getRMSE() const {return rmse;}

  private:
    g2o::SparseOptimizer *optimizer;

    std::vector<int> inlierMarks;
    //double rmse;

}; // class PnPSolver

} // supersurfel_fusion

#endif // PNP_SOLVER_HPP
