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


#include <supersurfel_fusion/pnp_solver.hpp>
#include <Eigen/StdVector>
#include <numeric>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/pcg/linear_solver_pcg.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#define N_PASSES 2
#define REPROJECTION_TH2 5.0


namespace supersurfel_fusion
{

PnPSolver::PnPSolver()
    : optimizer(nullptr)
{
    optimizer = new g2o::SparseOptimizer();
    optimizer->setVerbose(false);
    std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver = g2o::make_unique<g2o::LinearSolverPCG<g2o::BlockSolver_6_3::PoseMatrixType>>();
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(g2o::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver)));
    optimizer->setAlgorithm(solver);
}

PnPSolver::~PnPSolver()
{
    delete optimizer;
}

bool PnPSolver::computePose(Eigen::Isometry3f& cam_pose,
                            const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& points3D,
                            const std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>& points2D,
                            float fx,
                            float fy,
                            float cx,
                            float cy)
{
    bool valid = false;

    // add camera
    Eigen::Quaterniond orientation(cam_pose.linear().cast<double>());
    Eigen::Vector3d position = cam_pose.translation().cast<double>();
    g2o::SBACam sba_cam(orientation, position);
    sba_cam.setKcam(double(fx), double(fy), double(cx), double(cy), 0.0);
    g2o::VertexCam *cam_vertex = new g2o::VertexCam();
    cam_vertex->setId(0);
    cam_vertex->setEstimate(sba_cam);
    cam_vertex->setFixed(false);
    optimizer->addVertex(cam_vertex);

    // add mono measurments
    int nb_points = points3D.size();
    double delta = std::sqrt(REPROJECTION_TH2);
    int vertex_id = 1;
    std::vector<g2o::EdgeProjectP2MC *> edges;
    edges.reserve(nb_points);

    for(int i = 0; i < nb_points; i++)
    {
        g2o::VertexSBAPointXYZ *point_vertex = new g2o::VertexSBAPointXYZ();
        point_vertex->setId(vertex_id++);
        point_vertex->setMarginalized(false);
        point_vertex->setEstimate(points3D[i].cast<double>());
        point_vertex->setFixed(true);
        optimizer->addVertex(point_vertex);

        g2o::EdgeProjectP2MC *edge = new g2o::EdgeProjectP2MC();
        edge->setVertex(0, point_vertex);
        edge->setVertex(1, cam_vertex);
        edge->setMeasurement(points2D[i].cast<double>());
        edge->information() = Eigen::Matrix2d::Identity();
        g2o::RobustKernel *rkh = new g2o::RobustKernelCauchy;
        edge->setRobustKernel(rkh);
        rkh->setDelta(delta);
        optimizer->addEdge(edge);
        edges.push_back(edge);
    }

    // perform optimzation
    inlierMarks.clear();
    inlierMarks.resize(nb_points, 1);

    for(int i = 0; i < N_PASSES; i++)
    {
        optimizer->initializeOptimization(0);
        optimizer->optimize(5);

        for(int k = 0; k < nb_points; k++)
        {
            if(edges[k]->chi2() > REPROJECTION_TH2)
            {
                edges[k]->setLevel(1);
                inlierMarks[k] = 0;
            }
        }
    }

    nbInliers = std::accumulate(inlierMarks.begin(), inlierMarks.end(), 0);

    // retrieve optimized pose
    Eigen::Vector3f op_position = cam_vertex->estimate().translation().cast<float>();
    Eigen::Quaternionf op_orientation = cam_vertex->estimate().rotation().cast<float>();

    if(float(nbInliers) >= 0.2f * float(points2D.size())  && (op_position - cam_pose.translation()).norm() < 0.5f)
    {
        Eigen::Isometry3f op_pose;
        op_pose.translation() = op_position;
        op_pose.linear() = op_orientation.toRotationMatrix();

        cam_pose = op_pose;

        valid = true;
    }

    optimizer->clear();

    return valid;
}

} // supersurfel_fusion
