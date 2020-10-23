/*
 * This file is part of ElasticFusion.
 *
 * Copyright (C) 2015 Imperial College London
 *
 * The use of the code within this file and all code within files that
 * make up the software that is ElasticFusion is permitted for
 * non-commercial purposes only.  The full terms and conditions that
 * apply to the code within this file are detailed within the LICENSE.txt
 * file and at <http://www.imperial.ac.uk/dyson-robotics-lab/downloads/elastic-fusion/elastic-fusion-license/>
 * unless explicitly stated.  By downloading this file you agree to
 * comply with these terms.
 *
 * If you wish to use any of this code for commercial purposes then
 * please email researchcontracts.engineering@imperial.ac.uk.
 *
 */


#ifndef DEFORMATION_GRAPH_HPP
#define DEFORMATION_GRAPH_HPP

#include <supersurfel_fusion/deformation_graph_types.hpp>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "Jacobian.h"
#include <supersurfel_fusion/ferns_types.hpp>
#include <thrust/device_vector.h>


class CholeskyDecomp;

namespace supersurfel_fusion
{

class DeformationGraph
{

public:
    DeformationGraph();
    ~DeformationGraph();
    void buildGraphFromModel(const thrust::host_vector<float3>& model_positions,
                             const thrust::host_vector<int2>& model_stamps,
                             int sample_rate,
                             const Constraints& cons,
                             const std::vector<StampedPose>& pose_graph);
    void setConstraints();
    void weightVerticesSeq();
    void weightPosesSeq(const std::vector<StampedPose>& stamped_poses);
    inline const thrust::host_vector<float3>& getNodesPositions() const { return graphNodes.positions; }
    inline const thrust::host_vector<int4>& getNodesNeighbours() const { return graphNodes.neighbours; }
    inline const thrust::host_vector<int4>& getVerticesNodes() const { return vertexMap.nodes; }
    inline const thrust::host_vector<float3>& getConstraintsSources() const { return constraints.sources; }
    inline const thrust::host_vector<float3>& getConstraintsTargets() const { return constraints.targets; }
    void clear();
    void computeVertexPosition(int vertex_id, float3& position);
    float constraintError();
    void applyDeltaSparse(const Eigen::VectorXd& delta);
    bool optimiseGraphSparse(float& error, float& mean_cons_error/*, bool global*/);
    Eigen::VectorXd sparseResidual(int nb_rows);
    void sparseJacobian(Jacobian& jacobian, int nb_rows, int nb_cols, int back_set);
    bool constrain(thrust::device_vector<float3>& model_positions,
                   thrust::device_vector<Mat33>& model_orientations,
                   thrust::device_vector<Cov3>& model_shapes,
                   int model_size,
                   std::vector<StampedPose>& pose_graph,
                   /*bool global,*/
                   int time);
    void applyGraphToModel(thrust::device_vector<float3>& model_positions,
                           thrust::device_vector<Mat33>& model_orientations,
                           thrust::device_vector<Cov3>& model_shapes,
                           int model_size);
    void applyGraphToPoses(std::vector<StampedPose>& pose_graph);

private:
    bool initialised;

    int lastDeformTime;

    const double wRot;
    const double wReg;
    const double wCon;

    static const int nbNeighbours = 4;

    static const int nbVariables = 12;
    static const int eRotRows = 6;
    static const int eRegRows = 3;
    static const int eConRows = 3;

    GraphNodes graphNodes;
    VerticesWeights vertexMap;
    VerticesWeights poseMap;
    Constraints constraints;
    thrust::host_vector<float3> sourceVertices;
    thrust::host_vector<int2> vertexTimes;

    CholeskyDecomp* cholesky;

}; // class DeformationGraph

} // supersurfel_fusion

#endif // DEFORMATION_GRAPH_HPP
