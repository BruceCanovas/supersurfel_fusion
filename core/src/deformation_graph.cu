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


#include <supersurfel_fusion/deformation_graph.hpp>
#include <supersurfel_fusion/deformation_graph_kernels.cuh>
#include "CholeskyDecomp.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <random>
#include <supersurfel_fusion/cuda_error_check.h>
#include "supersurfel_fusion/matrix_math.cuh"


namespace supersurfel_fusion
{

DeformationGraph::DeformationGraph()
    : initialised(false),
      wRot(1.0),
      wReg(10.0),
      wCon(100.0),
      lastDeformTime(0)
{
    cholesky = new CholeskyDecomp();
}

DeformationGraph::~DeformationGraph()
{
    delete cholesky;
}

void DeformationGraph::clear()
{
    sourceVertices.clear();
    vertexTimes.clear();
    graphNodes.clear();
    vertexMap.clear();
    poseMap.clear();
    constraints.clear();
}

void DeformationGraph::buildGraphFromModel(const thrust::host_vector<float3>& model_positions,
                                           const thrust::host_vector<int2>& model_stamps,
                                           int sample_rate,
                                           Constraints const& cons,
                                           std::vector<StampedPose> const& pose_graph)
{
    //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    clear();

    sourceVertices = model_positions;
    vertexTimes = model_stamps;

    int nb_supersurfels = model_positions.size();

    int nb_samples = nb_supersurfels / sample_rate;

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> uni(0, nb_supersurfels - 1);

    int id = uni(rng);

    float3 position = sourceVertices[id];
    float3 translation = make_float3(0.0f);
    Mat33 rotation = make_mat33(1.0f, 0.0f, 0.0,
                                0.0f, 1.0f, 0.0f,
                                0.0f, 0.0f, 1.0f);
    int stamp = vertexTimes[id].x;
    bool enabled = true;
    int4 neighbours = make_int4(-1, -1, -1, -1);

    graphNodes.push_back(stamp, position, translation, rotation, neighbours, enabled);

    while(int(graphNodes.positions.size()) < nb_samples)
    {
        id += sample_rate;

        if(id > nb_supersurfels - 1)
            id -= nb_supersurfels;

        position = sourceVertices[id];
        stamp = vertexTimes[id].x;

        graphNodes.push_back(stamp, position, translation, rotation, neighbours, enabled);
    }

    thrust::host_vector<int> stamps = graphNodes.stamps;
    thrust::sort_by_key(stamps.begin(), stamps.end(), graphNodes.begin());
    int stamps_size = int(stamps.size());

    int cpt = 0;
    int neigh[nbNeighbours];

    for(int i = 0; i < nbNeighbours / 2; i++)
    {
        for(int n = 0; n < nbNeighbours + 1; n++)
        {
            if(i == n)
                continue;

            neigh[cpt] = n;
            cpt++;
        }

        graphNodes.neighbours[i].x = neigh[0];
        graphNodes.neighbours[i].y = neigh[1];
        graphNodes.neighbours[i].z = neigh[2];
        graphNodes.neighbours[i].w = neigh[3];

        cpt = 0;
    }

    for(int i = nbNeighbours / 2; i < stamps_size - (nbNeighbours / 2); i++)
    {
        for(int n = 0; n < nbNeighbours / 2; n++)
        {
            neigh[cpt] = i - (n + 1);
            cpt++;
            neigh[cpt] = i + n +1;
            cpt++;
        }

        graphNodes.neighbours[i].x = neigh[0];
        graphNodes.neighbours[i].y = neigh[1];
        graphNodes.neighbours[i].z = neigh[2];
        graphNodes.neighbours[i].w = neigh[3];

        cpt = 0;
    }

    for(int i =  stamps_size - (nbNeighbours / 2); i < stamps_size; i++)
    {
        for(int n = stamps_size - (nbNeighbours + 1); n < stamps_size; n++)
        {
            if(i == n)
                continue;
            neigh[cpt] = n;
            cpt++;
        }

        graphNodes.neighbours[i].x = neigh[0];
        graphNodes.neighbours[i].y = neigh[1];
        graphNodes.neighbours[i].z = neigh[2];
        graphNodes.neighbours[i].w = neigh[3];

        cpt = 0;
    }

    constraints = cons;
    setConstraints();
    weightVerticesSeq();
    //std::thread t0(&DeformationGraph::weightVerticesSeq, this);
    weightPosesSeq(pose_graph);

    initialised = true;

    //t0.join();

    //std::cout<<"Deformation graph size: "<<graphNodes.positions.size()<<std::endl;

    //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    //std::cout<<"Graph construction time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<std::endl;
}

void DeformationGraph::setConstraints()
{
    for(size_t i = 0; i < constraints.sources.size(); i++)
    {
        sourceVertices.push_back(constraints.sources[i]);
        constraints.idx[i] = sourceVertices.size() - 1;

        if(constraints.pin[i])
            vertexTimes.push_back(make_int2(constraints.targetStamps[i], constraints.targetStamps[i]));
        else
            vertexTimes.push_back(make_int2(constraints.sourceStamps[i], constraints.targetStamps[i]));
    }
}

void DeformationGraph::weightVerticesSeq()
{
    const int look_back = 15;

    vertexMap.weights.resize(sourceVertices.size());
    vertexMap.nodes.resize(sourceVertices.size());

    int graph_size = int(graphNodes.positions.size());

    //#pragma omp parallel for
    for(size_t i = 0; i < sourceVertices.size(); i++)
    {
        int vertex_time = vertexTimes[i].x;

        int found_index = 0;

        int imin = 0;
        int imax = graph_size - 1;
        int imid = (imin + imax) / 2;

        while(imax >= imin)
        {
            imid = (imin + imax) / 2;

            if(graphNodes.stamps[imid] < vertex_time)
                imin = imid + 1;
            else if(graphNodes.stamps[imid] > vertex_time)
                imax = imid - 1;
            else
                break;
        }

        imin = std::min(imin, graph_size - 1);

        if(std::abs(int64_t(graphNodes.stamps[imin]) - int64_t(vertex_time)) <= std::abs(int64_t(graphNodes.stamps[imid]) - int64_t(vertex_time)) &&
                std::abs(int64_t(graphNodes.stamps[imin]) - int64_t(vertex_time)) <= std::abs(int64_t(graphNodes.stamps[imax]) - int64_t(vertex_time)))
            found_index = imin;
        else if(std::abs(int64_t(graphNodes.stamps[imid]) - int64_t(vertex_time)) <= std::abs(int64_t(graphNodes.stamps[imin]) - int64_t(vertex_time)) &&
                std::abs(int64_t(graphNodes.stamps[imid]) - int64_t(vertex_time)) <= std::abs(int64_t(graphNodes.stamps[imax]) - int64_t(vertex_time)))
            found_index = imid;
        else
            found_index = imax;

        std::vector<std::pair<float, int>> near_nodes;

        if(found_index == graph_size)
            found_index = graph_size - 1;

        int dist_back = 0;

        for(int j = int(found_index); j >= 0; j--)
        {
            std::pair<float, int> new_node;
            new_node.first = length(graphNodes.positions[j] - sourceVertices[j]);
            new_node.second = j;

            near_nodes.push_back(new_node);

            if(++dist_back == look_back)
                break;
        }

        if(dist_back != look_back)
        {
            for(int j = found_index + 1; j < graph_size; j++)
            {
                std::pair<float, int> new_node;
                new_node.first = length(graphNodes.positions[j] - sourceVertices[j]);
                new_node.second = j;

                near_nodes.push_back(new_node);

                if(++dist_back == look_back)
                    break;
            }
        }

        std::sort(near_nodes.begin(), near_nodes.end(), [](const std::pair<float, int>& left, const std::pair<float, int>& right) {return left.first < right.first;});

        float3 vertex_position = sourceVertices[i];
        float dmax = near_nodes.at(nbNeighbours).first;

        float4 weights = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int4 nodes = make_int4(-1, -1, -1, -1);

        float weights_sum = 0.0;

        for(int j = 0; j < nbNeighbours; j++)
        {
            float w = std::pow(1.0f - (length(vertex_position - graphNodes.positions[near_nodes.at(j).second]) / dmax), 2.0f);
            set_coord(weights, j, w);
            set_coord(nodes, j, near_nodes.at(j).second);

            weights_sum += w;
        }

        vertexMap.weights[i] = weights / weights_sum;
        vertexMap.nodes[i] = nodes;

        vertexMap.sort(i);

        //std::cout<<vertexMap.nodes[i].x<<" "<<vertexMap.nodes[i].y<<" "<<vertexMap.nodes[i].z<<" "<<vertexMap.nodes[i].w<<std::endl;
    }
}

void DeformationGraph::weightPosesSeq(std::vector<StampedPose> const& stamped_poses)
{
    const int look_back = 10;

    int graph_size = int(graphNodes.positions.size());

    poseMap.weights.resize(stamped_poses.size());
    poseMap.nodes.resize(stamped_poses.size());

    for(size_t i = 0; i < stamped_poses.size(); i++)
    {
        int pose_time = stamped_poses[i].stamp;

        int found_index = 0;

        int imin = 0;
        int imax = graph_size - 1;
        int imid = (imin + imax) / 2;

        while(imax >= imin)
        {
            imid = (imin + imax) / 2;

            if(graphNodes.stamps[imid] < pose_time)
                imin = imid + 1;
            else if(graphNodes.stamps[imid] > pose_time)
                imax = imid - 1;
            else
                break;
        }

        imin = std::min(imin, graph_size - 1);

        if(abs(int64_t(graphNodes.stamps[imin]) - int64_t(pose_time)) <= abs(int64_t(graphNodes.stamps[imid]) - int64_t(pose_time)) &&
                abs(int64_t(graphNodes.stamps[imin]) - int64_t(pose_time)) <= abs(int64_t(graphNodes.stamps[imax]) - int64_t(pose_time)))
            found_index = imin;
        else if(abs(int64_t(graphNodes.stamps[imid]) - int64_t(pose_time)) <= abs(int64_t(graphNodes.stamps[imin]) - int64_t(pose_time)) &&
                abs(int64_t(graphNodes.stamps[imid]) - int64_t(pose_time)) <= abs(int64_t(graphNodes.stamps[imax]) - int64_t(pose_time)))
            found_index = imid;
        else
            found_index = imax;

        std::vector<std::pair<float, int>> near_nodes;

        if(found_index == graph_size)
            found_index = graph_size - 1;

        int dist_back = 0;

        for(int j = int(found_index); j >= 0; j--)
        {
            std::pair<float, int> new_node;
            new_node.first = length(graphNodes.positions[j] - stamped_poses[j].t);
            new_node.second = j;

            near_nodes.push_back(new_node);

            if(++dist_back == look_back)
                break;
        }

        if(dist_back != look_back)
        {
            for(int j = found_index + 1; j < graph_size; j++)
            {
                std::pair<float, int> new_node;
                new_node.first = length(graphNodes.positions[j] - stamped_poses[j].t);
                new_node.second = j;

                near_nodes.push_back(new_node);

                if(++dist_back == look_back)
                    break;
            }
        }

        std::sort(near_nodes.begin(), near_nodes.end(), [](const std::pair<float, int>& left, const std::pair<float, int>& right) {return left.first < right.first;});

        float3 vertex_position = stamped_poses[i].t;
        float dmax = near_nodes.at(nbNeighbours).first;

        float4 weights = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        int4 nodes = make_int4(-1, -1, -1, -1);

        float weights_sum = 0.0;

        for(int j = 0; j < nbNeighbours; j++)
        {
            float w = std::pow(1.0f - (length(vertex_position - graphNodes.positions[near_nodes.at(j).second]) / dmax), 2.0f);
            set_coord(weights, j, w);
            set_coord(nodes, j, near_nodes.at(j).second);

            weights_sum += w;
        }

        poseMap.weights[i] = weights / weights_sum;
        poseMap.nodes[i] = nodes;

        poseMap.sort(i);
    }
}

void DeformationGraph::computeVertexPosition(int vertex_id, float3& position)
{
    float4 weights = vertexMap.weights[vertex_id];
    int4 nodes = vertexMap.nodes[vertex_id];

    position = make_float3(0.0f);

    float3 src = sourceVertices[vertex_id];

    for(int i = 0; i < nbNeighbours; i++)
    {
        float w = get_coord(weights, i);
        int n = get_coord(nodes, i);

        position += w * (graphNodes.rotations[n] * (src - graphNodes.positions[n]) +
                         graphNodes.positions[n] + graphNodes.translations[n]);
    }
}

float DeformationGraph::constraintError()
{
    float result = 0;

    for(size_t l = 0; l < constraints.sources.size(); l++)
    {
        float3 position;

        computeVertexPosition(constraints.idx[l], position);

        result += length(position - constraints.targets[l]);
    }

    return result / float(constraints.sources.size());
}

void DeformationGraph::applyDeltaSparse(Eigen::VectorXd const& delta)
{
    for(size_t j=0; j<graphNodes.positions.size(); j++)
    {
        graphNodes.rotations[j].rows[0].x += float(delta(nbVariables * j));
        graphNodes.rotations[j].rows[1].x += float(delta(nbVariables * j + 1));
        graphNodes.rotations[j].rows[2].x += float(delta(nbVariables * j + 2));
        graphNodes.rotations[j].rows[0].y += float(delta(nbVariables *j + 3));
        graphNodes.rotations[j].rows[1].y += float(delta(nbVariables *j + 4));
        graphNodes.rotations[j].rows[2].y += float(delta(nbVariables *j + 5));
        graphNodes.rotations[j].rows[0].z += float(delta(nbVariables *j + 6));
        graphNodes.rotations[j].rows[1].z += float(delta(nbVariables *j + 7));
        graphNodes.rotations[j].rows[2].z += float(delta(nbVariables *j + 8));
        graphNodes.translations[j].x += float(delta(nbVariables * j + 9));
        graphNodes.translations[j].y += float(delta(nbVariables * j + 10));
        graphNodes.translations[j].z += float(delta(nbVariables * j + 11));
    }
}

bool DeformationGraph::optimiseGraphSparse(float& error, float& mean_cons_err/*, bool global*/)
{
    mean_cons_err = constraintError();

    //if(global && mean_cons_err < 0.06f)
    //    return false;

    int graph_size = int(graphNodes.positions.size());
    int nb_cons = int(constraints.sources.size());

    int max_rows = (eRotRows + eRegRows * nbNeighbours) * graph_size + eConRows * nb_cons;
    int nb_cols = 0;
    int back_set = nbVariables * graph_size;

    for(int i = 0; i < graph_size; i++)
    {
        graphNodes.enabled[i] = true;//graphNodes.stamps[i] > lastDeformTime;

        if(graphNodes.enabled[i])
        {
            nb_cols += nbVariables;
            back_set -= nbVariables;
        }
    }

    Jacobian jacobian;

    Eigen::VectorXd residual = sparseResidual(max_rows);
    sparseJacobian(jacobian, residual.rows(), nb_cols, back_set);

    error = residual.squaredNorm();

    float last_error = error;
    float error_diff = 0.0f;

    //std::cout<<"Deformation graph optimisation: Initial error = "<<error<<", "<<mean_cons_err<<std::endl;

    int iter = 0;

    while(iter++ < 3)
    {
        Eigen::VectorXd delta = cholesky->solve(jacobian, -residual, iter == 1);
        applyDeltaSparse(delta);

        residual = sparseResidual(max_rows);

        error = residual.squaredNorm();
        error_diff = error -last_error;

        //std::cout<<"Iteration "<<iter<<": "<<error<<std::endl;

        if(!std::isfinite(error) || error > last_error || delta.norm() < 1e-2 || error < 1e-3 || std::fabs(error_diff) < 1e-5 * error)
            break;

        last_error = error;

        sparseJacobian(jacobian, residual.rows(), nb_cols, back_set);
    }

    cholesky->freeFactor();

    mean_cons_err =  constraintError();

    if(!std::isfinite(error))
        return false;

    return true;
}

Eigen::VectorXd DeformationGraph::sparseResidual(const int max_rows)
{
    Eigen::VectorXd residual(max_rows);

    int row = 0;

    int graph_size = int(graphNodes.positions.size());
    int nb_cons = int(constraints.sources.size());

    for(int j = 0; j < graph_size; j++)
    {
        if(graphNodes.enabled[j])
        {
            Mat33 rot = transpose(graphNodes.rotations[j]);

            residual(row) = double(dot(rot.rows[0], rot.rows[1]));
            residual(row + 1) = double(dot(rot.rows[0], rot.rows[2]));
            residual(row + 2) = double(dot(rot.rows[1], rot.rows[2]));
            residual(row + 3) = double(dot(rot.rows[0], rot.rows[0]) - 1.0);
            residual(row + 4) = double(dot(rot.rows[1], rot.rows[1]) - 1.0);
            residual(row + 5) = double(dot(rot.rows[2], rot.rows[2]) - 1.0);

            row += eRotRows;
        }
    }

    for(int j = 0; j < graph_size; j++)
    {
        for(int k = 0; k < nbNeighbours; k++)
        {
            int neigh_id = get_coord(graphNodes.neighbours[j], k);

            if(graphNodes.enabled[neigh_id] || graphNodes.enabled[j])
            {
                float3 reg_res = graphNodes.rotations[j] * (graphNodes.positions[neigh_id] - graphNodes.positions[j]) + graphNodes.positions[j] + graphNodes.translations[j]
                        - (graphNodes.positions[neigh_id] + graphNodes.translations[neigh_id]);

                residual(row) = std::sqrt(wReg) * double(reg_res.x);
                residual(row + 1) = std::sqrt(wReg)* double(reg_res.y);
                residual(row +2) = std::sqrt(wReg) * double(reg_res.z);

                row += eRegRows;
            }
        }
    }

    for(int l = 0; l < nb_cons; l++)
    {
        int cons_id = constraints.idx[l];
        int4 nodes = vertexMap.nodes[cons_id];

        bool node_influences = false;

        for(int k = 0; k < nbNeighbours; k++)
        {
            int node_id = get_coord(nodes, k);

            if(graphNodes.enabled[node_id])
            {
                node_influences = true;
                break;
            }
        }

        if(node_influences)
        {
            float3 position;

            computeVertexPosition(cons_id, position);

            float3 con_res = position - constraints.targets[l];

            residual(row) = double(con_res.x) * std::sqrt(wCon);
            residual(row + 1) = double(con_res.y) * std::sqrt(wCon);
            residual(row + 2) = double(con_res.z) * std::sqrt(wCon);

            row += eConRows;
        }
    }

    residual.conservativeResize(row);

    return residual;
}

void DeformationGraph::sparseJacobian(Jacobian& jacobian, const int nb_rows, const int nb_cols, const int back_set)
{
    std::vector<OrderedJacobianRow*> rows(nb_rows);

    int row = 0;

    int graph_size = int(graphNodes.positions.size());
    int nb_cons = int(constraints.sources.size());

    for(int j = 0; j < graph_size; j++)
    {
        if(graphNodes.enabled[j])
        {
            int col = j * nbVariables;

            Mat33 rot = graphNodes.rotations[j];

            rows[row] = new OrderedJacobianRow(6);
            rows[row + 1] = new OrderedJacobianRow(6);
            rows[row + 2] = new OrderedJacobianRow(6);
            rows[row + 3] = new OrderedJacobianRow(3);
            rows[row + 4] = new OrderedJacobianRow(3);
            rows[row + 5] = new OrderedJacobianRow(3);

            rows[row]->append(col - back_set, rot.rows[0].y);
            rows[row]->append(col + 1 - back_set, rot.rows[1].y);
            rows[row]->append(col + 2 - back_set, rot.rows[2].y);
            rows[row]->append(col + 3 - back_set, rot.rows[0].x);
            rows[row]->append(col + 4 - back_set, rot.rows[1].x);
            rows[row]->append(col + 5 - back_set, rot.rows[2].x);

            rows[row + 1]->append(col - back_set, rot.rows[0].z);
            rows[row + 1]->append(col + 1 - back_set, rot.rows[1].z);
            rows[row + 1]->append(col + 2 - back_set, rot.rows[2].z);
            rows[row + 1]->append(col + 6 - back_set, rot.rows[0].x);
            rows[row + 1]->append(col + 7 - back_set, rot.rows[1].x);
            rows[row + 1]->append(col + 8 - back_set, rot.rows[2].x);

            rows[row + 2]->append(col + 3 - back_set, rot.rows[0].z);
            rows[row + 2]->append(col + 4 - back_set, rot.rows[1].z);
            rows[row + 2]->append(col + 5 - back_set, rot.rows[2].z);
            rows[row + 2]->append(col + 6 - back_set, rot.rows[0].y);
            rows[row + 2]->append(col + 7 - back_set, rot.rows[1].y);
            rows[row + 2]->append(col + 8 - back_set, rot.rows[2].y);

            rows[row + 3]->append(col - back_set, 2*rot.rows[0].x);
            rows[row + 3]->append(col + 1 - back_set, 2*rot.rows[1].x);
            rows[row + 3]->append(col + 2 - back_set, 2*rot.rows[2].x);

            rows[row + 4]->append(col + 3 - back_set, 2*rot.rows[0].y);
            rows[row + 4]->append(col + 4 - back_set, 2*rot.rows[1].y);
            rows[row + 4]->append(col + 5 - back_set, 2*rot.rows[2].y);

            rows[row + 5]->append(col + 6 - back_set, 2*rot.rows[0].z);
            rows[row + 5]->append(col + 7 - back_set, 2*rot.rows[1].z);
            rows[row + 5]->append(col + 8 - back_set, 2*rot.rows[2].z);

            row += eRotRows;
        }
    }

    double sqrt_wreg = std::sqrt(wReg);

    for(int j = 0; j < graph_size; j++)
    {
        int col = j * nbVariables;

        for(int k = 0; k < nbNeighbours; k++)
        {
            int neigh_id = get_coord(graphNodes.neighbours[j], k);

            if(graphNodes.enabled[neigh_id] && graphNodes.enabled[j])
            {
                rows[row] = new OrderedJacobianRow(5);
                rows[row + 1] = new OrderedJacobianRow(5);
                rows[row + 2] = new OrderedJacobianRow(5);

                float3 delta = graphNodes.positions[neigh_id] - graphNodes.positions[j];

                int col_neigh = neigh_id * nbVariables;

                if(col_neigh < col && graphNodes.enabled[neigh_id])
                {
                    rows[row]->append(col_neigh + 9 -back_set, -1.0 * sqrt_wreg);
                    rows[row + 1]->append(col_neigh + 10 -back_set, -1.0 * sqrt_wreg);
                    rows[row + 2]->append(col_neigh + 11 -back_set, -1.0 * sqrt_wreg);
                }

                if(graphNodes.enabled[j])
                {
                    rows[row]->append(col - back_set, double(delta.x)*sqrt_wreg);
                    rows[row]->append(col + 3 - back_set, double(delta.y)*sqrt_wreg);
                    rows[row]->append(col + 6 - back_set, double(delta.z)*sqrt_wreg);
                    rows[row]->append(col + 9 - back_set, 1.0*sqrt_wreg);

                    rows[row + 1]->append(col + 1 - back_set, double(delta.x)*sqrt_wreg);
                    rows[row + 1]->append(col + 4 - back_set, double(delta.y)*sqrt_wreg);
                    rows[row + 1]->append(col + 7 - back_set, double(delta.z)*sqrt_wreg);
                    rows[row + 1]->append(col + 10 - back_set, 1.0 * sqrt_wreg);

                    rows[row + 2]->append(col + 2 - back_set, double(delta.x)*sqrt_wreg);
                    rows[row + 2]->append(col + 5 - back_set, double(delta.y)*sqrt_wreg);
                    rows[row + 2]->append(col + 8 - back_set, double(delta.z)*sqrt_wreg);
                    rows[row + 2]->append(col + 11 - back_set, 1.0*sqrt_wreg);
                }

                if(col_neigh > col && graphNodes.enabled[neigh_id])
                {
                    rows[row]->append(col_neigh + 9 -back_set, -1.0 * sqrt_wreg);
                    rows[row + 1]->append(col_neigh + 10 -back_set, -1.0 * sqrt_wreg);
                    rows[row + 2]->append(col_neigh + 11 -back_set, -1.0 * sqrt_wreg);
                }

                row += eRegRows;
            }
        }
    }

    double sqrt_wcon = std::sqrt(wCon);

    for(int l = 0; l < nb_cons; l++)
    {
        int cons_id = constraints.idx[l];
        float4 weights = vertexMap.weights[cons_id];
        int4 nodes = vertexMap.nodes[cons_id];

        bool node_influences = false;

        for(int k = 0; k < nbNeighbours; k++)
        {
            int node_id = get_coord(nodes, k);

            if(graphNodes.enabled[node_id])
            {
                node_influences = true;
                break;
            }
        }

        if(node_influences)
        {
            float3 src_pos = constraints.sources[l];

            rows[row] = new OrderedJacobianRow(4 * nbNeighbours * 2);
            rows[row + 1] = new OrderedJacobianRow(4 * nbNeighbours * 2);
            rows[row + 2] = new OrderedJacobianRow(4 * nbNeighbours * 2);

            for(int k = 0; k < nbNeighbours; k++)
            {
                int node_id = get_coord(nodes, k);

                if(graphNodes.enabled[node_id])
                {
                    int col = node_id * nbVariables;

                    float w = get_coord(weights, k);

                    float3 delta = (src_pos - graphNodes.positions[node_id]) * w;

                    rows[row]->append(col, double(delta.x)*sqrt_wcon);
                    rows[row]->append(col + 3, double(delta.y)*sqrt_wcon);
                    rows[row]->append(col + 6, double(delta.z)*sqrt_wcon);
                    rows[row]->append(col + 9, double(w)*sqrt_wcon);

                    rows[row + 1]->append(col + 1, double(delta.x)*sqrt_wcon);
                    rows[row + 1]->append(col + 4, double(delta.y)*sqrt_wcon);
                    rows[row + 1]->append(col + 7, double(delta.z)*sqrt_wcon);
                    rows[row + 1]->append(col + 10, double(w)*sqrt_wcon);

                    rows[row + 2]->append(col + 2, double(delta.x)*sqrt_wcon);
                    rows[row + 2]->append(col + 5, double(delta.y)*sqrt_wcon);
                    rows[row + 2]->append(col + 8, double(delta.z)*sqrt_wcon);
                    rows[row + 2]->append(col + 11, double(w)*sqrt_wcon);
                }
            }

            row += eConRows;
        }
    }

    jacobian.assign(rows, nb_cols);
}

bool DeformationGraph::constrain(thrust::device_vector<float3>& model_positions,
                                 thrust::device_vector<Mat33>& model_orientations,
                                 thrust::device_vector<Cov3>& model_shapes,
                                 int model_size,
                                 std::vector<StampedPose>& pose_graph,
                                 /*bool global,*/
                                 int time)
{
    if(initialised)
    {
        float error = 0.0f;
        float mean_cons_error = 0.0f;

        //std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

        bool optimised = optimiseGraphSparse(error, mean_cons_error/*, global*/);

        //std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
        //std::cout<<"Graph optimisation time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<std::endl;

        //std::cout<<"Deformation graph optimised: "<<optimised<<" Deformation final error: " << error << ", " << mean_cons_error << std::endl;

        if(/*!global || */(optimised && error < 0.12f && mean_cons_error < 0.0003f))
        {
            //t1 = std::chrono::high_resolution_clock::now();
            //deform model
            applyGraphToModel(model_positions, model_orientations, model_shapes, model_size);
            //deform fern pose
            applyGraphToPoses(pose_graph);
            //t2 = std::chrono::high_resolution_clock::now();

            //std::cout<<"Graph application time: "<<std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count()<<std::endl;

            lastDeformTime = time;

            return true;
        }
    }

    initialised = false;

    return false;
}

void DeformationGraph::applyGraphToModel(thrust::device_vector<float3>& model_positions,
                                         thrust::device_vector<Mat33>& model_orientations,
                                         thrust::device_vector<Cov3>& model_shapes,
                                         int model_size)
{
    thrust::device_vector<float3> nodes_positions(graphNodes.positions);
    thrust::device_vector<Mat33> nodes_rotations(graphNodes.rotations);
    thrust::device_vector<float3> nodes_translations(graphNodes.translations);
    thrust::device_vector<float4> neighbours_weights(vertexMap.weights);
    thrust::device_vector<int4> neighbours_idx(vertexMap.nodes);

    applyDeformation<<<(model_size + 63) / 64, 64>>>(thrust::raw_pointer_cast(model_positions.data()),
                                                     thrust::raw_pointer_cast(model_orientations.data()),
                                                     thrust::raw_pointer_cast(model_shapes.data()),
                                                     thrust::raw_pointer_cast(nodes_positions.data()),
                                                     thrust::raw_pointer_cast(nodes_rotations.data()),
                                                     thrust::raw_pointer_cast(nodes_translations.data()),
                                                     thrust::raw_pointer_cast(neighbours_weights.data()),
                                                     thrust::raw_pointer_cast(neighbours_idx.data()),
                                                     model_size);
    cudaDeviceSynchronize();
    CudaCheckError();

    //    #pragma omp parallel for
    //    for(int i = 0; i < model_size; i++)
    //    {
    //        float4 weights = vertexMap.weights[i];
    //        int4 neighbours = vertexMap.nodes[i];

    //        float3 pi = model_positions[i];
    //        float3 po = make_float3(0.0f, 0.0f, 0.0f);

    //        Eigen::Quaternionf b_quat;

    //        for(int j = 0; j < 4; j++)
    //        {
    //            float wk = get_coord(weights, j);
    //            int node_id = get_coord(neighbours, j);

    //            Eigen::Vector3f e_gk(graphNodes.positions[node_id].x, graphNodes.positions[node_id].y, graphNodes.positions[node_id].z);
    //            Eigen::Vector3f e_tk(graphNodes.translations[node_id].x, graphNodes.translations[node_id].y, graphNodes.translations[node_id].z);
    //            Eigen::Matrix3f e_Rk;
    //            e_Rk << graphNodes.rotations[node_id].rows[0].x, graphNodes.rotations[node_id].rows[0].y, graphNodes.rotations[node_id].rows[0].z,
    //                    graphNodes.rotations[node_id].rows[1].x, graphNodes.rotations[node_id].rows[1].y, graphNodes.rotations[node_id].rows[1].z,
    //                    graphNodes.rotations[node_id].rows[2].x, graphNodes.rotations[node_id].rows[2].y, graphNodes.rotations[node_id].rows[2].z;
    //            Eigen::Quaternionf qk(e_Rk);

    //            Mat33 Rk = make_mat33(e_Rk(0,0), e_Rk(0,1), e_Rk(0,2),
    //                                  e_Rk(1,0), e_Rk(1,1), e_Rk(1,2),
    //                                  e_Rk(2,0), e_Rk(2,1), e_Rk(2,2));
    //            float3 gk = make_float3(e_gk(0), e_gk(1), e_gk(2));
    //            float3 tk = make_float3(e_tk(0), e_tk(1), e_tk(2));

    //            po += wk*(Rk*(pi-gk)+gk+tk);

    //            b_quat.x() += wk*qk.x();
    //            b_quat.y() += wk*qk.y();
    //            b_quat.z() += wk*qk.z();
    //            b_quat.w() += wk*qk.w();

    //        }

    //        b_quat.normalize();
    //        Eigen::Matrix3f b_rot = b_quat.toRotationMatrix();
    //        Mat33 blended_rotation = make_mat33(b_rot(0,0), b_rot(0,1), b_rot(0,2),
    //                                            b_rot(1,0), b_rot(1,1), b_rot(1,2),
    //                                            b_rot(2,0), b_rot(2,1), b_rot(2,2));

    //        model_orientations[i] = model_orientations[i]*transpose(blended_rotation);
    //        model_shapes[i] = mult_ABAt(blended_rotation, model_shapes[i]);
    //        model_positions[i] = po;
    //    }
}

void DeformationGraph::applyGraphToPoses(std::vector<StampedPose>& pose_graph)
{
    Eigen::Vector3f new_position;
    Eigen::Matrix3f rotation;

    for(size_t i = 0; i < pose_graph.size(); i++)
    {
        float4 weights = poseMap.weights[i];
        int4 neighbours = poseMap.nodes[i];

        new_position = Eigen::Vector3f::Zero();
        rotation = Eigen::Matrix3f::Zero();


        Mat33 R = pose_graph[i].R;
        float3 t = pose_graph[i].t;
        Eigen::Matrix4f pose;
        pose << R.rows[0].x, R.rows[0].y, R.rows[0].z, t.x,
                R.rows[1].x, R.rows[1].y, R.rows[1].z, t.y,
                R.rows[2].x, R.rows[2].y, R.rows[2].z, t.z;

        for(int j = 0; j < 4; j++)
        {
            float w = get_coord(weights, j);
            int node_id = get_coord(neighbours, j);

            Eigen::Vector3f node_position(graphNodes.positions[node_id].x, graphNodes.positions[node_id].y, graphNodes.positions[node_id].z);
            Eigen::Vector3f node_translation(graphNodes.translations[node_id].x, graphNodes.translations[node_id].y, graphNodes.translations[node_id].z);
            Eigen::Matrix3f node_rotation;
            node_rotation << graphNodes.rotations[node_id].rows[0].x, graphNodes.rotations[node_id].rows[0].y, graphNodes.rotations[node_id].rows[0].z,
                    graphNodes.rotations[node_id].rows[1].x, graphNodes.rotations[node_id].rows[1].y, graphNodes.rotations[node_id].rows[1].z,
                    graphNodes.rotations[node_id].rows[2].x, graphNodes.rotations[node_id].rows[2].y, graphNodes.rotations[node_id].rows[2].z;

            new_position += w * (node_rotation * (pose.topRightCorner(3, 1) - node_position) +
                                 node_position + node_translation);

            rotation += w * node_rotation;
        }

        Eigen::Matrix3f new_rotation = rotation * pose.topLeftCorner(3, 3);

        Eigen::JacobiSVD<Eigen::Matrix3f> svd(new_rotation, Eigen::ComputeFullU | Eigen::ComputeFullV);

        pose.topRightCorner(3, 1) = new_position;
        pose.topLeftCorner(3, 3) = svd.matrixU() * svd.matrixV().transpose();

        pose_graph[i].R = make_mat33(pose(0,0), pose(0,1), pose(0,2),
                                     pose(1,0), pose(1,1), pose(1,2),
                                     pose(2,0), pose(2,1), pose(2,2));
        pose_graph[i].t = make_float3(pose(0,3), pose(1,3), pose(2,3));
    }
}

} // supersurfel_fusion
