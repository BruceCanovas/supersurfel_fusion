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


#include <ros/ros.h>
#include "supersurfel_fusion_rgbd_benchmark_node.hpp"

using namespace supersurfel_fusion;


int main(int argc, char** argv)
{
  ros::init(argc, argv, "supersurfel_fusion_rgbd_benchmark_node");

  SupersurfelFusionRGBDBenchmarkNode node;

  dynamic_reconfigure::Server<supersurfel_fusion::SupersurfelFusionRGBDBenchmarkConfig> srv;
  dynamic_reconfigure::Server<supersurfel_fusion::SupersurfelFusionRGBDBenchmarkConfig>::CallbackType cb;
  cb = boost::bind(&SupersurfelFusionRGBDBenchmarkNode::configCallback, &node, _1, _2);
  srv.setCallback(cb);

  node.run();

  return 0;
}
