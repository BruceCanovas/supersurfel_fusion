cmake_minimum_required(VERSION 2.8.3)
project(supersurfel_fusion)

add_compile_options(--std=c++11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

find_package(catkin REQUIRED COMPONENTS
    roscpp
    cv_bridge
    image_transport
    geometry_msgs
    message_filters
    tf
    nav_msgs
    visualization_msgs
    pcl_ros
    tf_conversions
    dynamic_reconfigure
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(OpenCV 3 REQUIRED)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(CUDA REQUIRED)
include("${PROJECT_SOURCE_DIR}/cmake/UseCUDA.cmake")
list(APPEND CUDA_NVCC_FLAGS "--std=c++11 --expt-relaxed-constexpr")

find_package(SuiteSparse REQUIRED)

find_package(G2O REQUIRED)
set(G2O_LIBRARIES ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY} ${G2O_TYPES_SBA} ${G2O_TYPES_SLAM3D} ${G2O_SOLVER_PCG})

set(DARKNET_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/third_party/darknet/include)
set(DARKNET_LIBRARY ${PROJECT_SOURCE_DIR}/third_party/darknet/libdarknet.so)

#generate_messages(
#    DEPENDENCIES
#    )
    
generate_dynamic_reconfigure_options(
    config/SupersurfelFusionRGBDBenchmark.cfg
)

catkin_package(
    CATKIN_DEPENDS message_runtime roscpp cv_bridge image_transport geometry_msgs message_filters tf nav_msgs visualization_msgs pcl_ros tf_conversions dynamic_reconfigure
    )

include_directories(
    ${PROJECT_SOURCE_DIR}/core/include/
    ${PROJECT_SOURCE_DIR}/node/
    ${PROJECT_SOURCE_DIR}/third_party/eigen3/
    #${PROJECT_SOURCE_DIR}/third_party/ORB_CUDA/include/
    ${PROJECT_SOURCE_DIR}/third_party/ORB/
    ${PROJECT_SOURCE_DIR}/third_party/GMS/
    ${PROJECT_SOURCE_DIR}/third_party/DefGraph_utils/include/
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${SUITESPARSE_INCLUDE_DIRS}
    ${DARKNET_INCLUDE_DIR}
)

cuda_include_directories(
    ${PROJECT_SOURCE_DIR}/core/include/
    ${PROJECT_SOURCE_DIR}/third_party/eigen3/
    #${PROJECT_SOURCE_DIR}/third_party/ORB_CUDA/include/
    ${PROJECT_SOURCE_DIR}/third_party/DefGraph_utils/include/
    ${catkin_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${DARKNET_INCLUDE_DIR}
)

#cuda_add_library(orb_cuda
#    ${PROJECT_SOURCE_DIR}/third_party/ORB_CUDA/src/ORBextractor.cc
#    ${PROJECT_SOURCE_DIR}/third_party/ORB_CUDA/src/cuda/Allocator_gpu.cu
#    ${PROJECT_SOURCE_DIR}/third_party/ORB_CUDA/src/cuda/Fast_gpu.cu
#    ${PROJECT_SOURCE_DIR}/third_party/ORB_CUDA/src/cuda/Orb_gpu.cu
#    ${PROJECT_SOURCE_DIR}/third_party/ORB_CUDA/src/cuda/Cuda.cu
#)
#target_link_libraries(orb_cuda
#    ${OpenCV_LIBS}
#)
add_library(orb
    ${PROJECT_SOURCE_DIR}/third_party/ORB/ORBextractor.cpp
)
target_link_libraries(orb
    ${OpenCV_LIBS}
)

add_library(gms
    ${PROJECT_SOURCE_DIR}/third_party/GMS/gms_matcher.cpp
)
target_link_libraries(gms
    ${OpenCV_LIBS}
)

add_library(sparse_vo
    core/src/sparse_vo.cpp
    core/src/local_map.cpp
    core/src/pnp_solver.cpp
)
target_link_libraries(sparse_vo
    #orb_cuda
    orb
    gms
    ${OpenCV_LIBS}
    ${G2O_LIBRARIES}
)

add_library(cholesky_decomp
    ${PROJECT_SOURCE_DIR}/third_party/DefGraph_utils/src/CholeskyDecomp.cpp
)
target_link_libraries(cholesky_decomp
    ${SUITESPARSE_LIBRARIES}
)

cuda_add_library(sfusion
    core/src/supersurfel_fusion.cu
    core/src/TPS_RGBD.cu
    core/src/TPS_RGBD_kernels.cu
    core/src/supersurfel_fusion_kernels.cu
    core/src/dense_registration.cu
    core/src/dense_registration_kernels.cu
    core/src/cached_allocator.cpp
    core/src/ferns.cu
    core/src/ferns_kernels.cu
    core/src/deformation_graph.cu
    core/src/deformation_graph_kernels.cu
    core/src/motion_detection.cu
    core/src/motion_detection_kernels.cu
)
target_link_libraries(sfusion
    ${OpenCV_LIBS}
    ${catkin_LIBRARIES}
    sparse_vo
    gms
    cholesky_decomp
    ${DARKNET_LIBRARY}
    )

add_executable(supersurfel_fusion_node
     node/main.cpp 
     node/supersurfel_fusion_node.cpp
     )
target_link_libraries(supersurfel_fusion_node
    sfusion
    )

add_executable(supersurfel_fusion_rgbd_benchmark_node
     node/main_rgbd_benchmark.cpp 
     node/supersurfel_fusion_rgbd_benchmark_node.cpp
     )
target_link_libraries(supersurfel_fusion_rgbd_benchmark_node
    sfusion
    )
add_dependencies(supersurfel_fusion_rgbd_benchmark_node ${PROJECT_NAME}_gencfg)
