//
// Created by xunzhao on 2020/6/9.
//
#include <utility>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/filters/voxel_grid.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "ouster/autoexposure.h"
#include "ouster/beam_uniformity.h"
#include "ouster/client.h"
#include "ouster/types.h"
#include "ouster_ros/OSConfigSrv.h"
#include "ouster_ros/ros.h"

namespace sensor = ouster::sensor;
namespace viz = ouster::viz;

void DownSampling( pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_output,
                  float lx, float ly, float lz) {
    pcl::VoxelGrid<pcl::PointXYZ> voxels;
    voxels.setInputCloud(cloud_input);
    voxels.setLeafSize(lx, ly, lz);//0.05 0.05 0.1
    voxels.filter(*cloud_output);
}

void CloudConvert(ouster_ros::Cloud &cloud_input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out) {
    pcl::copyPointCloud(cloud_input, *cloud_out);
}
void CloudFilterDependOnCloudQuality(ouster_ros::Cloud &cloud_input,ouster_ros::Cloud &cloud_output){
    //ouster_ros::Cloud cloud_filter_;
    for(auto pt:cloud_input){
        // range measure in mm
        // code:xyz RDF
        // ouster:xyz BRU
        float original_x=pt.x;
        float original_y=pt.y;
        float original_z=pt.z;
        if(pt.range>3000){
            pt.x=original_y;
            pt.y=-original_z;
            pt.z=-original_x;
            cloud_output.points.push_back(pt);
        }
        // todo filter by noise and intensity
    }
}
void cloud_handler(const sensor_msgs::PointCloud2::ConstPtr &m) {
    ouster_ros::Cloud cloud_;
    ouster_ros::Cloud cloud_filter_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsmaple_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*m, cloud_);
    CloudFilterDependOnCloudQuality(cloud_,cloud_filter_);
    CloudConvert(cloud_filter_,cloud_xyz_ptr);
    DownSampling(cloud_xyz_ptr,cloud_downsmaple_ptr,0.05,0.05,0.1);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "cluster_node");
    ros::Subscriber pc_sub;
    ros::NodeHandle nh("~");
    pc_sub =
            nh.subscribe<sensor_msgs::PointCloud2>("/os_cloud_node/points", 500, cloud_handler);

    ros::spin();
    return EXIT_SUCCESS;
}


