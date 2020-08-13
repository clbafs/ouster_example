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
#include <pcl/filters/conditional_removal.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/region_growing.h>//区域增长点云分割算法
#include <pcl/filters/conditional_removal.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_line.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/sample_consensus/sac_model_circle.h>//add for 2d circle fitting
#include <pcl/common/transforms.h>
#include <pcl/features/moment_of_inertia_estimation.h>

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
#include "ouster_ros/LidarInfo.h"
#include "ouster_ros/LidarInfoArray.h"

namespace sensor = ouster::sensor;
namespace viz = ouster::viz;
ros::Publisher cluster_pub;
using namespace std;
using namespace pcl;

struct TClusterInfo {
    int id = 0;
    float l_edge = -1;
    float r_edge = -1;
    float c_edge = -1;
    float l_distance = -1;
    float r_distance = -1;
    float c_distance = -1;
    float min_distance = -1;
    int points_num = -1;
};

void DownSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_output,
                  float lx, float ly, float lz) {
    pcl::VoxelGrid<pcl::PointXYZ> voxels;
    voxels.setInputCloud(cloud_input);
    voxels.setLeafSize(lx, ly, lz);//0.05 0.05 0.1
    voxels.filter(*cloud_output);
}

void conditionalRemoval(const pcl::PointCloud<pcl::PointXYZ>::Ptr &input_cloud,
                        const pcl::PointCloud<pcl::PointXYZ>::Ptr &output_cloud,
                        float min, float max) {
    pcl::ConditionAnd<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionAnd<pcl::PointXYZ>());
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr
                                      (new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GE,
                                                                               min)));
    range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr
                                      (new pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LE,
                                                                               max)));
    // 创建滤波器并用条件定义对象初始化
    pcl::ConditionalRemoval<pcl::PointXYZ> condRem;//创建条件滤波器
    condRem.setCondition(range_cond); //并用条件定义对象初始化
    condRem.setInputCloud(input_cloud);     //输入点云
    //condRem.setKeepOrganized(true);    //设置保持点云的结构
    // 执行滤波
    condRem.filter(*output_cloud);
}

void CloudConvert(ouster_ros::Cloud &cloud_input, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud_out) {
    pcl::copyPointCloud(cloud_input, *cloud_out);
}

void CloudFilterDependOnCloudQuality(ouster_ros::Cloud &cloud_input, ouster_ros::Cloud &cloud_output) {
    //ouster_ros::Cloud cloud_filter_;
    for (auto pt:cloud_input) {
        // range measure in mm
        // code:xyz RDF
        // ouster:xyz FLU
        float original_x = pt.x;
        float original_y = pt.y;
        float original_z = pt.z;
        if (pt.range > 1000) {
            pt.x = original_y;
            pt.y = -original_z;
            pt.z = original_x;
            cloud_output.points.push_back(pt);
        }
        // todo filter by noise and intensity
    }
}

void EuclideanClusterKdTree(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                            std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> &cloud_eu, float eps,
                            int min_samples_size, std::vector<pcl::PointIndices> clusters_indices) {
    if (cloud->empty() || eps <= 0 || min_samples_size <= 0) {
        return;
    }
    // Creating the KdTree object for the search method of the extraction
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud);
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(eps);
    ec.setMinClusterSize(min_samples_size);
    ec.setMaxClusterSize(6400);
    ec.setSearchMethod(tree);
    ec.setInputCloud(cloud);
    ec.extract(clusters_indices);

    // if the cluster number is 0, lower the threshold for the distance and cluster number
    if (clusters_indices.empty()) {
        ec.setClusterTolerance(1.0);
        ec.setMinClusterSize(5);
        ec.extract(clusters_indices);
    }
    for (const auto &cluster : clusters_indices) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);

        for (int index1 : cluster.indices)
            cloud_cluster->points.push_back(cloud->points[index1]);
        cloud_cluster->width = static_cast<uint32_t>(cloud_cluster->points.size());
        cloud_cluster->height = 1;
        cloud_cluster->is_dense = true;
        cloud_eu.push_back(cloud_cluster);
    }
}

/*void GetOutput(const std::vector<pcl::PointIndices> &clusters_indices,
                                    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters,
                                    std::vector<TClusterInfo> &cluster_info) {
    std::vector<int> NumPoints;
    for (auto &cluster : clusters) {
        NumPoints.push_back(cluster->points.size());
    }
    sort(NumPoints.begin(), NumPoints.end(), std::greater<int>());
    for (int i = 0; i < int(clusters.size()); ++i) {
        TClusterInfo info;
        info.id = i;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
        //创建条件定义对象
        pcl::ConditionOr<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionOr<pcl::PointXYZ>());
        pcl::ConditionOr<pcl::PointXYZ>::Ptr range_cond2(new pcl::ConditionOr<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered2(new pcl::PointCloud<pcl::PointXYZ>);
        kdTree.setInputCloud(clusters[i]);

        pcl::PointXYZ minPt, maxPt, cenPt, min_xPt, max_xPt, min_zPt;
        // Get the min value and the max value of the cluster, not the point
        getMinMax3D(*clusters[i], minPt, maxPt);
        //为条件定义对象添加比较算子
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr
                                          (new pcl::FieldComparison<pcl::PointXYZ>("x", ComparisonOps::LE,
                                                                                   minPt.x)));
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr
                                          (new pcl::FieldComparison<pcl::PointXYZ>("x", ComparisonOps::GE,
                                                                                   maxPt.x)));
        // 创建滤波器并用条件定义对象初始化
        pcl::ConditionalRemoval<pcl::PointXYZ> condRem;//创建条件滤波器
        condRem.setCondition(range_cond); //并用条件定义对象初始化
        condRem.setInputCloud(clusters[i]);     //输入点云
        //condRem.setKeepOrganized(true);    //设置保持点云的结构
        // 执行滤波
        condRem.filter(*cloud_filtered);
        // this cloud only has two points
        if (cloud_filtered->points[0].x < cloud_filtered->points[1].x) {
            min_xPt = cloud_filtered->points[0];
            max_xPt = cloud_filtered->points[1];
        } else {
            min_xPt = cloud_filtered->points[1];
            max_xPt = cloud_filtered->points[0];
        }

        // Get the min distance
        pcl::ConditionalRemoval<pcl::PointXYZ> condRem2;//创建条件滤波器
        range_cond2->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr
                                           (new pcl::FieldComparison<pcl::PointXYZ>("z",
                                                                                    pcl::ComparisonOps::LE,
                                                                                    minPt.z)));
        condRem2.setCondition(range_cond2); //并用条件定义对象初2始化
        condRem2.setInputCloud(clusters[i]);  //输入点云
        // 执行滤波
        condRem2.filter(*cloud_filtered2);
        min_zPt = cloud_filtered2->points[0];
        //compute3DCentroid
        Eigen::Vector4f centroid;
        compute3DCentroid(*clusters[i], centroid);
        cenPt.x = centroid[0];
        cenPt.y = centroid[1];
        cenPt.z = centroid[2];
        info.center = cenPt;
        info.points_num = static_cast<int>(clusters[i]->points.size());
        info.cluster = clusters[i];

        if (clusters[i]->empty()) {
            return;
        }
        int K = 10;
        if (int(clusters[i]->size()) < K) {
            K = clusters[i]->size();
        }
        info.l_distance = GetKAverageDistance(kdTree, min_xPt, K, clusters[i]);
        //info.l_distance = sqrt(min_xPt.x * min_xPt.x + min_xPt.z * min_xPt.z);
        //info.r_distance = sqrt(max_xPt.x * max_xPt.x + max_xPt.z * max_xPt.z);
        info.r_distance = GetKAverageDistance(kdTree, max_xPt, K, clusters[i]);
        info.c_distance = GetKAverageDistance(kdTree, cenPt, K, clusters[i]);
        info.min_distance = GetKAverageDistance(kdTree, min_zPt, K, clusters[i]);

        pcl::PointXYZ min_average_point = GetKAveragePoint(kdTree, min_xPt, K, clusters[i]);
        pcl::PointXYZ max_average_point = GetKAveragePoint(kdTree, max_xPt, K, clusters[i]);

        //auto os = lidar_yaw_offset_;
        info.l_edge = static_cast<float>(0.5 + (2 * (rad2dgr(atan(min_average_point.x / min_average_point.z)) +
                                                     lidar_yaw_offset_) / CE30_FOV) * 0.5);
        info.r_edge = static_cast<float>(0.5 + (2 * (rad2dgr(atan(max_average_point.x / max_average_point.z)) +
                                                     lidar_yaw_offset_) / CE30_FOV) * 0.5);
        if (info.l_edge < 0) info.l_edge = 0;
        if (info.r_edge > 1) info.r_edge = 1;

        auto visible_angle = dgr2rad(h_fov_);
        float search_radius = cenPt.z * sin(visible_angle);
        bool flag = false;
        if ((cenPt.x - min_xPt.x) < search_radius && (cenPt.y - min_xPt.y) < search_radius &&
            (max_xPt.x - cenPt.x) < search_radius && (max_xPt.y - cenPt.y < search_radius)) {
            flag = true;
        }
        info.visible_to_cam = flag;

        vector<int> pointNKNSearch(static_cast<unsigned long>(K));
        vector<float> pointNKN(static_cast<unsigned long>(K));
        PointCloud<PointXYZ>::Ptr cloud_select(new PointCloud<PointXYZ>);
        if (kdTree.nearestKSearch(min_zPt, K, pointNKNSearch, pointNKN) > 0) {
            for (int j = 0; j < int(pointNKNSearch.size()); ++j) {
                cloud_select->points.push_back(clusters[i]->points[pointNKN[j]]);
            }
        }
        Eigen::Vector4f min_centroid;
        compute3DCentroid(*cloud_select, min_centroid);
        info.min_z_pt = {min_centroid[0], min_centroid[1], min_centroid[2]};

        if (clusters[i]->size() > 100) {
            K = 100;
            vector<int> pointIdNKNSearch(static_cast<unsigned long>(K));
            vector<float> pointNKNSquaredDistance(static_cast<unsigned long>(K));
            PointCloud<PointXYZ>::Ptr cloud_selected(new PointCloud<PointXYZ>);
            if (kdTree.nearestKSearch(cenPt, K, pointIdNKNSearch, pointNKNSquaredDistance) > 0) {
                for (int j = 0; j < int(pointIdNKNSearch.size()); ++j) {
                    cloud_selected->points.push_back(clusters[i]->points[pointNKNSquaredDistance[j]]);
                }
            }
            PointCloud<Normal>::Ptr normal_near_center(new PointCloud<Normal>);
            NormalEstimation<PointXYZ, Normal> normal_estimator;
            normal_estimator.setInputCloud(cloud_selected);
            normal_estimator.setKSearch(K - 1);
            normal_estimator.compute(*normal_near_center);
            float a = 0, b = 0, c = 0;
            for (auto &point : normal_near_center->points) {
                a += point.normal_x;
                b += point.normal_y;
                c += point.normal_z;
            }
            a /= normal_near_center->points.size();
            b /= normal_near_center->points.size();
            c /= normal_near_center->points.size();
            float norm = sqrt(a * a + b * b + c * c);
            info.normal = {a / norm, b / norm, c / norm};
        }

        //printf("cluster's size is %d\n",clusters.size());
        if (clusters.size() > 3) {
            if (info.points_num >= NumPoints[2]) {
                if (cluster_info.size() < 3) {
                    cluster_info.push_back(info);
                }
            }
        } else {
            cluster_info.push_back(info);
        }
        //PrintClusterInfo(info, i);
    }
    }
}*/

float GetKAverageDistance(const pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree, pcl::PointXYZ pt, int K,
                          const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster) {
    std::vector<int> pointIdxNKNSearch(static_cast<unsigned long>(K));
    std::vector<float> pointNKNSquaredDistance(static_cast<unsigned long>(K));
    if (kdTree.nearestKSearch(pt, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
        float distance = 0;
        float total_distance = 0;
        for (int j : pointIdxNKNSearch) {
            float x_pt = cluster->points[j].x;
            float y_pt = cluster->points[j].y;
            float z_pt = cluster->points[j].z;
            total_distance += sqrt(x_pt * x_pt + y_pt * y_pt + z_pt * z_pt);
        }
        distance = total_distance / K;
        return distance;
    }
    return 0;
}

pcl::PointXYZ GetKAveragePoint(const pcl::KdTreeFLANN<pcl::PointXYZ> &kdTree, pcl::PointXYZ pt, int K,
                               const pcl::PointCloud<pcl::PointXYZ>::Ptr &cluster) {
    std::vector<int> pointIdxNKNSearch(static_cast<unsigned long>(K));
    vector<float> pointNKNSquaredDistance(static_cast<unsigned long>(K));
    if (kdTree.nearestKSearch(pt, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
        float sum_x = 0, sum_y = 0, sum_z = 0;
        PointXYZ average_point;
        for (int j : pointIdxNKNSearch) {
            float x_pt = cluster->points[j].x;
            float y_pt = cluster->points[j].y;
            float z_pt = cluster->points[j].z;
            sum_x += x_pt;
            sum_y += y_pt;
            sum_z += z_pt;
        }
        average_point.x = sum_x / K;
        average_point.y = sum_y / K;
        average_point.z = sum_z / K;
        return average_point;
    }
    return PointXYZ(0, 0, 0);
}


void cloud_handler(const sensor_msgs::PointCloud2::ConstPtr &m) {
    ouster_ros::Cloud cloud_;
    ouster_ros::Cloud cloud_filter_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsmaple_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_conditional_filtered(new pcl::PointCloud<pcl::PointXYZ>);
    std::vector<pcl::PointIndices> clusters_indices;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
    pcl::fromROSMsg(*m, cloud_);

    CloudFilterDependOnCloudQuality(cloud_, cloud_filter_);
    cout << cloud_filter_.size() << endl;
    CloudConvert(cloud_filter_, cloud_xyz_ptr);
    //std::cout<<"condition point num:"<<cloud_xyz_ptr->points.size()<<std::endl;
    conditionalRemoval(cloud_xyz_ptr, cloud_conditional_filtered, 0.5, 30.0);
    if (cloud_conditional_filtered->empty()) {
        return;
    }
    //std::cout<<"condition point num:"<<cloud_conditional_filtered->points.size()<<std::endl;
    //DownSampling(cloud_xyz_ptr,cloud_downsmaple_ptr,0.05,0.05,0.1);
    EuclideanClusterKdTree(cloud_conditional_filtered, clusters, 0.3, 6, clusters_indices);


    std::vector<TClusterInfo> cluster_info;
    std::vector<int> NumPoints;
    for (auto &cluster : clusters) {
        NumPoints.push_back(cluster->points.size());
    }
    sort(NumPoints.begin(), NumPoints.end(), std::greater<int>());
    pcl::PointXYZ minPt, maxPt, cenPt, min_xPt, max_xPt, min_zPt;

    for (int i = 0; i < int(clusters.size()); i++) {
        TClusterInfo info;
        info.id = i;
        pcl::ConditionOr<pcl::PointXYZ>::Ptr range_cond(new pcl::ConditionOr<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
        Eigen::Vector4f centroid;
        compute3DCentroid(*clusters[i], centroid);
        cenPt.x = centroid[0];
        cenPt.y = centroid[1];
        cenPt.z = centroid[2];
        cout << "center pt.x:" << cenPt.x << endl;
        cout << "center pt.z:" << cenPt.z << endl;
        pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
        kdTree.setInputCloud(clusters[i]);
        if (clusters[i]->empty()) {
            return;
        }
        int K = 10;
        if (int(clusters[i]->size()) < K) {
            K = clusters[i]->size();
        }
        info.points_num = static_cast<int>(clusters[i]->points.size());

        info.c_distance = GetKAverageDistance(kdTree, cenPt, K, clusters[i]);

        getMinMax3D(*clusters[i], minPt, maxPt);
        //为条件定义对象添加比较算子
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr
                                          (new pcl::FieldComparison<pcl::PointXYZ>("x", ComparisonOps::LE,
                                                                                   minPt.x)));
        range_cond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr
                                          (new pcl::FieldComparison<pcl::PointXYZ>("x", ComparisonOps::GE,
                                                                                   maxPt.x)));
        // 创建滤波器并用条件定义对象初始化
        pcl::ConditionalRemoval<pcl::PointXYZ> condRem;//创建条件滤波器
        condRem.setCondition(range_cond); //并用条件定义对象初始化
        condRem.setInputCloud(clusters[i]);     //输入点云
        //condRem.setKeepOrganized(true);    //设置保持点云的结构
        // 执行滤波
        condRem.filter(*cloud_filtered);
        // this cloud only has two points
        if (cloud_filtered->points[0].x < cloud_filtered->points[1].x) {
            min_xPt = cloud_filtered->points[0];
            max_xPt = cloud_filtered->points[1];
        } else {
            min_xPt = cloud_filtered->points[1];
            max_xPt = cloud_filtered->points[0];
        }
        info.l_distance = GetKAverageDistance(kdTree, min_xPt, K, clusters[i]);
        info.r_distance = GetKAverageDistance(kdTree, max_xPt, K, clusters[i]);

        pcl::PointXYZ min_average_point = GetKAveragePoint(kdTree, min_xPt, K, clusters[i]);
        pcl::PointXYZ max_average_point = GetKAveragePoint(kdTree, max_xPt, K, clusters[i]);
        info.l_edge = static_cast<float>(0.5 +
                                         (2 * (180 / 3.14 * (atan(min_average_point.x / min_average_point.z))) / 180) *
                                         0.5);
        info.r_edge = static_cast<float>(0.5 +
                                         (2 * (180 / 3.14 * (atan(max_average_point.x / max_average_point.z))) / 180) *
                                         0.5);
        info.c_edge = (info.r_edge + info.l_edge) / 2;

        if (clusters.size() > 3) {
            if (info.points_num >= NumPoints[2]) {
                if (cluster_info.size() < 3) {
                    cluster_info.push_back(info);
                }
            }
        } else {
            cluster_info.push_back(info);
        }

    }



    ouster_ros::LidarInfoArray lidarinfo_array;
    for (const auto &cluster: cluster_info) {
        ouster_ros::LidarInfo lidar_info;
        lidar_info.l_edge = cluster.l_edge;
        lidar_info.r_edge = cluster.r_edge;
        lidar_info.c_edge = cluster.c_edge;
        lidar_info.points_num = cluster.points_num;
        lidar_info.r_distance = cluster.r_distance;
        lidar_info.c_distance = cluster.c_distance;
        lidar_info.l_distance = cluster.l_distance;
        lidar_info.header.stamp = ros::Time::now();//use this as unique id, sensor info keep the same timestamp
        lidar_info.header.frame_id = "lidar_ouster";
        //lidar_info.header.seq = seq;
        lidarinfo_array.lidar_array.push_back(lidar_info);
    }
    cluster_pub.publish(lidarinfo_array);


}

int main(int argc, char **argv) {
    ros::init(argc, argv, "cluster_node");
    ros::Subscriber pc_sub;
    ros::Subscriber pc_one_line_sub;
    ros::NodeHandle nh("~");
    pc_sub =
            nh.subscribe<sensor_msgs::PointCloud2>("/os_cloud_node/one_line_points", 500, cloud_handler);
    cluster_pub = nh.advertise<ouster_ros::LidarInfoArray>("/os_cloud_node/cluster", 10);

    ros::spin();
    return EXIT_SUCCESS;
}
