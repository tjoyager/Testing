#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/rendering/Camera.hh>
#include <gazebo_ros/node.hpp>
#include <gazebo_ros/conversions/geometry_msgs.hpp>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <rclcpp/rclcpp.hpp>

#include <ignition/math/Vector3.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Quaternion.hh>

namespace gazebo
{
    class ROVController : public ModelPlugin
    {
    public:
        ROVController() = default;

        void Load(physics::ModelPtr model, sdf::ElementPtr sdf) override
        {
            model_ = model;
            node_ = gazebo_ros::Node::Get(sdf);

            // Subscribe ke cmd_vel
            sub_cmd_vel_ = node_->create_subscription<geometry_msgs::msg::Twist>(
                "/cmd_vel", 10, std::bind(&ROVController::OnCmdVel, this, std::placeholders::_1));
            
            odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);
            pub_camera_ = node_->create_publisher<sensor_msgs::msg::Image>("/camera/image_raw", 10);
            tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);

            update_connection_ = event::Events::ConnectWorldUpdateBegin(
                std::bind(&ROVController::OnUpdate, this));

            // Setup Kamera
            auto sensor = gazebo::sensors::SensorManager::Instance()->GetSensor("camera_sensor");
            if (sensor) {
                camera_sensor_ = std::dynamic_pointer_cast<gazebo::sensors::CameraSensor>(sensor);
                camera_ = camera_sensor_->Camera();
                camera_connection_ = camera_->ConnectNewImageFrame(
                    std::bind(&ROVController::OnNewFrame, this, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4, std::placeholders::_5));
            }
            
            target_lin_vel_ = ignition::math::Vector3d::Zero;
            target_ang_vel_ = ignition::math::Vector3d::Zero;

            RCLCPP_INFO(node_->get_logger(), "ROV 6DOF Controller Loaded Successfully!");
        }

        void OnUpdate()
        {
            // 1. Dapatkan Pose Robot saat ini
            ignition::math::Pose3d pose = model_->WorldPose();
            
            // 2. TRANSFORMASI VEKTOR (KUNCI 6DOF)
            ignition::math::Vector3d world_lin_vel = pose.Rot().RotateVector(target_lin_vel_);
            ignition::math::Vector3d world_ang_vel = pose.Rot().RotateVector(target_ang_vel_);

            // 3. Kecepatan ke Gazebo (Kinematic Control)
            model_->SetLinearVel(world_lin_vel);
            model_->SetAngularVel(world_ang_vel);

            // 4. Publish Odometry
            PublishOdometry(pose);
        }

        void OnCmdVel(const geometry_msgs::msg::Twist::SharedPtr msg)
        {
            // Simpan input joystick ke variabel global class
            // Input ini diasumsikan dalam BODY FRAME (Relatif terhadap robot)
            target_lin_vel_.Set(msg->linear.x, msg->linear.y, msg->linear.z);
            target_ang_vel_.Set(msg->angular.x, msg->angular.y, msg->angular.z);
        }
        
        void PublishOdometry(ignition::math::Pose3d pose)
        {
             nav_msgs::msg::Odometry odom;
             odom.header.stamp = node_->now();
             odom.header.frame_id = "odom";
             odom.child_frame_id = "base_link";
             odom.pose.pose = gazebo_ros::Convert<geometry_msgs::msg::Pose>(pose);
             
             // Twist (Velocity)
             auto linear_vel = model_->WorldLinearVel();
             auto angular_vel = model_->WorldAngularVel();
             odom.twist.twist.linear.x = linear_vel.X();
             odom.twist.twist.linear.y = linear_vel.Y();
             odom.twist.twist.linear.z = linear_vel.Z();
             odom.twist.twist.angular.x = angular_vel.X();
             odom.twist.twist.angular.y = angular_vel.Y();
             odom.twist.twist.angular.z = angular_vel.Z();

             odom_pub_->publish(odom);

             // TF
             geometry_msgs::msg::TransformStamped tf_msg;
             tf_msg.header.stamp = node_->now();
             tf_msg.header.frame_id = "odom";
             tf_msg.child_frame_id = "base_link";
             tf_msg.transform = gazebo_ros::Convert<geometry_msgs::msg::Transform>(pose);
             tf_broadcaster_->sendTransform(tf_msg);
        }

        void OnNewFrame(const unsigned char *image, unsigned int width, unsigned int height, unsigned int depth, const std::string &format)
        {
            sensor_msgs::msg::Image msg;
            msg.header.stamp = node_->now();
            msg.header.frame_id = "camera_frame_link";
            msg.height = height;
            msg.width = width;
            msg.encoding = "bgr8";
            msg.step = width * 3;
            msg.data.resize(height * msg.step);
            memcpy(msg.data.data(), image, height * msg.step);
            pub_camera_->publish(msg);
        }

    private:
        physics::ModelPtr model_;
        std::shared_ptr<rclcpp::Node> node_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_cmd_vel_;
        rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_camera_;
        std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
        event::ConnectionPtr update_connection_;
        
        // Variabel untuk menyimpan target kecepatan
        ignition::math::Vector3d target_lin_vel_;
        ignition::math::Vector3d target_ang_vel_;

        // Camera vars
        gazebo::sensors::CameraSensorPtr camera_sensor_;
        gazebo::rendering::CameraPtr camera_;
        event::ConnectionPtr camera_connection_;
    };
    GZ_REGISTER_MODEL_PLUGIN(ROVController)
}