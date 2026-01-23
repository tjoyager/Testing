/**
 * SAUVC 2026 MISSION CONTROLLER - "SMOOTH CURVE" EDITION
 * Logika:
 * - Input Vision: x=0 (Tengah), x>0 (Kanan), x<0 (Kiri)
 * - PID: Menggunakan P-Controller negatif untuk koreksi arah.
 * - Gerakan: Maju sambil belok (Curve) untuk manuver halus.
 */

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include "sauvc_msgs/msg/obj_array.hpp"

using std::placeholders::_1;

enum MissionState {
    WAIT_FOR_SENSORS,
    DIVE_TO_DEPTH,
    SEARCH_GATE,
    ALIGN_GATE,         
    PASS_THROUGH_GATE,  
    SEARCH_FLARE,
    ALIGN_BUMP_FLARE,
    SURFACE
};

class MissionNode : public rclcpp::Node {
public:
    MissionNode() : Node("mission_node") {
        pub_vel_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        
        sub_vision_ = this->create_subscription<sauvc_msgs::msg::ObjArray>(
            "/vision/detections", 10, std::bind(&MissionNode::vision_cb, this, _1));
            
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 10, std::bind(&MissionNode::odom_cb, this, _1));

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50), std::bind(&MissionNode::control_loop, this));
        
        state_ = WAIT_FOR_SENSORS;
        target_depth_ = -1.0; 
        current_depth_ = 0.0;
        
        RCLCPP_INFO(this->get_logger(), "Mission Start: Smooth Curve Logic Active");
    }

private:
    void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg) {
        current_depth_ = msg->pose.pose.position.z;
    }

    void vision_cb(const sauvc_msgs::msg::ObjArray::SharedPtr msg) {
        latest_detections_ = *msg;
        last_vision_time_ = this->now();
    }

    void control_loop() {
        geometry_msgs::msg::Twist cmd;
        bool vision_active = (this->now() - last_vision_time_).seconds() < 1.0;

        switch (state_) {
            case WAIT_FOR_SENSORS:
                if (current_depth_ != 0.0) { 
                    state_ = DIVE_TO_DEPTH;
                    RCLCPP_INFO(this->get_logger(), "Sensors OK. Diving...");
                }
                break;

            case DIVE_TO_DEPTH:
                {
                    double error = target_depth_ - current_depth_;
                    if (std::abs(error) < 0.1) {
                        cmd.linear.z = 0.0; 
                        state_ = SEARCH_GATE;
                        RCLCPP_INFO(this->get_logger(), "Depth Reached! Searching Gate...");
                    } else {
                        cmd.linear.z = std::clamp(error * 1.5, -0.5, 0.5); 
                    }
                }
                break;

            case SEARCH_GATE:
                if (vision_active && has_detection("GATE")) {
                    cmd.angular.z = 0.0; 
                    state_ = ALIGN_GATE;
                    RCLCPP_INFO(this->get_logger(), "Gate Found! Aligning...");
                } else {
                    cmd.angular.z = 0.2; // Putar pelan cari target
                    maintain_depth(cmd);
                }
                break;

            // --- INI BAGIAN UTAMA PERUBAHAN ---
            case ALIGN_GATE:
                if (vision_active && has_detection("GATE")) {
                    auto gate = get_detection("GATE");
                    
                    // PERBAIKAN KOORDINAT:
                    // Vision Node mengirim x dimana 0 adalah tengah.
                    // Jadi error yaw adalah nilai x itu sendiri.
                    double err_yaw = gate.x; 
                    
                    // RUMUS PID (Aggressive Curve):
                    // Jika Gate di KANAN (x positif), Robot harus putar KANAN (Yaw Negatif).
                    // Rumus: cmd.angular.z = -1 * Gain * Error
                    // Gain 0.0025 memberikan respons tajam tapi halus.
                    cmd.angular.z = -err_yaw * 0.0025; 
                    
                    // MAJU KONSTAN (Curve Motion):
                    // Robot tidak berhenti untuk belok, tapi maju sambil belok.
                    cmd.linear.x = 0.4; 

                    // Logika Tembus:
                    // Jika jarak/area cukup besar, gas pol!
                    if (gate.distance > 45000) { 
                        state_ = PASS_THROUGH_GATE;
                        pass_through_timer_ = this->now();
                        RCLCPP_INFO(this->get_logger(), "GATE LOCK! BLIND PASS START!");
                    }
                } else {
                    // Jika hilang sebentar, diamkan yaw agar tidak liar
                    cmd.angular.z = 0.0;
                    cmd.linear.x = 0.0;
                    // state_ = SEARCH_GATE; 
                }
                maintain_depth(cmd);
                break;

            case PASS_THROUGH_GATE:
                cmd.linear.x = 0.6; // Speed boost
                cmd.angular.z = 0.0; // Lock direction
                maintain_depth(cmd);
                
                if ((this->now() - pass_through_timer_).seconds() > 5.0) {
                    state_ = SEARCH_FLARE;
                    RCLCPP_INFO(this->get_logger(), "Gate Passed. Searching Flare...");
                }
                break;

            case SEARCH_FLARE:
                if (vision_active && has_detection("FLARE")) {
                    state_ = ALIGN_BUMP_FLARE;
                    RCLCPP_INFO(this->get_logger(), "Flare Found! Attacking...");
                } else {
                    cmd.linear.x = 0.2; 
                    cmd.angular.z = 0.2; 
                    maintain_depth(cmd);
                }
                break;
            
            case ALIGN_BUMP_FLARE:
                 if (vision_active && has_detection("FLARE")) {
                    auto flare = get_detection("FLARE");
                    
                    // Gunakan logika koordinat yang sama (pusat 0)
                    double err_x = flare.x;
                    cmd.angular.z = -err_x * 0.004; // Lebih agresif untuk target kecil
                    cmd.linear.x = 0.3;

                    if (flare.distance > 20000) {
                         RCLCPP_INFO(this->get_logger(), "BUMP!");
                         state_ = SURFACE;
                    }
                 } else {
                     state_ = SEARCH_FLARE;
                 }
                 maintain_depth(cmd);
                 break;

            case SURFACE:
                cmd.linear.x = 0.0;
                cmd.linear.z = 0.5;
                break;
        }

        pub_vel_->publish(cmd);
    }

    void maintain_depth(geometry_msgs::msg::Twist &cmd) {
        double error = target_depth_ - current_depth_;
        if (std::abs(error) > 0.05) {
            cmd.linear.z = error * 1.0;
        }
    }

    bool has_detection(std::string label) {
        for(auto& d : latest_detections_.detections) {
            if (d.label == label) return true;
        }
        return false;
    }

    sauvc_msgs::msg::ObjDetection get_detection(std::string label) {
        for(auto& d : latest_detections_.detections) {
            if (d.label == label) return d;
        }
        return sauvc_msgs::msg::ObjDetection();
    }

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_vel_;
    rclcpp::Subscription<sauvc_msgs::msg::ObjArray>::SharedPtr sub_vision_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    sauvc_msgs::msg::ObjArray latest_detections_;
    rclcpp::Time last_vision_time_;
    rclcpp::Time pass_through_timer_;
    
    MissionState state_;
    double target_depth_;
    double current_depth_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MissionNode>());
    rclcpp::shutdown();
    return 0;
}