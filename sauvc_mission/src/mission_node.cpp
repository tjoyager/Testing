/**
 * SAUVC 2026 NAVIGATION TASK - "HYBRID COMPATIBLE"
 * Kompatibel dengan: sauvc_vision (Hybrid Edition)
 * Logika: 
 * - Input Vision X = 0 (Tengah), -X (Kiri), +X (Kanan)
 * - PID Controller menyesuaikan error menuju 0.
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
    SURFACE_END
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
        
        RCLCPP_INFO(this->get_logger(), "Navigation Mission Started (Hybrid Logic).");
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
                    cmd.angular.z = 0.2; // Putar pelan (Scanning)
                    maintain_depth(cmd);
                }
                break;

            case ALIGN_GATE:
                if (vision_active && has_detection("GATE")) {
                    auto gate = get_detection("GATE");
                    
                    // --- LOGIKA BARU (Hybrid) ---
                    // Karena Vision mengirim 0 sebagai titik tengah:
                    // Jika gate.x positif (Kanan), Robot harus putar Kanan (negatif z?) -> Cek koordinat ROS (ENU)
                    // Standar ROS: Yaw Positif = Putar Kiri (CCW).
                    // Jika objek ada di KANAN (x > 0), kita mau putar KANAN (Yaw Negatif).
                    // Maka rumusnya: -Kp * error
                    
                    double error_yaw = gate.x; // Jarak dari tengah (0)
                    
                    // Tuning PID (P-Control)
                    // 0.003 adalah gain. Semakin besar = semakin agresif (awas overshoot)
                    cmd.angular.z = -error_yaw * 0.003; 
                    
                    // Maju pelan sambil koreksi (Curve motion)
                    cmd.linear.x = 0.3;

                    // Logika Tembus (Jika Area sudah besar)
                    if (gate.distance > 45000) { 
                        state_ = PASS_THROUGH_GATE;
                        pass_through_timer_ = this->now();
                        RCLCPP_INFO(this->get_logger(), "GATE LOCKED! FULL SPEED!");
                    }
                } else {
                    // Jika hilang sebentar, diam dulu (tunggu Vision Counter pulih)
                    // atau kembali SEARCH jika lama hilang
                    cmd.linear.x = 0.0;
                    // state_ = SEARCH_GATE; 
                }
                maintain_depth(cmd);
                break;

            case PASS_THROUGH_GATE:
                // Blind pass (Maju buta lurus)
                cmd.linear.x = 0.6; 
                cmd.angular.z = 0.0; 
                maintain_depth(cmd);
                
                // Maju selama 5 detik
                if ((this->now() - pass_through_timer_).seconds() > 5.0) {
                    state_ = SURFACE_END; // Atau lanjut ke task Drum
                    RCLCPP_INFO(this->get_logger(), "Gate Passed. Mission Complete.");
                }
                break;

            case SURFACE_END:
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
