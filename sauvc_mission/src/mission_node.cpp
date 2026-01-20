/**
 * SAUVC 2026 MISSION CONTROLLER - "SMOOTH OPERATOR" EDITION
 * Karakteristik:
 * - Simultaneous Yaw & Surge (Belok sambil jalan)
 * - Aggressive P-Gain untuk alignment cepat
 * - Blind Pass Timer untuk menembus gate dengan percaya diri
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
    ALIGN_GATE,         // Visual Servoing (Mulus)
    PASS_THROUGH_GATE,  // Blind Pass (Gas Pol)
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

        // Timer 20Hz (50ms) -> Cukup responsif untuk kontrol halus
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(50), std::bind(&MissionNode::control_loop, this));
        
        state_ = WAIT_FOR_SENSORS;
        target_depth_ = -1.0; // Sesuaikan dengan kedalaman gate di video (~1 meter)
        current_depth_ = 0.0;
        
        RCLCPP_INFO(this->get_logger(), "Mission Start: Waiting for Odom...");
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
                        // Turun cepat tapi terkontrol
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

            // --- INI BAGIAN KUNCI GERAKAN MULUS ---
            case ALIGN_GATE:
                if (vision_active && has_detection("GATE")) {
                    auto gate = get_detection("GATE");
                    
                    // Error Posisi (Pixel)
                    double err_yaw = (320 - gate.x); // 320 = Center Image
                    
                    // TUNING PARAMETER:
                    // 0.004 = Cukup agresif untuk koreksi arah (lihat video detik 0:02-0:04)
                    cmd.angular.z = err_yaw * 0.004; 
                    
                    // Maju konstan biar gerakan mengalir (Curve Motion)
                    cmd.linear.x = 0.4; 

                    // Logika Tembus:
                    // Jika Area Gate > 45.000 (Sangat besar/dekat), langsung tembus!
                    if (gate.distance > 45000) { 
                        state_ = PASS_THROUGH_GATE;
                        pass_through_timer_ = this->now();
                        RCLCPP_INFO(this->get_logger(), "GATE LOCK! BLIND PASS START!");
                    }
                } else {
                    state_ = SEARCH_GATE; // Hilang target, cari lagi
                }
                maintain_depth(cmd);
                break;

            // --- INI AGAR TIDAK RAGU SAAT LEWAT ---
            case PASS_THROUGH_GATE:
                // Gas Lurus tanpa belok (Blind)
                cmd.linear.x = 0.6; // Lebih cepat sedikit
                cmd.angular.z = 0.0; // Kunci arah lurus
                maintain_depth(cmd);
                
                // Maju buta selama 5 detik (pastikan seluruh badan lewat)
                if ((this->now() - pass_through_timer_).seconds() > 5.0) {
                    state_ = SEARCH_FLARE;
                    RCLCPP_INFO(this->get_logger(), "Gate Passed. Searching Flare...");
                }
                break;

            case SEARCH_FLARE:
                // Cari Flare (Prioritas Target, abaikan Obstacle dulu kalau mau simpel)
                if (vision_active && has_detection("FLARE")) {
                    state_ = ALIGN_BUMP_FLARE;
                    RCLCPP_INFO(this->get_logger(), "Flare Found! Attacking...");
                } else {
                    cmd.linear.x = 0.2; // Maju pelan
                    cmd.angular.z = 0.2; // Sambil scanning
                    maintain_depth(cmd);
                }
                break;
            
            case ALIGN_BUMP_FLARE:
                 if (vision_active && has_detection("FLARE")) {
                    auto flare = get_detection("FLARE");
                    double err_x = (320 - flare.x);

                    // Alignment Flare juga harus agresif
                    cmd.angular.z = err_x * 0.005; // Lebih tajam karena flare kecil
                    cmd.linear.x = 0.3; // Dekati perlahan

                    if (flare.distance > 20000) { // Jika sudah dekat sekali
                         RCLCPP_INFO(this->get_logger(), "BUMP!");
                         // Stop atau lanjut tugas berikutnya
                         state_ = SURFACE;
                    }
                 } else {
                     state_ = SEARCH_FLARE;
                 }
                 maintain_depth(cmd);
                 break;

            case SURFACE:
                cmd.linear.x = 0.0;
                cmd.linear.z = 0.5; // Naik ke permukaan
                break;
        }

        pub_vel_->publish(cmd);
    }

    void maintain_depth(geometry_msgs::msg::Twist &cmd) {
        double error = target_depth_ - current_depth_;
        // P-Control simpel untuk jaga kedalaman
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