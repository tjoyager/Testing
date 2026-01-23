/**
 * SAUVC 2026 VISION SYSTEM - "TOP BAR SAMPLING"
 * Update:
 * - Mengambil sampel warna di BAGIAN ATAS kotak (Top Bar) untuk menghindari air di tengah gate.
 * - Output koordinat dipusatkan (0 di tengah).
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp> 
#include <opencv2/highgui.hpp>
#include <map>

#include "sauvc_vision/yolo_engine.hpp"
#include "sauvc_msgs/msg/obj_array.hpp"

using std::placeholders::_1;

const float SMOOTH_ALPHA = 0.6f;     
const int STABILITY_THRESH = 2;      
const int MIN_AREA = 300;            

class SauvcVisionNode : public rclcpp::Node {
public:
    SauvcVisionNode() : Node("sauvc_vision_node") {
        this->declare_parameter("model_path", "");
        this->declare_parameter("image_topic", "/camera_sensor/image_raw");
        this->declare_parameter("conf_thresh", 0.40); 

        std::string model_path = this->get_parameter("model_path").as_string();
        float conf = this->get_parameter("conf_thresh").as_double();

        if(model_path.empty()) {
            RCLCPP_FATAL(this->get_logger(), "Model Path Empty!");
            exit(1);
        }

        engine_ = std::make_unique<YoloEngine>(model_path, conf, 0.4, 0.4);

        pub_detections_ = this->create_publisher<sauvc_msgs::msg::ObjArray>("/vision/detections", 10);
        pub_debug_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/debug", 10);
        
        sub_cam_ = this->create_subscription<sensor_msgs::msg::Image>(
            this->get_parameter("image_topic").as_string(), 10, 
            std::bind(&SauvcVisionNode::process_frame, this, _1));

        RCLCPP_INFO(this->get_logger(), "Vision Node READY. Logic: Top Bar Sampling.");
    }

private:
    void process_frame(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try { frame = cv_bridge::toCvCopy(msg, "bgr8")->image; } catch(...) { return; }

        auto yolo_results = engine_->run_inference(frame);
        sauvc_msgs::msg::ObjArray msg_array;
        
        std::map<std::string, bool> current_frame_hits;

        for (auto& det : yolo_results) {
            if(det.box.area() < MIN_AREA) continue; 

            // 1. Cek Warna (PENTING: Gunakan Top Bar Sampling)
            std::string color = get_color_robust(frame, det.box);
            
            bool is_gate = (det.label == "Gate" || det.label == "gate");
            bool is_flare = (det.label == "Flare" || det.label == "flare");

            // --- LOGIKA ID ---
            if (is_gate) {
                // Cek Flare nyasar
                float aspect_ratio = (float)det.box.height / (float)det.box.width;
                if ((color == "ORANGE" || color == "YELLOW") && aspect_ratio > 1.5) {
                    is_gate = false;
                    is_flare = true;
                    RCLCPP_WARN_ONCE(this->get_logger(), "Gate corrected to Flare");
                } 
            }

            std::string unique_id = "";
            std::string final_label = "UNKNOWN";
            std::string type = "UNKNOWN";

            if (is_gate) {
                unique_id = "GATE";
                final_label = "GATE";
                type = "GATE"; 
            }
            else if (is_flare) {
                if (color == "ORANGE" || color == "YELLOW") {
                    unique_id = "FLARE_OBSTACLE";
                    final_label = "FLARE";
                    type = "OBSTACLE";
                } else {
                    unique_id = "FLARE_" + color; 
                    final_label = "FLARE";
                    type = "TARGET";
                }
            }
            else if (det.label == "Drum" || det.label == "drum" || det.label == "Baskom") {
                unique_id = "DRUM_" + color;
                final_label = "DRUM";
                type = (color == "BLUE") ? "TARGET_MAIN" : "TARGET_SEC";
            }

            if (unique_id.empty()) continue;

            // --- STABILITY & SMOOTHING ---
            current_frame_hits[unique_id] = true;
            stability_counters_[unique_id]++; 

            if (stability_counters_[unique_id] < STABILITY_THRESH) {
                cv::rectangle(frame, det.box, cv::Scalar(128,128,128), 1);
                continue; 
            }

            int raw_cx = det.box.x + det.box.width / 2;
            int raw_cy = det.box.y + det.box.height / 2;
            
            if (smooth_history_.find(unique_id) == smooth_history_.end()) {
                smooth_history_[unique_id] = { (float)raw_cx, (float)raw_cy };
            }

            float smooth_x = SMOOTH_ALPHA * raw_cx + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].x;
            float smooth_y = SMOOTH_ALPHA * raw_cy + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].y;
            smooth_history_[unique_id] = { smooth_x, smooth_y };

            // KOORDINAT TENGAH (0,0 di pusat layar)
            int center_x = (int)smooth_x - (frame.cols / 2);
            int center_y = (int)smooth_y - (frame.rows / 2);

            sauvc_msgs::msg::ObjDetection obj;
            obj.label = final_label;
            obj.detection_type = type;
            obj.color = color;
            obj.x = center_x;
            obj.y = center_y; 
            obj.width = det.box.width;
            obj.height = det.box.height;
            obj.confidence = det.confidence;
            obj.distance = det.box.area(); 
            
            msg_array.detections.push_back(obj);

            // DRAW DEBUG
            cv::Scalar draw_c = (type == "OBSTACLE") ? cv::Scalar(0,165,255) : 
                                (type == "GATE") ? cv::Scalar(0,255,0) : cv::Scalar(255,0,0);
            
            cv::rectangle(frame, det.box, draw_c, 2);
            cv::circle(frame, cv::Point((int)smooth_x, (int)smooth_y), 5, cv::Scalar(0,0,255), -1); 
            
            std::string info = unique_id + " (" + color + ")";
            cv::putText(frame, info, cv::Point(det.box.x, det.box.y-10), 0, 0.6, draw_c, 2);
        }

        // Cleanup
        for (auto it = stability_counters_.begin(); it != stability_counters_.end(); ) {
            if (!current_frame_hits[it->first]) {
                it->second -= 1; 
                if (it->second <= 0) {
                    smooth_history_.erase(it->first);
                    it = stability_counters_.erase(it);
                } else {
                    ++it;
                }
            } else {
                if (it->second > 20) it->second = 20; 
                ++it;
            }
        }

        pub_detections_->publish(msg_array);
        
        sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        pub_debug_->publish(*debug_msg);
    }

    // --- LOGIKA WARNA BARU: Top Bar Sampling ---
    std::string get_color_robust(const cv::Mat& frame, cv::Rect box) {
        cv::Rect roi = box & cv::Rect(0,0, frame.cols, frame.rows);
        
        // Ambil 25% BAGIAN ATAS (Top Bar)
        // Ambil 60% Lebar (Agar aman dari background pinggir)
        int h_crop = roi.height * 0.25; 
        int w_crop = roi.width * 0.6;   
        
        int x_crop = roi.x + (roi.width - w_crop) / 2; // Center Horizontal
        int y_crop = roi.y; // Mulai dari ATAS box (y)
        
        if (w_crop <= 0 || h_crop <= 0) return "UNKNOWN";
        
        cv::Mat crop = frame(cv::Rect(x_crop, y_crop, w_crop, h_crop));
        cv::Mat hsv; cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
        cv::Scalar avg = cv::mean(hsv);
        
        double H = avg[0]; double S = avg[1]; double V = avg[2];

        // 1. Cek Hitam/Gelap (Bar Atas)
        if (V < 70) return "BLACK"; 
        
        // 2. Cek Noise Putih
        if (S < 40) return "UNKNOWN"; 

        // 3. Warna Lain
        if (H < 10 || H > 160) return "RED";      
        if (H >= 10 && H < 25) return "ORANGE";   
        if (H >= 25 && H < 45) return "YELLOW";
        if (H >= 45 && H < 85) return "GREEN";    
        if (H >= 90 && H < 140) return "BLUE";    

        return "UNKNOWN";
    }

    struct Point2D { float x, y; };
    std::map<std::string, Point2D> smooth_history_;
    std::map<std::string, int> stability_counters_;

    std::unique_ptr<YoloEngine> engine_;
    rclcpp::Publisher<sauvc_msgs::msg::ObjArray>::SharedPtr pub_detections_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_cam_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SauvcVisionNode>());
    rclcpp::shutdown();
    return 0;
}