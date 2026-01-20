/**
 * SAUVC 2026 VISION SYSTEM - "GACOR & STABLE" EDITION
 * Fitur:
 * 1. EMA Smoothing: Menstabilkan koordinat X/Y agar robot tidak jitter.
 * 2. Robust Color ID: Memisahkan Red/Orange/Yellow dengan ketat.
 * 3. Sauvc Msgs Integration.
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

// Parameter Smoothing (0.1 - 1.0). 
// Kecil = Sangat halus tapi agak delay (bagus buat video). 
// Besar = Responsif tapi agak jitter.
// 0.7 adalah sweet spot untuk manuver agresif.
const float SMOOTH_ALPHA = 0.7f; 

class SauvcVisionNode : public rclcpp::Node {
public:
    SauvcVisionNode() : Node("sauvc_vision_node") {
        this->declare_parameter("model_path", "");
        this->declare_parameter("image_topic", "/camera_sensor/image_raw");
        this->declare_parameter("conf_thresh", 0.45);

        std::string model_path = this->get_parameter("model_path").as_string();
        float conf = this->get_parameter("conf_thresh").as_double();

        if(model_path.empty()) {
            RCLCPP_FATAL(this->get_logger(), "Model Path Empty! Cek launch file.");
            exit(1);
        }

        engine_ = std::make_unique<YoloEngine>(model_path, conf, 0.4, 0.4);

        pub_detections_ = this->create_publisher<sauvc_msgs::msg::ObjArray>("/vision/detections", 10);
        pub_debug_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/debug", 10);
        
        sub_cam_ = this->create_subscription<sensor_msgs::msg::Image>(
            this->get_parameter("image_topic").as_string(), 10, 
            std::bind(&SauvcVisionNode::process_frame, this, _1));

        RCLCPP_INFO(this->get_logger(), "Vision Node GACOR Started. Smoothing Alpha: %.2f", SMOOTH_ALPHA);
    }

private:
    void process_frame(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try { frame = cv_bridge::toCvCopy(msg, "bgr8")->image; } catch(...) { return; }

        auto yolo_results = engine_->run_inference(frame);
        sauvc_msgs::msg::ObjArray msg_array;
        
        // Reset tracker frame ini (nanti diisi yang ketemu)
        std::map<std::string, bool> detected_labels;

        for (auto& det : yolo_results) {
            // 1. Filter Noise Area
            if(det.box.area() < 400) continue; 

            // 2. Analisis Warna (Center Crop 20% biar fokus inti)
            std::string color = get_color_robust(frame, det.box);
            
            // 3. Tentukan Label Unik untuk Smoothing (Misal: "FLARE_RED", "GATE")
            std::string unique_id = "";
            std::string final_label = "UNKNOWN";
            std::string type = "UNKNOWN";

            // LOGIKA KLASIFIKASI RULEBOOK SAUVC
            if (det.label == "Gate" || det.label == "gate") {
                unique_id = "GATE";
                final_label = "GATE";
                type = "GATE";
            }
            else if (det.label == "Flare" || det.label == "flare") {
                if (color == "ORANGE") {
                    unique_id = "FLARE_OBSTACLE";
                    final_label = "FLARE";
                    type = "OBSTACLE";
                } else {
                    unique_id = "FLARE_" + color; // FLARE_RED, FLARE_YELLOW
                    final_label = "FLARE";
                    type = "TARGET";
                }
            }
            else if (det.label == "Drum" || det.label == "drum" || det.label == "Baskom") {
                unique_id = "DRUM_" + color;
                final_label = "DRUM";
                type = (color == "BLUE") ? "TARGET_MAIN" : "TARGET_SEC";
            }

            // Jika Unknown, skip smoothing
            if (unique_id.empty()) continue;

            // 4. COORDINATE SMOOTHING (The Magic Sauce)
            int raw_cx = det.box.x + det.box.width / 2;
            int raw_cy = det.box.y + det.box.height / 2;
            
            // Ambil history sebelumnya
            if (smooth_history_.find(unique_id) == smooth_history_.end()) {
                smooth_history_[unique_id] = { (float)raw_cx, (float)raw_cy }; // Init
            }

            // Rumus EMA: New = Alpha * Raw + (1-Alpha) * Old
            float smooth_x = SMOOTH_ALPHA * raw_cx + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].x;
            float smooth_y = SMOOTH_ALPHA * raw_cy + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].y;
            
            // Simpan balik ke history
            smooth_history_[unique_id] = { smooth_x, smooth_y };
            detected_labels[unique_id] = true;

            // 5. Pack Message
            sauvc_msgs::msg::ObjDetection obj;
            obj.label = final_label;
            obj.detection_type = type;
            obj.color = color;
            obj.x = (int)smooth_x;      // Pakai yang sudah halus
            obj.y = (int)smooth_y;      // Pakai yang sudah halus
            obj.width = det.box.width;
            obj.height = det.box.height;
            obj.confidence = det.confidence;
            obj.distance = det.box.area(); 
            
            msg_array.detections.push_back(obj);

            // 6. Draw Debug (Lebih Pro)
            cv::Scalar draw_c = (type == "OBSTACLE") ? cv::Scalar(0,165,255) : 
                                (type == "GATE") ? cv::Scalar(0,255,0) : cv::Scalar(255,0,0);
            
            cv::rectangle(frame, det.box, draw_c, 2);
            
            // Gambar Crosshair di titik halus (Bukti smoothing bekerja)
            cv::circle(frame, cv::Point(obj.x, obj.y), 5, cv::Scalar(0,0,255), -1); 
            
            std::string info = unique_id + " " + std::to_string((int)(det.confidence*100)) + "%";
            cv::putText(frame, info, cv::Point(det.box.x, det.box.y-10), 0, 0.6, draw_c, 2);
        }

        // Clean up history yang sudah lama tidak terlihat (biar memori tidak bocor)
        for (auto it = smooth_history_.begin(); it != smooth_history_.end(); ) {
            if (!detected_labels[it->first]) {
                it = smooth_history_.erase(it);
            } else {
                ++it;
            }
        }

        pub_detections_->publish(msg_array);
        
        sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        pub_debug_->publish(*debug_msg);
    }

    // --- LOGIKA WARNA GAZEBO (Sangat Spesifik) ---
    std::string get_color_robust(const cv::Mat& frame, cv::Rect box) {
        cv::Rect roi = box & cv::Rect(0,0, frame.cols, frame.rows);
        
        // CROP TENGAH 20% (Sangat agresif membuang background)
        int w = roi.width * 0.2; int h = roi.height * 0.2;
        int x = roi.x + (roi.width - w)/2; int y = roi.y + (roi.height - h)/2;
        
        if (w <= 0 || h <= 0) return "UNKNOWN";
        
        cv::Mat crop = frame(cv::Rect(x,y,w,h));
        cv::Mat hsv; cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
        cv::Scalar avg = cv::mean(hsv);
        
        double H = avg[0]; double S = avg[1]; double V = avg[2];

        // 1. BLACK (Gate Bar)
        // Gazebo black biasanya V sangat rendah (<50)
        if (V < 60) return "BLACK"; 

        // 2. WHITE/GREY (Background/Busa) -> Ignore
        if (S < 40) return "UNKNOWN"; 

        // 3. COLOR CLASSIFICATION (Tuned for Gazebo Default Materials)
        // Red    : 0-10 & 160-180
        // Orange : 11-24
        // Yellow : 25-35
        // Blue   : 100-130
        
        if (H < 10 || H > 160) return "RED";
        if (H >= 10 && H < 25) return "ORANGE";
        if (H >= 25 && H < 45) return "YELLOW";
        if (H >= 90 && H < 140) return "BLUE"; // Range biru diperlebar sedikit
        if (H >= 45 && H < 85) return "GREEN"; // Jaga-jaga tiang gate

        return "UNKNOWN";
    }

    struct Point2D { float x, y; };
    std::map<std::string, Point2D> smooth_history_;

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