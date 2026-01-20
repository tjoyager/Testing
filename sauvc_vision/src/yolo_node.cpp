/**
 * SAUVC 2026 VISION SYSTEM - "HYBRID GACOR" EDITION
 * Gabungan:
 * 1. YOLO (Deteksi Objek Cerdas)
 * 2. EMA Smoothing (Anti-Jitter)
 * 3. LOGIKA LAMA: Coordinate Centering (0 di tengah)
 * 4. LOGIKA LAMA: Stability Counter (Filter Noise)
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

// --- TUNING PARAMETERS ---
const float SMOOTH_ALPHA = 0.7f;     // Smoothing (0.1 - 1.0)
const int STABILITY_THRESH = 5;      // Harus terlihat 5 frame berturut-turut baru publish
const int LOST_TOLERANCE = 3;        // Toleransi hilang 3 frame sebelum counter di-reset

class SauvcVisionNode : public rclcpp::Node {
public:
    SauvcVisionNode() : Node("sauvc_vision_node") {
        this->declare_parameter("model_path", "");
        this->declare_parameter("image_topic", "/camera_sensor/image_raw");
        this->declare_parameter("conf_thresh", 0.45);

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

        RCLCPP_INFO(this->get_logger(), "Vision Hybrid Node Started.");
        RCLCPP_INFO(this->get_logger(), "Logic: Center Coordinates & Stability Filter Active.");
    }

private:
    void process_frame(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try { frame = cv_bridge::toCvCopy(msg, "bgr8")->image; } catch(...) { return; }

        auto yolo_results = engine_->run_inference(frame);
        sauvc_msgs::msg::ObjArray msg_array;
        
        // Melacak ID mana yang terlihat di frame INI
        std::map<std::string, bool> current_frame_hits;

        for (auto& det : yolo_results) {
            if(det.box.area() < 400) continue; 

            // 1. Analisis Warna
            std::string color = get_color_robust(frame, det.box);
            
            // 2. Generate ID Unik
            std::string unique_id = "";
            std::string final_label = "UNKNOWN";
            std::string type = "UNKNOWN";

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

            // Tandai bahwa objek ini terlihat sekarang
            current_frame_hits[unique_id] = true;

            // --- LOGIKA LAMA #1: STABILITY COUNTER ---
            // Tambah counter kehadiran
            stability_counters_[unique_id]++; 
            
            // Jika belum mencapai threshold (misal 5 frame), skip! Jangan kirim dulu.
            if (stability_counters_[unique_id] < STABILITY_THRESH) {
                // Gambar kotak putus-putus/abu-abu di debug (Tanda: Mendeteksi tapi belum yakin)
                cv::rectangle(frame, det.box, cv::Scalar(100,100,100), 1);
                continue; 
            }
            // ----------------------------------------

            // 3. EMA SMOOTHING
            int raw_cx = det.box.x + det.box.width / 2;
            int raw_cy = det.box.y + det.box.height / 2;
            
            if (smooth_history_.find(unique_id) == smooth_history_.end()) {
                smooth_history_[unique_id] = { (float)raw_cx, (float)raw_cy };
            }

            float smooth_x = SMOOTH_ALPHA * raw_cx + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].x;
            float smooth_y = SMOOTH_ALPHA * raw_cy + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].y;
            smooth_history_[unique_id] = { smooth_x, smooth_y };

            // --- LOGIKA LAMA #2: COORDINATE CENTERING ---
            // Mengubah koordinat piksel (0..640) menjadi Cartesius (-320..+320)
            // 0 = Tengah Layar
            int center_x = (int)smooth_x - (frame.cols / 2);
            int center_y = (int)smooth_y - (frame.rows / 2);
            // ------------------------------------------

            sauvc_msgs::msg::ObjDetection obj;
            obj.label = final_label;
            obj.detection_type = type;
            obj.color = color;
            obj.x = center_x; // PENTING: Ini sekarang koordinat tengah
            obj.y = center_y; 
            obj.width = det.box.width;
            obj.height = det.box.height;
            obj.confidence = det.confidence;
            obj.distance = det.box.area(); 
            
            msg_array.detections.push_back(obj);

            // Draw Debug
            cv::Scalar draw_c = (type == "OBSTACLE") ? cv::Scalar(0,165,255) : 
                                (type == "GATE") ? cv::Scalar(0,255,0) : cv::Scalar(255,0,0);
            
            cv::rectangle(frame, det.box, draw_c, 2);
            // Gambar titik tengah (Crosshair)
            cv::circle(frame, cv::Point((int)smooth_x, (int)smooth_y), 5, cv::Scalar(0,0,255), -1); 
            
            // Tampilkan Info: Nama + Counter Stabil
            std::string info = unique_id + " [" + std::to_string(stability_counters_[unique_id]) + "]";
            cv::putText(frame, info, cv::Point(det.box.x, det.box.y-10), 0, 0.6, draw_c, 2);
        }

        // Clean Up Counters (Logika Hilang)
        // Jika objek tidak terlihat di frame ini, kurangi counternya (jangan langsung nol)
        for (auto it = stability_counters_.begin(); it != stability_counters_.end(); ) {
            if (!current_frame_hits[it->first]) {
                // Objek hilang! Kurangi counter
                it->second -= 1;
                if (it->second <= 0) {
                    // Jika sudah nol/minus, hapus dari memori
                    smooth_history_.erase(it->first);
                    it = stability_counters_.erase(it);
                } else {
                    ++it;
                }
            } else {
                // Objek ada, batasi max counter biar ga overflow (optional, misal max 100)
                if (it->second > 100) it->second = 100;
                ++it;
            }
        }

        pub_detections_->publish(msg_array);
        
        sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        pub_debug_->publish(*debug_msg);
    }

    std::string get_color_robust(const cv::Mat& frame, cv::Rect box) {
        // ... (KODE WARNA SAMA SEPERTI SEBELUMNYA, TIDAK BERUBAH) ...
        // ... Copy dari jawaban sebelumnya agar hemat tempat ...
        // (Pastikan fungsi get_color_robust yang ada logic black/white/red/orange tetap ada disini)
        
        cv::Rect roi = box & cv::Rect(0,0, frame.cols, frame.rows);
        int w = roi.width * 0.2; int h = roi.height * 0.2;
        int x = roi.x + (roi.width - w)/2; int y = roi.y + (roi.height - h)/2;
        if (w <= 0 || h <= 0) return "UNKNOWN";
        cv::Mat crop = frame(cv::Rect(x,y,w,h));
        cv::Mat hsv; cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
        cv::Scalar avg = cv::mean(hsv);
        double H = avg[0]; double S = avg[1]; double V = avg[2];
        if (V < 60) return "BLACK"; 
        if (S < 40) return "UNKNOWN"; 
        if (H < 10 || H > 160) return "RED";
        if (H >= 10 && H < 25) return "ORANGE";
        if (H >= 25 && H < 45) return "YELLOW";
        if (H >= 90 && H < 140) return "BLUE"; 
        if (H >= 45 && H < 85) return "GREEN"; 
        return "UNKNOWN";
    }

    struct Point2D { float x, y; };
    std::map<std::string, Point2D> smooth_history_;
    std::map<std::string, int> stability_counters_; // Logic Counter Lama

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
