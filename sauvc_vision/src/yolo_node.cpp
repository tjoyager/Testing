/**
 * SAUVC 2026 VISION SYSTEM - "DEBUG & ROBUST EDITION"
 * Fitur:
 * 1. Mencetak SEMUA hasil deteksi mentah ke terminal (untuk debugging).
 * 2. Menggunakan 'Center Sampling' untuk deteksi warna yang lebih stabil di air keruh.
 * 3. Logika klasifikasi yang mengutamakan label YOLO.
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
        // Deklarasi Parameter
        this->declare_parameter("model_path", "");
        this->declare_parameter("image_topic", "/camera_sensor/image_raw");
        this->declare_parameter("conf_thresh", 0.25); // Default diturunkan agar lebih sensitif saat debug

        std::string model_path = this->get_parameter("model_path").as_string();
        float conf = this->get_parameter("conf_thresh").as_double();

        // Cek Path Model
        if(model_path.empty()) {
            RCLCPP_FATAL(this->get_logger(), "Model Path Empty! PASTIKAN ANDA MENJALANKAN DENGAN: --ros-args -p model_path:=/path/to/model.onnx");
            exit(1);
        }

        // Inisialisasi Engine
        // Note: conf threshold dikirim ke engine.
        engine_ = std::make_unique<YoloEngine>(model_path, conf, 0.4, 0.4);

        pub_detections_ = this->create_publisher<sauvc_msgs::msg::ObjArray>("/vision/detections", 10);
        pub_debug_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/debug", 10);
        
        sub_cam_ = this->create_subscription<sensor_msgs::msg::Image>(
            this->get_parameter("image_topic").as_string(), 10, 
            std::bind(&SauvcVisionNode::process_frame, this, _1));

        RCLCPP_INFO(this->get_logger(), "Vision Node READY. Debug Mode: ON");
        RCLCPP_INFO(this->get_logger(), "Model: %s", model_path.c_str());
        RCLCPP_INFO(this->get_logger(), "Confidence Threshold: %.2f", conf);
    }

private:
    void process_frame(const sensor_msgs::msg::Image::SharedPtr msg) {
        cv::Mat frame;
        try { frame = cv_bridge::toCvCopy(msg, "bgr8")->image; } catch(...) { return; }

        // 1. Lakukan Inferensi YOLO
        auto yolo_results = engine_->run_inference(frame);
        
        // --- BAGIAN DEBUGGING (PENTING UNTUK CEK DETEKSI) ---
        // Mencetak info jika ada deteksi APAPUN yang masuk
        if (!yolo_results.empty()) {
             RCLCPP_INFO(this->get_logger(), "--- Frame Detection Summary ---");
             for(auto& d : yolo_results) {
                 RCLCPP_INFO(this->get_logger(), "DETECTED: Label='%s' | Conf=%.2f | Area=%.0f", 
                     d.label.c_str(), d.confidence, d.box.area());
             }
        }
        // ----------------------------------------------------

        sauvc_msgs::msg::ObjArray msg_array;
        std::map<std::string, bool> current_frame_hits;

        for (auto& det : yolo_results) {
            // Filter area kecil (noise)
            if(det.box.area() < MIN_AREA) continue; 

            // 2. Ambil Warna Dominan (Metode Center Sampling)
            std::string color = get_color_simple(frame, det.box);
            
            // 3. Klasifikasi Objek
            std::string final_label = "UNKNOWN";
            std::string type = "UNKNOWN";
            std::string unique_id = "";

            // -> LOGIKA GATE
            if (det.label == "Gate" || det.label == "gate") {
                // Cek rasio aspek untuk membedakan Gate vs Flare jika label salah
                float aspect_ratio = (float)det.box.height / (float)det.box.width;
                
                // Jika sangat tinggi (tiang) dan warnanya oranye/kuning -> Mungkin Flare
                if ((color == "ORANGE" || color == "YELLOW") && aspect_ratio > 2.5) {
                    final_label = "FLARE";
                    unique_id = "FLARE_CORRECTED";
                    type = "TARGET";
                } else {
                    final_label = "GATE";
                    unique_id = "GATE";
                    type = "GATE";
                }
            }
            // -> LOGIKA FLARE
            else if (det.label == "Flare" || det.label == "flare") {
                final_label = "FLARE";
                unique_id = "FLARE";
                type = "TARGET"; 
            }
            // -> LOGIKA LAIN (Drum/Baskom)
            else {
                final_label = det.label;
                unique_id = det.label;
                type = "UNKNOWN";
            }

            if (unique_id.empty()) continue;

            // --- 4. STABILITY & SMOOTHING ---
            current_frame_hits[unique_id] = true;
            stability_counters_[unique_id]++; 

            // Belum cukup stabil? Skip publish, tapi gambar kotak abu-abu
            if (stability_counters_[unique_id] < STABILITY_THRESH) {
                cv::rectangle(frame, det.box, cv::Scalar(128,128,128), 1);
                continue; 
            }

            // Hitung Pusat Massa
            int raw_cx = det.box.x + det.box.width / 2;
            int raw_cy = det.box.y + det.box.height / 2;
            
            // Inisialisasi history jika baru
            if (smooth_history_.find(unique_id) == smooth_history_.end()) {
                smooth_history_[unique_id] = { (float)raw_cx, (float)raw_cy };
            }

            // Low Pass Filter (Smoothing)
            float smooth_x = SMOOTH_ALPHA * raw_cx + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].x;
            float smooth_y = SMOOTH_ALPHA * raw_cy + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].y;
            smooth_history_[unique_id] = { smooth_x, smooth_y };

            // Koordinat Pusat (0,0 di tengah layar) untuk kontrol PID
            int center_x = (int)smooth_x - (frame.cols / 2);
            int center_y = (int)smooth_y - (frame.rows / 2);

            // Isi Pesan ROS
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

            // DRAW DEBUG VISUALIZATION
            cv::Scalar draw_c = cv::Scalar(0, 255, 0); // Default Hijau
            if (final_label == "FLARE") draw_c = cv::Scalar(0, 165, 255); // Orange
            if (final_label == "GATE") draw_c = cv::Scalar(0, 0, 255); // Merah

            cv::rectangle(frame, det.box, draw_c, 2);
            cv::circle(frame, cv::Point((int)smooth_x, (int)smooth_y), 5, cv::Scalar(255,0,255), -1); 
            
            std::string info = final_label + " [" + color + "]";
            cv::putText(frame, info, cv::Point(det.box.x, det.box.y-10), 0, 0.6, draw_c, 2);
        }

        // Cleanup Counters (Hapus objek yang hilang dari layar)
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
                if (it->second > 20) it->second = 20; // Cap max counter
                ++it;
            }
        }

        pub_detections_->publish(msg_array);
        
        sensor_msgs::msg::Image::SharedPtr debug_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        pub_debug_->publish(*debug_msg);
    }

    // --- LOGIKA WARNA SIMPEL (Center Sampling) ---
    // Dioptimalkan untuk air keruh/foggy
    std::string get_color_simple(const cv::Mat& frame, cv::Rect box) {
        cv::Rect roi = box & cv::Rect(0,0, frame.cols, frame.rows);
        
        // Ambil 50% bagian tengah box
        int w_crop = roi.width * 0.5;
        int h_crop = roi.height * 0.5;
        int x_crop = roi.x + (roi.width - w_crop) / 2;
        int y_crop = roi.y + (roi.height - h_crop) / 2;

        if (w_crop <= 0 || h_crop <= 0) return "UNKNOWN";

        cv::Mat crop = frame(cv::Rect(x_crop, y_crop, w_crop, h_crop));
        cv::Mat hsv; 
        cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
        
        cv::Scalar avg = cv::mean(hsv);
        double H = avg[0]; 
        double S = avg[1]; 
        double V = avg[2];

        // LOGIKA DETEKSI WARNA (HSV Range)
        
        // 1. Merah (Gate/Flare)
        // Rentang merah ada di ujung bawah (0-15) dan ujung atas (160-180)
        if ((H < 15 || H > 160) && S > 50) return "RED";

        // 2. Oranye/Kuning (Flare)
        if (H >= 15 && H < 40 && S > 50) return "ORANGE";

        // 3. Hijau (Tiang Gate Kanan)
        if (H >= 40 && H < 90 && S > 60) return "GREEN";

        // 4. Biru (Drum)
        if (H >= 90 && H < 130 && S > 50) return "BLUE";

        // 5. Hitam (Bar Atas Gate - Siluet)
        if (V < 60) return "BLACK";

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
