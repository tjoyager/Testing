/**
 * SAUVC 2026 VISION SYSTEM - "ROBUST COLOR" EDITION
 * Logika:
 * - YOLO: Mendeteksi keberadaan objek (Gate/Flare/Drum).
 * - OpenCV: Mengambil sampel area tengah (Center Crop) untuk analisis warna dominan.
 * - Logika warna disesuaikan untuk air keruh (Foggy Water).
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
            RCLCPP_FATAL(this->get_logger(), "Model Path Empty! Use: --ros-args -p model_path:=/path/to/model");
            exit(1);
        }

        engine_ = std::make_unique<YoloEngine>(model_path, conf, 0.4, 0.4);

        pub_detections_ = this->create_publisher<sauvc_msgs::msg::ObjArray>("/vision/detections", 10);
        pub_debug_ = this->create_publisher<sensor_msgs::msg::Image>("/vision/debug", 10);
        
        sub_cam_ = this->create_subscription<sensor_msgs::msg::Image>(
            this->get_parameter("image_topic").as_string(), 10, 
            std::bind(&SauvcVisionNode::process_frame, this, _1));

        RCLCPP_INFO(this->get_logger(), "Vision Node READY. Logic: YOLO + Center Color Sampling.");
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

            // 1. Ambil Warna Dominan (Metode Center Sampling)
            std::string color = get_color_simple(frame, det.box);
            
            // 2. Tentukan Tipe Objek berdasarkan YOLO Label + Warna
            std::string final_label = "UNKNOWN";
            std::string type = "UNKNOWN";
            std::string unique_id = "";

            // LOGIKA GATE
            if (det.label == "Gate" || det.label == "gate") {
                // Gate biasanya merah/hitam. Tapi di air keruh, kita percaya YOLO dulu.
                // Kecuali jika warnanya SANGAT Orange/Kuning dan bentuknya tiang, mungkin itu Flare.
                float aspect_ratio = (float)det.box.height / (float)det.box.width;
                
                if ((color == "ORANGE" || color == "YELLOW") && aspect_ratio > 2.0) {
                    final_label = "FLARE";
                    unique_id = "FLARE_CORRECTED";
                    type = "TARGET";
                } else {
                    final_label = "GATE";
                    unique_id = "GATE";
                    type = "GATE";
                }
            }
            // LOGIKA FLARE
            else if (det.label == "Flare" || det.label == "flare") {
                final_label = "FLARE";
                unique_id = "FLARE";
                type = "TARGET"; // Di rulebook SAUVC flare adalah target untuk ditabrak/dijatuhkan
            }
            // LOGIKA DRUM / LAINNYA
            else {
                final_label = det.label;
                unique_id = det.label;
                type = "UNKNOWN";
            }

            if (unique_id.empty()) continue;

            // --- STABILITY & SMOOTHING (Sama seperti sebelumnya) ---
            current_frame_hits[unique_id] = true;
            stability_counters_[unique_id]++; 

            if (stability_counters_[unique_id] < STABILITY_THRESH) {
                // Gambar kotak abu-abu jika belum stabil
                cv::rectangle(frame, det.box, cv::Scalar(128,128,128), 1);
                continue; 
            }

            // Hitung Pusat Massa
            int raw_cx = det.box.x + det.box.width / 2;
            int raw_cy = det.box.y + det.box.height / 2;
            
            if (smooth_history_.find(unique_id) == smooth_history_.end()) {
                smooth_history_[unique_id] = { (float)raw_cx, (float)raw_cy };
            }

            // Low Pass Filter
            float smooth_x = SMOOTH_ALPHA * raw_cx + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].x;
            float smooth_y = SMOOTH_ALPHA * raw_cy + (1.0f - SMOOTH_ALPHA) * smooth_history_[unique_id].y;
            smooth_history_[unique_id] = { smooth_x, smooth_y };

            // Koordinat Pusat (0,0 di tengah layar)
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
            cv::Scalar draw_c = cv::Scalar(0, 255, 0); // Default Hijau
            if (final_label == "FLARE") draw_c = cv::Scalar(0, 165, 255); // Orange
            if (final_label == "GATE") draw_c = cv::Scalar(0, 0, 255); // Merah

            cv::rectangle(frame, det.box, draw_c, 2);
            cv::circle(frame, cv::Point((int)smooth_x, (int)smooth_y), 5, cv::Scalar(255,0,255), -1); 
            
            std::string info = final_label + " [" + color + "]";
            cv::putText(frame, info, cv::Point(det.box.x, det.box.y-10), 0, 0.6, draw_c, 2);
        }

        // Cleanup Counters
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

    // --- LOGIKA WARNA SIMPEL (Center Sampling) ---
    std::string get_color_simple(const cv::Mat& frame, cv::Rect box) {
        // Pastikan ROI aman
        cv::Rect roi = box & cv::Rect(0,0, frame.cols, frame.rows);
        
        // Ambil bagian tengah (50% dari lebar/tinggi box)
        int w_crop = roi.width * 0.5;
        int h_crop = roi.height * 0.5;
        int x_crop = roi.x + (roi.width - w_crop) / 2;
        int y_crop = roi.y + (roi.height - h_crop) / 2;

        if (w_crop <= 0 || h_crop <= 0) return "UNKNOWN";

        cv::Mat crop = frame(cv::Rect(x_crop, y_crop, w_crop, h_crop));
        
        // Convert ke HSV
        cv::Mat hsv; 
        cv::cvtColor(crop, hsv, cv::COLOR_BGR2HSV);
        
        // Hitung rata-rata warna
        cv::Scalar avg = cv::mean(hsv);
        double H = avg[0]; 
        double S = avg[1]; 
        double V = avg[2];

        // LOGIKA WARNA UNTUK AIR KERUH
        // Warna air keruh biasanya Saturation rendah atau Value rendah (gelap).
        
        // 1. Deteksi Merah (Gate/Flare)
        // Rentang merah ada dua di OpenCV (0-10 dan 160-180)
        if ((H < 15 || H > 160) && S > 50) return "RED";

        // 2. Deteksi Oranye/Kuning (Flare)
        if (H >= 15 && H < 40 && S > 50) return "ORANGE"; // Digabung jadi Orange/Yellow

        // 3. Deteksi Hijau (Mungkin tiang gate hijau atau lumut)
        // Hati-hati, air keruh kadang terbaca hijau muda. Naikkan threshold Saturation.
        if (H >= 40 && H < 90 && S > 60) return "GREEN";

        // 4. Deteksi Biru (Drum target)
        if (H >= 90 && H < 130 && S > 50) return "BLUE";

        // 5. Hitam (Gate bar atas sering tampak hitam siluet)
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
