#include "sauvc_vision/yolo_engine.hpp"
#include <algorithm>

// Pastikan urutan ini SAMA dengan urutan di file data.yaml saat training!
const std::vector<std::string> CLASS_NAMES = {"Baskom", "Flare", "Gate"};

YoloEngine::YoloEngine(const std::string& model_path, float conf_thr, float nms_thr, float score_thr) 
    : conf_threshold_(conf_thr), nms_threshold_(nms_thr), score_threshold_(score_thr) {
    
    // Load Model
    std::shared_ptr<ov::Model> model = core_.read_model(model_path);
    
    // Preprocessing setup
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::RGB);
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({255.0f, 255.0f, 255.0f});
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    
    compiled_model_ = core_.compile_model(model, "CPU");
    infer_request_ = compiled_model_.create_infer_request();
}

void YoloEngine::preprocess(const cv::Mat& frame, float& ratio, int& pad_w, int& pad_h) {
    float scale = std::min((float)INPUT_W / frame.cols, (float)INPUT_H / frame.rows);
    int new_unpad_w = std::round(frame.cols * scale);
    int new_unpad_h = std::round(frame.rows * scale);
    pad_w = (INPUT_W - new_unpad_w) / 2;
    pad_h = (INPUT_H - new_unpad_h) / 2;
    ratio = scale;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_unpad_w, new_unpad_h));
    cv::Mat canvas(cv::Size(INPUT_W, INPUT_H), CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(pad_w, pad_h, new_unpad_w, new_unpad_h)));
    
    cv::Mat rgb_frame;
    cv::cvtColor(canvas, rgb_frame, cv::COLOR_BGR2RGB);
    
    input_tensor_ = infer_request_.get_input_tensor();
    std::memcpy(input_tensor_.data(), rgb_frame.data, INPUT_W * INPUT_H * 3);
}

std::vector<BBox> YoloEngine::run_inference(cv::Mat& frame) {
    float ratio; int dw, dh;
    preprocess(frame, ratio, dw, dh);
    infer_request_.infer();
    const auto& output_tensor = infer_request_.get_output_tensor();
    return postprocess(output_tensor.data<const float>(), output_tensor.get_shape(), frame.size(), ratio, dw, dh);
}

std::vector<BBox> YoloEngine::postprocess(const float* data, const ov::Shape& shape, const cv::Size& orig_size, float ratio, int dw, int dh) {
    std::vector<cv::Rect> boxes;
    std::vector<int> class_ids;
    std::vector<float> confidences;

    // Deteksi Format Output (YOLOv5 vs YOLOv8)
    // YOLOv5/7: [1, 25200, 85] -> dim[1] > dim[2]
    // YOLOv8:   [1, 84, 8400]  -> dim[1] < dim[2] (Transposed)
    
    size_t dim1 = shape[1];
    size_t dim2 = shape[2];

    bool is_yolov8 = (dim1 < dim2); 

    if (is_yolov8) {
        // --- LOGIKA YOLOv8 (Transposed) ---
        // Shape: [1, 4+Classes, Anchors]
        // data layout: [cx, cy, w, h, class0, class1, ...] berulang per anchor tapi transposed
        // Akses memori melompat sebesar jumlah Anchor
        
        size_t num_classes = dim1 - 4;
        size_t num_anchors = dim2;
        
        for (size_t i = 0; i < num_anchors; ++i) {
            // Cari skor kelas tertinggi
            float max_score = 0.0f;
            int best_class_id = -1;
            
            // Loop semua kelas untuk anchor ke-i
            for (size_t c = 0; c < num_classes; ++c) {
                // Lokasi data: (4 + c) * num_anchors + i
                float score = data[(4 + c) * num_anchors + i];
                if (score > max_score) {
                    max_score = score;
                    best_class_id = c;
                }
            }

            if (max_score >= score_threshold_) {
                // Lokasi box: 0*num_anchors+i, 1*num_anchors+i, dst
                float cx = data[0 * num_anchors + i];
                float cy = data[1 * num_anchors + i];
                float w  = data[2 * num_anchors + i];
                float h  = data[3 * num_anchors + i];

                int x = (cx - w * 0.5f - dw) / ratio;
                int y = (cy - h * 0.5f - dh) / ratio;
                
                boxes.emplace_back(x, y, (int)(w / ratio), (int)(h / ratio));
                class_ids.push_back(best_class_id);
                confidences.push_back(max_score);
            }
        }
    } 
    else {
        // --- LOGIKA YOLOv5/v7 (Legacy) ---
        // Shape: [1, Anchors, 5+Classes]
        size_t rows = dim1; 
        size_t dimensions = dim2;

        for (size_t i = 0; i < rows; ++i) {
            const float* current_det = data + i * dimensions;
            
            // Index 4 adalah objectness score
            if (current_det[4] < 0.1f) continue;

            float max_score = 0.0f; 
            int best_class_id = -1;
            
            // Kelas mulai dari index 5
            for (size_t j = 5; j < dimensions; ++j) {
                if (current_det[j] > max_score) { 
                    max_score = current_det[j]; 
                    best_class_id = j - 5; 
                }
            }
            
            float final_conf = current_det[4] * max_score;
            
            if (final_conf >= score_threshold_) {
                float cx = current_det[0]; 
                float cy = current_det[1]; 
                float w = current_det[2]; 
                float h = current_det[3];
                
                int x = (cx - w * 0.5f - dw) / ratio; 
                int y = (cy - h * 0.5f - dh) / ratio;
                
                boxes.emplace_back(x, y, w / ratio, h / ratio);
                class_ids.push_back(best_class_id); 
                confidences.push_back(final_conf);
            }
        }
    }

    // NMS (Non-Maximum Suppression)
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);
    
    std::vector<BBox> results;
    for (int idx : indices) {
        BBox res;
        res.box = boxes[idx] & cv::Rect(0, 0, orig_size.width, orig_size.height);
        res.class_id = class_ids[idx];
        res.confidence = confidences[idx];
        
        // Safety check untuk label
        if (res.class_id >= 0 && res.class_id < (int)CLASS_NAMES.size()) {
            res.label = CLASS_NAMES[res.class_id];
        } else {
            res.label = "Unknown";
        }
        
        results.push_back(res);
    }
    return results;
}
