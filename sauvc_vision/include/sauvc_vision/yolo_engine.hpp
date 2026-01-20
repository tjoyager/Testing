#ifndef YOLO_ENGINE_HPP
#define YOLO_ENGINE_HPP

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <vector>
#include <string>
#include <memory>

struct BBox {
    int class_id;
    float confidence;
    cv::Rect box;
    std::string label;
};

class YoloEngine {
public:
    YoloEngine(const std::string& model_path, float conf_thr, float nms_thr, float score_thr);
    ~YoloEngine() = default;
    std::vector<BBox> run_inference(cv::Mat& frame);

private:
    const int INPUT_W = 640;
    const int INPUT_H = 640;
    float conf_threshold_, nms_threshold_, score_threshold_;
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;
    ov::Tensor input_tensor_;

    void preprocess(const cv::Mat& frame, float& ratio, int& dw, int& dh);
    std::vector<BBox> postprocess(const float* detections, const ov::Shape& out_shape, const cv::Size& original_size, float ratio, int dw, int dh);
};
#endif