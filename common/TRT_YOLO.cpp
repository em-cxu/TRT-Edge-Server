
#include "TensorRT_CPP/TRT_inference_engine.hpp"

namespace TRT::YOLO
{       

    /// @brief Flag indicating the initialization status of the model.
    static bool initialized = false;

    static std::unique_ptr<TrtInferenceEngine> engine = nullptr;

    static std::vector<size_t> input_sizes;
    static std::vector<size_t> output_sizes;

    static std::vector<void*> input_data;
    static std::vector<void*> output_data;

    static std::vector<std::string> input_names;
    static std::vector<std::string> output_names;

    /// @brief output_cuda_buffer_izes the detection model by loading it into CPU or GPU memory (implementation-defined)
    /// @param path_to_model Path to the model, such as ONNX or TensorRT engine file.
    /// @return 0 on success.
    int load_model(std::string &path_to_model)
    {
        if (initialized || engine != nullptr)
        {
            std::cerr << "[TRT-YOLO] TensorRT instance already initialized! Initialization aborted." << std::endl;
            return -1;
        }
        
        engine = std::make_unique<TrtInferenceEngine>(path_to_model);
        if (engine == nullptr) 
        {
            std::cerr << "[TRT-YOLO] TRT engine loading failed!" << std::endl;
            return -1;
        }
        
        // Get required buffer sizes
        input_sizes = engine->get_input_sizes_bytes();
        output_sizes = engine->get_output_sizes_bytes();

        // Get the names of inputs and outputs
        input_names = engine->get_input_names();
        output_names = engine->get_output_names();
        
        // Allocate buffers to hold inputs and outputs...
        for (int i = 0; i < MODEL_NUM_INPUTS; i++)
        {
            void* chunk = calloc(input_sizes[i], sizeof(uint8_t));
            input_data.push_back(chunk);
        }        
        for (int i = 0; i < MODEL_NUM_OUTPUTS; i++)
        {
            void* chunk = calloc(output_sizes[i], sizeof(uint8_t));
            output_data.push_back(chunk);
        }

        initialized = true;
        return 0;
    }

    /// @brief Performs inference on the input buffer and stores detection results
    /// @param input_img Input image data (must match model input dimensions)
    /// @param results_detections Output vector for detection results
    /// @return Number of detections (-1 on failure)
    int identify_objects(const std::vector<float> &input_img, std::vector<detected_object_info_t> &results_detections)
    {
        // Validate engine state
        if (!initialized || !engine) 
        {
            std::cerr << "[TRT-YOLO] Engine or buffers not initialized!" << std::endl;
            return -1;
        }

        // Validate single input/output
        if (engine->get_num_inputs() != MODEL_NUM_INPUTS || engine->get_num_outputs() != MODEL_NUM_OUTPUTS) {
            std::cerr << "[TRT-YOLO] Unexpected I/O count ("
                    << engine->get_num_inputs() << " inputs, " 
                    << engine->get_num_outputs() << " outputs)" << std::endl;
            return -1;
        }

        // From model file:
        // images tensor: float32 [1, 3, 640, 640]
        if (input_sizes[0] != input_img.size() * sizeof(float))
        {
            std::cerr << "[TRT-YOLO] Input sizes do not match! Expected: "
                << std::to_string(input_sizes[0]) << " got: " 
                << std::to_string(input_img.size() * sizeof(float)) << std::endl;
            return -1;
        }
        memcpy(input_data[0], input_img.data(), input_img.size() * sizeof(float));

        // Run inference (synchronous)
        bool success = engine->infer_b(
            input_data,    // Input buffers
            input_sizes,   // Input sizes
            output_data, // Output buffers
            output_sizes   // Output sizes
        );

        if (!success) {
            std::cerr << "[TRT-YOLO] Inference failed!" << std::endl;
            return -1;
        }

        // Post-process the results.
        results_buffer.clear();

        // From model file:
        // num_dets tensor: int32 [1,1]
        // bboxes tensor: float32 [1,100,4]
        // scores tensor: float32 [1,100]
        // labels tensor: int32 [1,100]
        int32_t num_dets = *static_cast<int32_t*>(output_data[OUTPUT_INDEX_NUM_DETS]); 
        float* bboxes = static_cast<float*>(output_data[OUTPUT_INDEX_BBOXES]);
        float* scores = static_cast<float*>(output_data[OUTPUT_INDEX_SCORES]);
        int32_t* labels = static_cast<int32_t*>(output_data[OUTPUT_INDEX_LABELS]);

        if (num_dets < 0 || num_dets > 100)
        {
            std::cerr << "[TRT-YOLO] Error: Aborted - Received impossible number of detections (0-100): " << std::to_string(num_dets) << std::endl;
            return -1;
        }

        // Post-process by rejecting low-confidence scores.
        // For now, it is assumed NMS is done within the model.
        int index_best_detection = 0;
        float best_score = -1;
        detected_object_info_t current_obj;
        for (int i = 0; i < num_dets; ++i) 
        {
            float x1 = bboxes[i * 4 + 0];
            float y1 = bboxes[i * 4 + 1];
            float x2 = bboxes[i * 4 + 2];
            float y2 = bboxes[i * 4 + 3];
            float score = scores[i];
            int32_t label = labels[i];
            if (score < CONFIDENCE_SCORE_THRESHOLD)
            {
                continue;
            }                
            current_obj.rect.x = x1;
            current_obj.rect.y = y1;
            current_obj.rect.width = x2 - x1;
            current_obj.rect.height = y2 - y1;
            current_obj.confidence = score;
            current_obj.class_id = label;
            results_detections.push_back(current_obj);
        }

        return results_buffer.size();
    }

    /// @brief Unloads Cuda/Tensor RT resources prior to exit.
    /// Could also in principle be used to re-initialize the engine for a new model.
    void unload_model()
    {
        engine.reset(); engine = nullptr;
        // Deallocate buffers for inputs and outputs...
        for (int i = 0; i < MODEL_NUM_INPUTS; i++)
        {
            free(input_data[i]);
        }        
        for (int i = 0; i < MODEL_NUM_OUTPUTS; i++)
        {
            free(output_data[i]);
        }
        input_data.clear();
        output_data.clear();
        input_sizes.clear();
        output_sizes.clear();
        initialized = false;
    }

} // namespace TRT::YOLO
