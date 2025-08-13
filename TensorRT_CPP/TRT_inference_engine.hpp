
#pragma once

#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include <iostream>
#include <vector>
#include <cstring>
#include <string>
#include <memory>
#include <fstream>

using namespace nvinfer1;

// TensorRT Logger wrapper class with detailed log levels
class TrtLogger : public nvinfer1::ILogger 
{
public:
    void log(Severity severity, const char* msg) noexcept override 
    {
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                std::cerr << "[TRT INTERNAL ERROR] " << msg << std::endl;
                break;
            case Severity::kERROR:
                std::cerr << "[TRT ERROR] " << msg << std::endl;
                break;
            case Severity::kWARNING:
                std::cout << "[TRT WARNING] " << msg << std::endl;
                break;
            case Severity::kINFO:
                std::cout << "[TRT INFO] " << msg << std::endl;
                break;
            case Severity::kVERBOSE:
                #ifdef TRT_DEBUG_VERBOSE
                std::cout << "[TRT VERBOSE] " << msg << std::endl;
                #endif
                break;
            default:
                std::cout << "[TRT UNKNOWN] " << msg << std::endl;
        }
    }
};

/// @brief Generic TensorRT inference engine class.
/// This implements low-level inference functions.
class TrtInferenceEngine 
{
public:
    explicit TrtInferenceEngine(const std::string& engine_path)
    {
        this->load_engine(engine_path);
        this->calculate_model_parameters();
        this->allocate_buffers();
    }

    ~TrtInferenceEngine()
    {                    
        this->deallocate_buffers();
        this->shutdown_engine();
    }

    /// @brief Performs inference using the loaded model. The input/output must be sized appropriately.
    /// Recommended to pre-allocate them using get_input_size_bytes() and get_output_size_bytes()
    /// prior to calling this function. 
    /// @param input_buf An std::vector containing the pointer to the input buffer(s).
    /// @param size_in_param An std::vector containing the size (bytes) of the elements of input_buf. 
    /// Should == this->get_input_size_bytes()
    /// @param output_buf An std::vector containing the pointer to the output buffer.
    /// @param size_out_param An std::vector containing the size (bytes) of the elements of the output buffer. 
    /// Should == this->get_output_size_bytes()
    /// @return TRUE if inference succeeded.
    bool infer_b(const std::vector<void*> &input_buf, std::vector<size_t> &size_in_param, 
        std::vector<void*> &output_buf, std::vector<size_t> &size_out_param);

    /// @brief Performs inference using the loaded model. The input/output must be sized appropriately.
    /// Recommended to pre-allocate them using get_input_size_bytes() and get_output_size_bytes()
    /// prior to calling this function. 
    /// This function enqueues the inference result using Cuda stream so that you can asynchronously
    /// execute other functions in the meantime.
    /// @param input_buf The pointer to the input buffer.
    /// @param size_in_param The size (bytes) of the input buffer. 
    /// Should == this->get_input_size_bytes()
    /// @return TRUE if successfully enqueued.
    bool infer_async(const std::vector<void*> &input_buf, std::vector<size_t> &size_in_param);

    /// @brief Performs inference using the loaded model. The input/output must be sized appropriately.
    /// Recommended to pre-allocate them using get_input_size_bytes() and get_output_size_bytes()
    /// prior to calling this function. 
    /// This function retrieves an async result from the Cuda stream (if available).
    /// @param output_buf The pointer to the output buffer.
    /// @param size_out_param The size (bytes) of the output buffer. 
    /// Should == this->get_output_size_bytes()
    /// @return Output size, or 0 on failure.
    size_t retrieve_infer_result_async(const std::vector<void*> &output_buf, 
        std::vector<size_t> &size_out_param);

    /// @brief Get number of model inputs
    int get_num_inputs() const noexcept { return this->num_inputs_; }

    /// @brief Get number of model outputs
    int get_num_outputs() const noexcept { return this->num_outputs_; }

    /// @brief Get shapes for all inputs
    const std::vector<std::vector<int>>& get_input_shapes() const 
    {
        return this->input_shapes_;
    }

    /// @brief Get shapes for all outputs
    const std::vector<std::vector<int>>& get_output_shapes() const 
    {
        return this->output_shapes_;
    }

    /// @brief Get element counts for all inputs
    const std::vector<size_t>& get_input_elements() const 
    {
        return this->trt_input_element_counts_;
    }

    /// @brief Get element counts for all outputs
    const std::vector<size_t>& get_output_elements() const 
    {
        return this->trt_output_element_counts_;
    }

    /// @brief Get byte sizes for all inputs
    /// @param element_size Size per element in bytes (default float32)
    std::vector<size_t> get_input_size_bytes(size_t element_size = sizeof(float)) const 
    {
        const auto& elems = get_input_elements();
        std::vector<size_t> sizes(this->num_inputs_);
        for (int i = 0; i < this->num_inputs_; i++)
        {
            sizes[i] = trt_input_element_counts_[i] * element_size;
        }
        return sizes;
    }

    /// @brief Get byte sizes for all outputs  
    /// @param element_size Size per element in bytes (default float32)
    std::vector<size_t> get_output_size_bytes(size_t element_size = sizeof(float)) const 
    {
        std::vector<size_t> sizes(this->num_outputs_);
        for (int i = 0; i < this->num_outputs_; i++)
        {
            sizes[i] = trt_output_element_counts_[i] * element_size;
        }
        return sizes;
    }

private:

    // TensorRT logger instance
    TrtLogger logger_;

    // TensorRT runtime resources - must be initialized in constructor.
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // Model dimensions - these are assigned in the constructor when model is loaded.
    int num_inputs_; // Number of inputs (e.g. 1 image for YOLO)
    int num_outputs_; // Number of outputs (e.g. 4 objects for YOLO)
    
    std::vector<nvinfer1::Dims> input_dims_;
    std::vector<nvinfer1::Dims> output_dims_;

    std::vector<std::vector<int>> input_shapes_;
    std::vector<std::vector<int>> output_shapes_;

    // The names of each input or output
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    // Dynamically allocated buffers for inference.
    // Must be allocated in the constructor, and never again.
    std::vector<void*> input_cuda_buffers_;  // Stores pointers to input buffers
    std::vector<void*> output_cuda_buffers_; // Stores pointers to output buffers
    std::vector<void*> bindings_; // References the combined input_cuda_buffers_ and output_cuda_buffers_. 

    // The size of the I/O - in elements only - to get bytes you need to call
    // this->get_output_size_bytes() or this->get_input_size_bytes()
    std::vector<size_t> trt_input_element_counts_;
    std::vector<size_t> trt_output_element_counts_;

    /// @brief Load the .engine file from a text string.
    /// This should only be called at the start of the program.
    /// @param engine_path Path to the .engine file...
    void load_engine(const std::string& engine_path);

    /// @brief Calculates and validates model parameters, populating input/output dimensions
    /// @throws std::runtime_error if model structure is invalid
    void calculate_model_parameters();
    
    /// @brief Helper function to validate and calculate the elements.
    size_t validate_and_calculate_elements(const nvinfer1::Dims& dims, const std::string& name);

    /// @brief Helper function to count the number of input or output 
    size_t count_number_of_io(const nvinfer1::Dims& dims, const std::string &name)

    /// @brief Helper function to print the memory layout of model.
    void print_strides(const nvinfer1::Dims& dims);

    /// @brief Allocate the CUDA buffers for inference.
    /// This must be called after load_engine() in the constructor. 
    void allocate_buffers();

    /// @brief Calculate volume of dimensions (helper function)
    size_t dims_volume(const nvinfer1::Dims& d) noexcept {
        return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<size_t>());
    }

    /// @brief Prior to inference, checks to make sure that the I/O buffers are
    /// correctly sized for the loaded model.
    /// @param input_bufs Input buffers
    /// @param input_sizes Sizes of input buffers (bytes)
    /// @param output_bufs Output buffers
    /// @param output_sizes Sizes of output buffers (bytes)
    /// @return TRUE if valid, else FALSE. (An error message will print.)
    bool validate_buffers(
        const std::vector<void*>& input_bufs, 
        const std::vector<size_t>& input_sizes,
        const std::vector<void*>& output_bufs,
        const std::vector<size_t>& output_sizes) const ;

    /// @brief Deallocate the CUDA buffers.        
    // /// This should be called in the destructor only.
    void deallocate_buffers();

    /// @brief Deallocates all TensorRT runtime resources. 
    /// This should be called in the destructor only.
    void shutdown_engine();
};

