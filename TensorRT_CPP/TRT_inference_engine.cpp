#include "TRT_inference_engine.hpp"

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
bool TrtInferenceEngine::infer_b(const std::vector<void*> &input_buf, std::vector<size_t> &size_in_param, 
    std::vector<void*> &output_buf, std::vector<size_t> &size_out_param)
{
    if (input_size != this->get_input_size_bytes() || output_size != this->get_output_size_bytes()) 
    {
        std::cerr << "[TRT_ENGINE] Buffer size mismatch\n"
                << "\tExpected input: " << this->get_input_size_bytes() << " bytes\n"
                << "\tReceived input: " << input_size << " bytes\n"
                << "\tExpected output: " << this->get_output_size_bytes() << " bytes\n"
                << "\tReceived output: " << output_size << " bytes" << std::endl;
        return false;
    }

    if (input_buf.size() != this->input_cuda_buffers_.size() 
        || output_buf.size() != this->output_cuda_buffers_.size()) 
    {
        std::cerr << "[TRT_ENGINE] Buffer size mismatch!\n"
                << "\tInput buffer: [Actual]" << input_buf.size() << "\t[Expected] " << this->input_cuda_buffers_.size() << "\n"
                << "\tOutput buffer: [Actual]" << output_buf.size() << "\t[Expected] " << this->output_cuda_buffers_.size() << std::endl;
        return false;
    }

    // Copy input to GPU.
    for (int i = 0; i < this->input_cuda_buffers_.size(); i++)
    {
        cudaMemcpy(this->input_cuda_buffers_[i], input_buf[i], size_in_param[i], cudaMemcpyHostToDevice);
    }

    // Execute inference.
    // this->bindings_ should already have been populated with the appropriate members
    // of this->input_cuda_buffers and this->output_cuda_buffers, so there
    // should not be any need to update them prior to executing this function.
    this->context_->executeV2(this->bindings_.data());

    for (int i = 0; i < this->output_cuda_buffers_.size(); i++)
    {
        // Copy results back from GPU.
        cudaMemcpy(output_buf[i], this->output_cuda_buffers_[i], size_out_param[i], cudaMemcpyDeviceToHost);
    }
    
    return true;
}

/// @brief Performs inference using the loaded model. The input/output must be sized appropriately.
/// Recommended to pre-allocate them using get_input_size_bytes() and get_output_size_bytes()
/// prior to calling this function. 
/// This function enqueues the inference result using Cuda stream so that you can asynchronously
/// execute other functions in the meantime.
/// @param input_buf The pointer to the input buffer.
/// @param size_in_param The size (bytes) of the input buffer. 
/// Should == this->get_input_size_bytes()
/// @return TRUE if successfully enqueued.
bool TrtInferenceEngine::infer_async(const std::vector<void*> &input_buf, std::vector<size_t> &size_in_param)
{
    return false; // Not yet implemented!
}

/// @brief Performs inference using the loaded model. The input/output must be sized appropriately.
/// Recommended to pre-allocate them using get_input_size_bytes() and get_output_size_bytes()
/// prior to calling this function. 
/// This function retrieves an async result from the Cuda stream (if available).
/// @param output_buf The pointer to the output buffer.
/// @param size_out_param The size (bytes) of the output buffer. 
/// Should == this->get_output_size_bytes()
/// @return Output size, or 0 on failure.
size_t TrtInferenceEngine::retrieve_infer_result_async(const std::vector<void*> &output_buf, 
    std::vector<size_t> &size_out_param)
{
    return false; // Not yet implemented!
}


/// --- The functions below this line are internal use only ---


void TrtInferenceEngine::load_engine(const std::string& engine_path)
{
    // Load .engine file
    std::ifstream engine_file(engine_path, std::ios::binary);
    engine_file.seekg(0, std::ios::end);
    size_t size = engine_file.tellg();
    engine_file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    engine_file.read(engine_data.data(), size);

    // Instantiate the model
    this->runtime_ = std::unique_ptr<nvinfer1::IRuntime>(
        nvinfer1::createInferRuntime(this->logger_));
    this->engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
        this->runtime_->deserializeCudaEngine(engine_data.data(), size));
    this->context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
        this->engine_->createExecutionContext());

    std::cout << "[TRT_ENGINE] Loaded TensorRT model from path: " << engine_path << std::endl;
}

void TrtInferenceEngine::calculate_model_parameters() 
{
    if (!this->engine_) 
    {
        throw std::runtime_error("Engine not initialized");
    }

    const int num_bindings = this->engine_->getNbBindings();
    std::cout << "[TRT_ENGINE] Model Parameters:\n";
    std::cout << "  Number of bindings: " << num_bindings << "\n";

    // Clear existing data
    this->input_dims_.clear();
    this->output_dims_.clear();
    this->input_names_.clear();
    this->output_names_.clear();
    this->trt_input_element_counts_.clear();
    this->trt_output_element_counts_.clear();

    // Update counts
    this->num_inputs_ = static_cast<int>(this->input_dims_.size());
    this->num_outputs_ = static_cast<int>(this->output_dims_.size());

    // Process all bindings
    for (int i = 0; i < num_bindings; i++) {
        const char* name = this->engine_->getBindingName(i);
        const auto dims = this->engine_->getBindingDimensions(i);
        const bool is_input = this->engine_->bindingIsInput(i);

        std::cout << (is_input ? "  Input" : "  Output") << " [" << i << "]: " 
                << (name ? name : "unnamed") << "\n";
        std::cout << "    Dimensions: [";
        
        // Validate dimensions
        for (int j = 0; j < dims.nbDims; j++) {
            if (dims.d[j] <= 0) {
                throw std::runtime_error("[TRT_ENGINE] Invalid dimension value in binding " + 
                                    std::to_string(i) + " dimension " + 
                                    std::to_string(j));
            }
            std::cout << dims.d[j];
            if (j < dims.nbDims - 1) std::cout << " x ";
        }
        std::cout << "]\n";

        // Store dimensions
        if (is_input) {
            this->input_dims_.push_back(dims);
            this->input_names_.push_back(std::string(name));
            std::vector<int> shape;
            for (int j = 0; j < dims.nbDims; ++j) {
                shape.push_back(dims.d[j]);
            }
            this->input_shapes_.push_back(shape);
        } else {
            this->output_dims_.push_back(dims);
            this->output_names_.push_back(std::string(name));            
            std::vector<int> shape;
            for (int j = 0; j < dims.nbDims; ++j) {
                shape.push_back(dims.d[j]);
            }
            this->output_shapes_.push_back(shape);
        }
    }

    // Calculate element counts
    for (const auto& dims : this->input_dims_) {
        this->trt_input_element_counts_.push_back(this->validate_and_calculate_elements(dims, "input"));
    }
    for (const auto& dims : this->output_dims_) {
        this->trt_output_element_counts_.push_back(this->validate_and_calculate_elements(dims, "output"));
    }

    // Print summary
    std::cout << "\tInput Count: " << this->num_inputs_ << "\n";
    std::cout << "\tOutput Count: " << this->num_outputs_ << "\n";
    
    constexpr float KB_TO_BYTES = 1024.0f;

    for (int i = 0; i < this->num_inputs_; i++) 
    {
        std::cout << "\tInput " << i << "\tElements: " << this->trt_input_element_counts_[i] 
                << " (" << (this->trt_input_element_counts_[i] * sizeof(float) / KB_TO_BYTES) << " KB)\n";
    }
    for (int i = 0; i < this->num_outputs_; i++) 
    {
        std::cout << "\tOutput " << i << "\tElements: " << this->trt_output_element_counts_[i] 
                << " (" << (this->trt_output_element_counts_[i] * sizeof(float) / KB_TO_BYTES) << " KB)\n";
    }
    std::cout << "\n";
}

size_t TrtInferenceEngine::validate_and_calculate_elements(const nvinfer1::Dims& dims, const std::string& name) 
{
    if (dims.nbDims == 0) {
        throw std::runtime_error("No " + name + " dimensions detected");
    }

    size_t elements = 1;
    for (int j = 0; j < dims.nbDims; ++j) {
        if (dims.d[j] <= 0) {
            throw std::runtime_error("Invalid " + name + " dimension " + 
                                std::to_string(j) + ": " + 
                                std::to_string(dims.d[j]));
        }
        elements *= dims.d[j];
    }
    return elements;
}

void TrtInferenceEngine::print_strides(const nvinfer1::Dims& dims) 
{
    size_t stride = 1;
    std::cout << "[";
    for (int j = dims.nbDims - 1; j >= 0; --j) {
        std::cout << stride;
        if (j > 0) std::cout << ", ";
        stride *= dims.d[j];
    }
    std::cout << "]";
}

void TrtInferenceEngine::allocate_buffers() 
{
    // Check if buffers are already allocated
    if (!this->input_cuda_buffers_.empty() || !this->output_cuda_buffers_.empty()) {
        std::cerr << "[TRT_ENGINE] Error: CUDA buffers already allocated!\n"
                  << "\tEnsure proper cleanup with deallocate_buffers() first." << std::endl;
        return;
    }

    // Validate model parameters
    if (this->num_inputs_ < 1 || this->num_outputs_ < 1) {
        std::cerr << "[TRT_ENGINE] Error: Invalid model I/O counts ("
                  << this->num_inputs_ << " inputs, " << this->num_outputs_ << " outputs)" << std::endl;
        return;
    }

    // Get required byte sizes
    const auto input_sizes = this->get_input_size_bytes();
    const auto output_sizes = this->get_output_size_bytes();

    // Allocate input buffers
    this->input_cuda_buffers_.resize(this->num_inputs_);
    for (int i = 0; i < this->num_inputs_; ++i) {
        if (cudaMalloc(&this->input_cuda_buffers_[i], input_sizes[i]) != cudaSuccess) {
            std::cerr << "[TRT_ENGINE] Failed to allocate input buffer " << i 
                      << " (" << input_sizes[i] << " bytes)" << std::endl;
            deallocate_buffers(); // Clean up any partial allocations
            return;
        }
    }

    // Allocate output buffers
    this->output_cuda_buffers_.resize(this->num_outputs_);
    for (int i = 0; i < this->num_outputs_; ++i) {
        if (cudaMalloc(&this->output_cuda_buffers_[i], output_sizes[i]) != cudaSuccess) {
            std::cerr << "[TRT_ENGINE] Failed to allocate output buffer " << i 
                      << " (" << output_sizes[i] << " bytes)" << std::endl;
            deallocate_buffers(); // Clean up any partial allocations
            return;
        }
    }

    // Update the references to the memory locations used during inference
    this->bindings_.clear();
    this->bindings_.insert(this->bindings_.end(), this->input_cuda_buffers_.begin(), this->input_cuda_buffers_.end());
    this->bindings_.insert(this->bindings_.end(), this->output_cuda_buffers_.begin(), this->output_cuda_buffers_.end());

    std::cout << "[TRT_ENGINE] Successfully allocated " 
              << this->num_inputs_ << " input and " << this->num_outputs_ << " output buffers\n";
    std::cout << "\tTotal size of bindings is " 
              << this->bindings_.size() << " elements." << std::endl;
}

bool TrtInferenceEngine::validate_buffers(
    const std::vector<void*>& input_bufs, 
    const std::vector<size_t>& input_sizes,
    const std::vector<void*>& output_bufs,
    const std::vector<size_t>& output_sizes) const 
{
    // Check counts match
    if (input_bufs.size() != this->num_inputs_ || output_bufs.size() != this->num_outputs_) {
        std::cerr << "[TRT_ENGINE] Buffer count mismatch\n"
                  << "Expected " << this->num_inputs_ << " inputs, got " << input_bufs.size() << "\n"
                  << "Expected " << this->num_outputs_ << " outputs, got " << output_bufs.size() << std::endl;
        return false;
    }

    // Check sizes match
    const auto expected_input_sizes = this->get_input_size_bytes();
    const auto expected_output_sizes = this->get_output_size_bytes();
    
    bool valid = true;
    
    // Validate input buffers
    for (int i = 0; i < this->num_inputs_; ++i) {
        if (!input_bufs[i]) {
            std::cerr << "[TRT_ENGINE] Null input buffer at index " << i << std::endl;
            valid = false;
        }
        if (input_sizes[i] != expected_input_sizes[i]) {
            std::cerr << "[TRT_ENGINE] Input " << i << " size mismatch\n"
                      << "Expected " << expected_input_sizes[i] << " bytes, got " 
                      << input_sizes[i] << std::endl;
            valid = false;
        }
    }

    // Validate output buffers
    for (int i = 0; i < this->num_outputs_; ++i) {
        if (!output_bufs[i]) {
            std::cerr << "[TRT_ENGINE] Null output buffer at index " << i << std::endl;
            valid = false;
        }
        if (output_sizes[i] != expected_output_sizes[i]) {
            std::cerr << "[TRT_ENGINE] Output " << i << " size mismatch\n"
                      << "Expected " << expected_output_sizes[i] << " bytes, got " 
                      << output_sizes[i] << std::endl;
            valid = false;
        }
    }

    return valid;
}

void TrtInferenceEngine::deallocate_buffers() 
{
    for (void* buf : input_cuda_buffers_) 
    {
        if (buf) {
            cudaFree(buf);
            buf = nullptr;
        }
    }
    this->input_cuda_buffers_.clear();

    for (void* buf : output_cuda_buffers_) 
    {
        if (buf) {
            cudaFree(buf);
            buf = nullptr;
        }
    }
    this->output_cuda_buffers_.clear();
    this->bindings_.clear();
    std::cout << "[TRT_ENGINE] Freed all Cuda allocated memory..." << std::endl;
}

void TrtInferenceEngine::shutdown_engine()
{
    this->context_.reset(); 
    this->engine_.reset();  
    this->runtime_.reset(); 
    std::cout << "[TRT_ENGINE] Deallocated TensorRT resources..." << std::endl;
}
