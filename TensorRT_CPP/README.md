# Tensor RT C++ Library

`TrtInferenceEngine` (from `TRT_inference_engine.hpp`) is a wrapper around the TensorRT inference runtime.
This does not require OpenCV or any other dependencies aside from Nvidia TensorRT and CudaRT libraries.

In the future, this can be moved into a global library of some form...

### How to use

(1) Include the `TRT_inference_engine.hpp` header.

(2) Initialize an instance of `TrtInferenceEngine(const std::string& engine_path)`.
This loads a `.engine` model from `engine_path` and allocates appropriate resources.

(3) To perform inference, use the following function:
```
        /// @brief Performs inference using the loaded model. The input/output must be sized appropriately.
        /// Recommended to pre-allocate them using get_input_size_bytes() and get_output_size_bytes()
        /// prior to calling this function. 
        /// @param input_buf The pointer to the input buffer.
        /// @param size_in_param The size (bytes) of the input buffer. 
        /// Should == this->get_input_size_bytes()
        /// @param output_buf The pointer to the output buffer.
        /// @param size_out_param The size (bytes) of the output buffer. 
        /// Should == this->get_output_size_bytes()
        /// @return Output size, or 0 on failure.
        size_t TrtInferenceEngine::infer_b(const float* input_buf, size_t size_in_param, float* output_buf, size_t size_out_param)
```
This is a synchronous (blocking) call. Make sure all buffers are appropriately allocated.

Async version will be implemented in the future, with the following functions:
```
        /// @brief Performs inference using the loaded model. The input/output must be sized appropriately.
        /// Recommended to pre-allocate them using get_input_size_bytes() and get_output_size_bytes()
        /// prior to calling this function. 
        /// This function enqueues the inference result using Cuda stream so that you can asynchronously
        /// execute other functions in the meantime.
        /// @param input_buf The pointer to the input buffer.
        /// @param size_in_param The size (bytes) of the input buffer. 
        /// Should == this->get_input_size_bytes()
        /// @return TRUE if successfully enqueued.
        bool TrtInferenceEngine::infer_async(const float* input_buf, size_t size_in_param)

        /// @brief Performs inference using the loaded model. The input/output must be sized appropriately.
        /// Recommended to pre-allocate them using get_input_size_bytes() and get_output_size_bytes()
        /// prior to calling this function. 
        /// This function retrieves an async result from the Cuda stream (if available).
        /// @param output_buf The pointer to the output buffer.
        /// @param size_out_param The size (bytes) of the output buffer. 
        /// Should == this->get_output_size_bytes()
        /// @return Output size, or 0 on failure.
        size_t TrtInferenceEngine::retrieve_infer_result_async(float* output_buf, size_t size_out_param)
```

