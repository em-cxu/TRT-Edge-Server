#pragma once

#include <iostream>
#include <fstream>
#include <vector>

namespace TRT::YOLO
{

    // YOLO object detector usually consists of 640 x 640 x 3 inputs (RGB floats)
    constexpr int MODEL_INPUT_WIDTH = 640;
    constexpr int MODEL_INPUT_HEIGHT = 640;

    constexpr int MODEL_NUM_INPUTS = 1;
    constexpr int MODEL_NUM_OUTPUTS = 4;

    constexpr int OUTPUT_INDEX_NUM_DETS = 0;
    constexpr int OUTPUT_INDEX_BBOXES = 1;
    constexpr int OUTPUT_INDEX_SCORES = 2;
    constexpr int OUTPUT_INDEX_LABELS = 3;

    constexpr int CONFIDENCE_SCORE_THRESHOLD = 0.25f;

    /// @brief A bounding box, consisting of a rectangle x, y, and height.
    typedef struct bounding_box
    {
        float x, y; // Corner 1.
        float width, height; // The width [x] and height [y] of the box.
    } bounding_box_t;

    /// @brief Container for detected object information.
    typedef struct detected_object_info 
    {
        bounding_box_t rect;
        int class_id;
        float confidence;
    } detected_object_info_t;

}
