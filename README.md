# Model Testing and Quantization

## Overview

This repository contains scripts and notebooks for testing ONNX models and engine files, specifically focusing on models created using TensorRT. The main components are:

1. **GPU_Test.py** - A Python script to test ONNX models with specific input sizes.
2. **inference_test.py** - A Python script for testing ONNX models that have been created using TensorRT and performing inference with them, present in export folder.

## Components

### 1. `GPU_Test.py`

- **Purpose**: This script is used to run and test ONNX models.
- **Supported Input Size**: The script supports specific input sizes defined within the code.

### 2. `inference_test.py`

- **Purpose**: This Jupyter Notebook is designed for testing ONNX models that have been quantized and optimized using TensorRT. It includes functions for plotting inference results.
- **File Types**:
  - **ONNX Model**: `old.onnx` - This is an older ONNX model.
  - **Quantized ONNX Model**: `yolov8s_quantized.onnx` - This ONNX model has been quantized using TensorRT.
  - **Engine File**: `yolov8s_quantized.engine` - The main engine model file, which is an INT8 quantized TensorRT engine.

## Usage

### Testing with `GPU_Test.py`

1. Ensure that you have the necessary dependencies installed.
2. Adjust the input size settings as needed in the script.
3. Run the script with the ONNX model you wish to test.

### Testing with `inference_test.py`

1. Ensure that you have the necessary dependencies installed.
2. Load the ONNX models and engine files as specified.
3. Run the script to perform inference and plot the results.

## Dependencies

- ONNX Runtime
- TensorRT
- NumPy
- Matplotlib
- Other Python packages (as specified in the scripts)

## Notes

- Ensure that the ONNX models are compatible with the input sizes and formats expected by the scripts and notebook.
- The quantization process involves converting models to INT8 precision using TensorRT, which is optimized for performance on NVIDIA GPUs.



