#!/usr/bin/env python3
"""
Test script to generate WebGPU WGSL shader code for MatMul FP32.
This script creates a simple MatMul model, runs it with WebGPU,
and captures the generated shader code via verbose logging.
"""

import sys
import numpy as np
import onnx
from onnx import helper, TensorProto
import tempfile
import os

# Set environment for verbose logging before importing onnxruntime
os.environ["ORT_LOG_LEVEL"] = "VERBOSE"

# Insert build directory at the front of path to use built onnxruntime
sys.path.insert(0, "/home/riksharm/onnxruntime/build-webgpu/Release")
import onnxruntime as ort

def create_matmul_model(m, k, n, dtype=TensorProto.FLOAT):
    """Create a simple ONNX model with a single MatMul operation."""

    # Create input tensors
    A = helper.make_tensor_value_info('A', dtype, [m, k])
    B = helper.make_tensor_value_info('B', dtype, [k, n])

    # Create output tensor
    C = helper.make_tensor_value_info('C', dtype, [m, n])

    # Create MatMul node
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['A', 'B'],
        outputs=['C'],
        name='matmul_op'
    )

    # Create the graph
    graph = helper.make_graph(
        [matmul_node],
        'matmul_graph',
        [A, B],
        [C]
    )

    # Create the model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8

    return model

def main():
    print("=" * 60)
    print("WebGPU MatMul FP32 1024x1024 Shader Generation Test")
    print("=" * 60)

    # Set verbose logging level (0 = VERBOSE)
    ort.set_default_logger_severity(0)

    # Matrix dimensions: 1024x1024 matmul
    M, K, N = 1024, 1024, 1024

    print(f"\nCreating MatMul model with dimensions: M={M}, K={K}, N={N}")

    # Create the model
    model = create_matmul_model(M, K, N)

    # Save model to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        model_path = f.name
        onnx.save(model, model_path)
        print(f"Model saved to: {model_path}")

    try:
        # Check available providers
        providers = ort.get_available_providers()
        print(f"\nAvailable providers: {providers}")

        if 'WebGpuExecutionProvider' not in providers:
            print("\nERROR: WebGpuExecutionProvider not available!")
            print("Make sure you built ONNX Runtime with --use_webgpu")
            return

        # Create session options with verbose logging
        session_options = ort.SessionOptions()
        session_options.log_severity_level = 0  # VERBOSE
        session_options.log_verbosity_level = 1000  # Maximum verbosity

        # Use NVIDIA GPU (device_id=1) instead of Intel Arc (device_id=0)
        device_id = 1  # NVIDIA GPU
        print(f"\nCreating inference session with WebGPU provider on device_id={device_id} (NVIDIA)...")
        print("(Shader code should be printed below in VERBOSE logs)\n")
        print("-" * 60)

        # Create inference session with WebGPU on NVIDIA
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=[('WebGpuExecutionProvider', {'device_id': device_id})]
        )

        # Create input data
        np.random.seed(42)
        A = np.random.randn(M, K).astype(np.float32)
        B = np.random.randn(K, N).astype(np.float32)

        print("-" * 60)
        print("\nRunning inference...")
        print("-" * 60)

        # Run inference
        outputs = session.run(None, {'A': A, 'B': B})

        print("-" * 60)

        # Verify output
        C_ort = outputs[0]
        C_numpy = np.matmul(A, B)

        print(f"\nOutput shape: {C_ort.shape}")
        print(f"Max absolute error vs NumPy: {np.max(np.abs(C_ort - C_numpy)):.6f}")
        print(f"Results match: {np.allclose(C_ort, C_numpy, rtol=1e-4, atol=1e-4)}")

    finally:
        # Clean up
        os.unlink(model_path)

    print("\n" + "=" * 60)
    print("Done! Check the output above for shader code.")
    print("=" * 60)

if __name__ == '__main__':
    main()
