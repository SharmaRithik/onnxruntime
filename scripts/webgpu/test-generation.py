#!/usr/bin/env python3
"""
=============================================================================
test-generation.py - Generate and display WebGPU WGSL shaders for operators
=============================================================================

Usage:
    ./test-generation.py                    # Default: MatMul 1024x1024
    ./test-generation.py --op matmul --m 512 --k 512 --n 512
    ./test-generation.py --op matmul --m 64 --k 64 --n 64   # Small (different kernel)
    ./test-generation.py --save shader.wgsl  # Save shader to file
    ./test-generation.py --list              # List available operators

=============================================================================
"""

import sys
import os
import argparse
import re
import tempfile
import subprocess

# Add build directory to path
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'build-webgpu', 'Release')
sys.path.insert(0, BUILD_DIR)

import numpy as np
import onnx
from onnx import helper, TensorProto


def create_matmul_model(m, k, n, dtype=TensorProto.FLOAT):
    """Create a MatMul model."""
    A = helper.make_tensor_value_info('A', dtype, [m, k])
    B = helper.make_tensor_value_info('B', dtype, [k, n])
    C = helper.make_tensor_value_info('C', dtype, [m, n])

    node = helper.make_node('MatMul', ['A', 'B'], ['C'])
    graph = helper.make_graph([node], 'matmul_test', [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = 8
    np_dtype = np.float16 if dtype == TensorProto.FLOAT16 else np.float32
    return model, {'A': (m, k), 'B': (k, n)}, np_dtype


def create_add_model(shape):
    """Create an Add model."""
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, shape)
    B = helper.make_tensor_value_info('B', TensorProto.FLOAT, shape)
    C = helper.make_tensor_value_info('C', TensorProto.FLOAT, shape)

    node = helper.make_node('Add', ['A', 'B'], ['C'])
    graph = helper.make_graph([node], 'add_test', [A, B], [C])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    return model, {'A': shape, 'B': shape}, np.float32


def create_softmax_model(shape, axis=-1):
    """Create a Softmax model."""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, shape)
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, shape)

    node = helper.make_node('Softmax', ['X'], ['Y'], axis=axis)
    graph = helper.make_graph([node], 'softmax_test', [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    return model, {'X': shape}, np.float32


def create_conv_model(batch, in_c, out_c, h, w, kernel_size=3):
    """Create a Conv model."""
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [batch, in_c, h, w])
    W = helper.make_tensor_value_info('W', TensorProto.FLOAT, [out_c, in_c, kernel_size, kernel_size])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, None)  # Let ONNX infer

    node = helper.make_node('Conv', ['X', 'W'], ['Y'], kernel_shape=[kernel_size, kernel_size], pads=[1, 1, 1, 1])
    graph = helper.make_graph([node], 'conv_test', [X, W], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    return model, {'X': (batch, in_c, h, w), 'W': (out_c, in_c, kernel_size, kernel_size)}, np.float32


def extract_shader(output):
    """Extract WGSL shader code from verbose output."""
    shaders = []

    # Pattern for shader blocks
    pattern = r'=== WebGPU Shader code \[([^\]]+)\] Start ===(.*?)=== WebGPU Shader code \[\1\] End ==='
    matches = re.findall(pattern, output, re.DOTALL)

    for name, code in matches:
        # Clean up the code
        lines = code.strip().split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove timestamp prefixes
            line = re.sub(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \[.*?\] ', '', line)
            cleaned_lines.append(line)
        shaders.append((name, '\n'.join(cleaned_lines)))

    return shaders


def run_and_capture_shader(model_path, input_shapes, np_dtype=np.float32):
    """Run inference and capture shader output."""
    # Build inputs string
    dtype_str = 'np.float16' if np_dtype == np.float16 else 'np.float32'
    inputs_code = ""
    for name, shape in input_shapes.items():
        inputs_code += f'inputs["{name}"] = np.random.randn{shape}.astype({dtype_str})\n'

    # Create a script to run with verbose logging
    script = f'''
import sys
import os
# Set logging BEFORE importing onnxruntime
os.environ["ORT_LOG_LEVEL"] = "VERBOSE"

sys.path.insert(0, "{BUILD_DIR}")
import numpy as np
import onnxruntime as ort

# Also set via API
ort.set_default_logger_severity(0)

sess = ort.InferenceSession("{model_path}", providers=["WebGpuExecutionProvider"])

inputs = {{}}
{inputs_code}
outputs = sess.run(None, inputs)
print("INFERENCE_COMPLETE", file=sys.stderr)
'''

    result = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        cwd=BUILD_DIR
    )

    return result.stderr + result.stdout


def main():
    parser = argparse.ArgumentParser(description='Generate WebGPU WGSL shaders')
    parser.add_argument('--op', choices=['matmul', 'add', 'softmax', 'conv'], default='matmul',
                        help='Operator to test (default: matmul)')
    parser.add_argument('--m', type=int, default=1024, help='M dimension for MatMul')
    parser.add_argument('--k', type=int, default=1024, help='K dimension for MatMul')
    parser.add_argument('--n', type=int, default=1024, help='N dimension for MatMul')
    parser.add_argument('--shape', type=int, nargs='+', default=[1024, 1024],
                        help='Shape for Add/Softmax')
    parser.add_argument('--dtype', choices=['f32', 'f16'], default='f32',
                        help='Data type: f32 (float32) or f16 (float16)')
    parser.add_argument('--save', type=str, help='Save shader to file')
    parser.add_argument('--list', action='store_true', help='List available operators')
    parser.add_argument('--quiet', action='store_true', help='Only show shader code')

    args = parser.parse_args()

    if args.list:
        print("Available operators:")
        print("  matmul  - Matrix multiplication (--m, --k, --n)")
        print("  add     - Element-wise addition (--shape)")
        print("  softmax - Softmax activation (--shape)")
        print("  conv    - 2D Convolution")
        return

    if not args.quiet:
        print("=" * 60)
        print("WebGPU WGSL Shader Generation")
        print("=" * 60)

    # Map dtype string to TensorProto type
    from onnx import TensorProto
    dtype_map = {
        'f32': TensorProto.FLOAT,
        'f16': TensorProto.FLOAT16,
    }
    dtype = dtype_map.get(args.dtype, TensorProto.FLOAT)

    # Create model based on operator
    np_dtype = np.float32  # default
    if args.op == 'matmul':
        if not args.quiet:
            print(f"Operator: MatMul ({args.m} x {args.k}) @ ({args.k} x {args.n}) dtype={args.dtype}")
        model, input_shapes, np_dtype = create_matmul_model(args.m, args.k, args.n, dtype=dtype)
    elif args.op == 'add':
        if not args.quiet:
            print(f"Operator: Add (shape: {args.shape})")
        model, input_shapes, np_dtype = create_add_model(args.shape)
    elif args.op == 'softmax':
        if not args.quiet:
            print(f"Operator: Softmax (shape: {args.shape})")
        model, input_shapes, np_dtype = create_softmax_model(args.shape)
    elif args.op == 'conv':
        if not args.quiet:
            print("Operator: Conv2D (1, 64, 64, 224, 224)")
        model, input_shapes, np_dtype = create_conv_model(1, 64, 64, 224, 224)

    # Save model to temp file
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        onnx.save(model, f.name)
        model_path = f.name

    try:
        if not args.quiet:
            print("Running inference to capture shader...")
            print("-" * 60)

        # Run and capture output
        output = run_and_capture_shader(model_path, input_shapes, np_dtype)

        # Extract shaders
        shaders = extract_shader(output)

        if not shaders:
            print("No shaders captured. Raw output:")
            print(output[:2000])
            return

        for name, code in shaders:
            if not args.quiet:
                print(f"\n{'=' * 60}")
                print(f"Shader: {name}")
                print("=" * 60)
            print(code)

            if args.save:
                save_path = args.save if len(shaders) == 1 else f"{name}_{args.save}"
                with open(save_path, 'w') as f:
                    f.write(f"// Shader: {name}\n")
                    f.write(f"// Generated for: {args.op}\n\n")
                    f.write(code)
                if not args.quiet:
                    print(f"\nSaved to: {save_path}")

        if not args.quiet:
            print("\n" + "=" * 60)
            print("Done!")
            print("=" * 60)

    finally:
        os.unlink(model_path)


if __name__ == '__main__':
    main()
