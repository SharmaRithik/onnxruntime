#!/bin/bash
# =============================================================================
# test-build.sh - Test ONNX Runtime WebGPU backend build
# =============================================================================
# Usage: ./test-build.sh [--full] [--filter PATTERN]
#
# Options:
#   --full             Run all WebGPU tests (takes longer)
#   --filter PATTERN   Run only tests matching pattern (e.g., "MatMul*")
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ORT_ROOT/build-webgpu"
BUILD_TYPE="Release"
FULL_TEST=false
FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            FULL_TEST=true
            shift
            ;;
        --filter)
            FILTER="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--full] [--filter PATTERN]"
            echo ""
            echo "Options:"
            echo "  --full             Run all WebGPU tests (takes longer)"
            echo "  --filter PATTERN   Run only tests matching pattern"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "ONNX Runtime WebGPU Backend Test Script"
echo "=============================================="

# Check if build exists
if [ ! -d "$BUILD_DIR/$BUILD_TYPE" ]; then
    echo "ERROR: Build directory not found: $BUILD_DIR/$BUILD_TYPE"
    echo "Run ./build-webgpu.sh first"
    exit 1
fi

cd "$BUILD_DIR/$BUILD_TYPE"

# Test 1: Check Python import
echo ""
echo "Test 1: Checking Python import..."
echo "----------------------------------------------"
python3 -c "
import sys
sys.path.insert(0, '.')
import onnxruntime as ort
print(f'ONNX Runtime version: {ort.__version__}')
providers = ort.get_available_providers()
print(f'Available providers: {providers}')
if 'WebGpuExecutionProvider' in providers:
    print('✓ WebGpuExecutionProvider is available')
else:
    print('✗ WebGpuExecutionProvider NOT available')
    sys.exit(1)
"

# Test 2: Quick inference test
echo ""
echo "Test 2: Quick inference test..."
echo "----------------------------------------------"
# Note: WebGPU may segfault on cleanup, but inference should work
python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import onnxruntime as ort
from onnx import helper, TensorProto
import onnx
import tempfile
import os

# Create simple MatMul model
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [4, 4])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [4, 4])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [4, 4])
node = helper.make_node('MatMul', ['A', 'B'], ['C'])
graph = helper.make_graph([node], 'test', [A, B], [C])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    model_path = f.name
    onnx.save(model, f.name)

# Run with WebGPU
sess = ort.InferenceSession(model_path, providers=['WebGpuExecutionProvider'])

a = np.random.randn(4, 4).astype(np.float32)
b = np.random.randn(4, 4).astype(np.float32)

result = sess.run(None, {'A': a, 'B': b})[0]
expected = np.matmul(a, b)

max_error = np.max(np.abs(result - expected))
print(f'Max error vs NumPy: {max_error:.6e}')

if np.allclose(result, expected, rtol=1e-4, atol=1e-4):
    print('✓ Inference test passed')
    # Write success marker before cleanup (which may segfault)
    with open('/tmp/webgpu_test_passed', 'w') as f:
        f.write('passed')
else:
    print('✗ Inference test FAILED')

os.unlink(model_path)
" 2>&1 || true

# Check if test passed (before potential segfault on cleanup)
if [ -f /tmp/webgpu_test_passed ]; then
    rm /tmp/webgpu_test_passed
    echo "(Note: segfault on cleanup is a known Dawn issue, test passed)"
else
    echo "✗ Inference test FAILED"
    exit 1
fi

# Test 3: Run C++ unit tests (optional)
if [ "$FULL_TEST" = true ]; then
    echo ""
    echo "Test 3: Running C++ unit tests..."
    echo "----------------------------------------------"

    TEST_CMD="./onnxruntime_test_all"
    if [ -n "$FILTER" ]; then
        TEST_CMD="$TEST_CMD --gtest_filter=$FILTER"
    else
        # Default: run WebGPU-specific tests
        TEST_CMD="$TEST_CMD --gtest_filter=*WebGpu*:*MatMul*"
    fi

    echo "Running: $TEST_CMD"
    $TEST_CMD
fi

echo ""
echo "=============================================="
echo "All tests passed!"
echo "=============================================="
