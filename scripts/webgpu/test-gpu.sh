#!/bin/bash
# =============================================================================
# test-gpu.sh - Detect and display available GPUs for WebGPU
# =============================================================================
# Usage: ./test-gpu.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ORT_ROOT/build-webgpu"
BUILD_TYPE="Release"

echo "=============================================="
echo "GPU Detection for WebGPU"
echo "=============================================="

# Check Vulkan devices (WebGPU uses Vulkan backend on Linux)
echo ""
echo "1. Vulkan Devices (via vulkaninfo):"
echo "----------------------------------------------"
if command -v vulkaninfo &> /dev/null; then
    vulkaninfo --summary 2>/dev/null | grep -E "GPU|deviceName|driverVersion|deviceType" | head -20 || echo "No Vulkan devices found"
else
    echo "vulkaninfo not installed. Install with: sudo apt install vulkan-tools"
fi

# Check NVIDIA driver
echo ""
echo "2. NVIDIA Driver Status:"
echo "----------------------------------------------"
if [ -f /proc/driver/nvidia/version ]; then
    cat /proc/driver/nvidia/version | head -1
    echo "NVIDIA GPU detected and driver loaded"
else
    echo "No NVIDIA driver loaded"
fi

# Check Intel GPU
echo ""
echo "3. Intel GPU Status:"
echo "----------------------------------------------"
if lspci | grep -i "VGA.*Intel" &> /dev/null; then
    lspci | grep -i "VGA.*Intel"
else
    echo "No Intel GPU detected"
fi

# Check AMD GPU
echo ""
echo "4. AMD GPU Status:"
echo "----------------------------------------------"
if lspci | grep -i "VGA.*AMD\|VGA.*ATI" &> /dev/null; then
    lspci | grep -i "VGA.*AMD\|VGA.*ATI"
else
    echo "No AMD GPU detected"
fi

# Check all discrete GPUs
echo ""
echo "5. All Graphics Devices (lspci):"
echo "----------------------------------------------"
lspci | grep -iE "VGA|3D|Display" || echo "No graphics devices found"

# Test WebGPU device selection
echo ""
echo "6. WebGPU Device Selection Test:"
echo "----------------------------------------------"

if [ -d "$BUILD_DIR/$BUILD_TYPE" ]; then
    cd "$BUILD_DIR/$BUILD_TYPE"
    python3 << 'EOF'
import sys
sys.path.insert(0, '.')
import os
os.environ['ORT_LOG_LEVEL'] = 'VERBOSE'

import onnxruntime as ort
from onnx import helper, TensorProto
import onnx
import tempfile
import re

# Create minimal model
A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [2, 2])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [2, 2])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [2, 2])
node = helper.make_node('MatMul', ['A', 'B'], ['C'])
graph = helper.make_graph([node], 'test', [A, B], [C])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])

with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
    onnx.save(model, f.name)
    model_path = f.name

import subprocess
import io

# Capture verbose output to detect GPU
result = subprocess.run(
    [sys.executable, '-c', f'''
import sys
sys.path.insert(0, ".")
import os
os.environ["ORT_LOG_LEVEL"] = "VERBOSE"
import onnxruntime as ort
sess = ort.InferenceSession("{model_path}", providers=["WebGpuExecutionProvider"])
'''],
    capture_output=True,
    text=True,
    cwd='.'
)

# Parse output for device info
output = result.stderr + result.stdout
print("Detected from WebGPU initialization:")

# Look for device info
device_id_match = re.search(r'Device ID:\s*(\d+)', output)
power_pref_match = re.search(r'power preference:\s*(\d+)', output)
context_match = re.search(r'Context is created', output)

if device_id_match:
    print(f"  Device ID: {device_id_match.group(1)}")
if power_pref_match:
    pref = int(power_pref_match.group(1))
    pref_name = {0: 'undefined', 1: 'low-power', 2: 'high-performance'}.get(pref, str(pref))
    print(f"  Power Preference: {pref_name}")
if context_match:
    print("  ✓ WebGPU context created successfully")

# Subgroup info (indicates GPU architecture)
if 'subgroup' in output.lower():
    print("  Subgroups: enabled")

os.unlink(model_path)
EOF
else
    echo "Build not found. Run ./build-webgpu.sh first"
fi

echo ""
echo "=============================================="
echo "Summary"
echo "=============================================="
echo ""
echo "Note: WebGPU on Linux uses the Vulkan backend (via Dawn)."
echo "The GPU selection is done by Dawn based on power preference."
echo ""
echo "GPU ordering in Vulkan (device_id):"
vulkaninfo 2>/dev/null | grep -E "GPU id" | head -5 || echo "  (run vulkaninfo for details)"
echo ""
