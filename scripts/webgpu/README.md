# WebGPU Shader Generation Scripts

Build and test ONNX Runtime WebGPU backend, inspect generated WGSL shaders.

## Quick Start

```bash
./build-webgpu.sh        # Build
./test-build.sh          # Test
./test-gpu.sh            # Check GPUs
python3 test-generation.py  # Generate shader
```

---

## Prerequisites

- Python 3.8+
- CMake 3.26+
- Vulkan SDK (for Linux WebGPU backend)
- NumPy, ONNX (`pip install numpy onnx`)

## Scripts

### 1. `build-webgpu.sh` - Build WebGPU Backend

```bash
# Basic build (Release mode)
./build-webgpu.sh

# Clean build
./build-webgpu.sh --clean

# Debug build
./build-webgpu.sh --debug
```

Build output: `build-webgpu/Release/`

### 2. `test-build.sh` - Test the Build

```bash
# Quick test (Python import + basic inference)
./test-build.sh

# Full C++ unit tests
./test-build.sh --full

# Run specific tests
./test-build.sh --full --filter "MatMul*"
```

### 3. `test-gpu.sh` - Detect GPUs

```bash
./test-gpu.sh
```

Shows:
- Vulkan devices available
- NVIDIA/Intel/AMD GPU status
- WebGPU device selection

### 4. `test-generation.py` - Generate WGSL Shaders

```bash
# Default: MatMul 1024x1024
python3 test-generation.py

# Different matrix sizes (affects kernel selection)
python3 test-generation.py --op matmul --m 64 --k 64 --n 64    # Small (MatMulPacked)
python3 test-generation.py --op matmul --m 1024 --k 1024 --n 1024  # Large (MatMulSubgroup)

# Other operators
python3 test-generation.py --op add --shape 1024 1024
python3 test-generation.py --op softmax --shape 32 1024

# Save shader to file
python3 test-generation.py --save shader.wgsl

# Quiet mode (only shader code)
python3 test-generation.py --quiet > shader.wgsl

# List operators
python3 test-generation.py --list
```

## MatMul Kernel Selection

The WebGPU backend selects different MatMul kernels based on matrix dimensions:

| Condition | Kernel |
|-----------|--------|
| N < 8 and K < 8 | MatMulNaive |
| Subgroups available | MatMulSubgroup |
| Otherwise | MatMulPacked |

## GPU Selection

WebGPU uses Dawn's Vulkan backend on Linux. GPU selection is based on:

1. **Power Preference**: `high-performance` selects discrete GPUs
2. **Device ID**: Can specify specific Vulkan device index

```python
# Python example
providers = [('WebGpuExecutionProvider', {
    'powerPreference': 'high-performance',
    'deviceId': 0  # Vulkan device index
})]
```

## Subgroup Sizes by GPU Vendor

| Vendor | Subgroup Size |
|--------|---------------|
| Intel Arc | 8, 16, 32 |
| NVIDIA | 32 (warp) |
| AMD | 32, 64 (wavefront) |

The shader includes branches for all sizes, but only one executes at runtime.

## Troubleshooting

### NVIDIA Vulkan not working
```bash
# Check driver version match
cat /proc/driver/nvidia/version  # Kernel module
dpkg -l | grep libnvidia-compute  # Userspace

# If mismatched, reboot to load new kernel module
sudo reboot
```

### No WebGPU provider
```bash
# Rebuild with WebGPU
./build-webgpu.sh --clean
```

### Check Vulkan devices
```bash
vulkaninfo --summary
```
