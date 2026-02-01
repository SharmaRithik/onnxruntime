#!/bin/bash
# =============================================================================
# build-webgpu.sh - Build ONNX Runtime with WebGPU backend
# =============================================================================
# Usage: ./build-webgpu.sh [--clean] [--debug]
#
# Options:
#   --clean    Clean build directory before building
#   --debug    Build in Debug mode (default: Release)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$ORT_ROOT/build-webgpu"
BUILD_TYPE="Release"
CLEAN_BUILD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--clean] [--debug]"
            echo ""
            echo "Options:"
            echo "  --clean    Clean build directory before building"
            echo "  --debug    Build in Debug mode (default: Release)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "ONNX Runtime WebGPU Backend Build Script"
echo "=============================================="
echo "ORT Root:    $ORT_ROOT"
echo "Build Dir:   $BUILD_DIR"
echo "Build Type:  $BUILD_TYPE"
echo "Clean Build: $CLEAN_BUILD"
echo "=============================================="

# Check and install Python dependencies
echo ""
echo "Checking Python dependencies..."

install_pip_package() {
    local package=$1
    if ! python3 -c "import $package" 2>/dev/null; then
        echo "  Installing $package..."
        # Try normal install first, fall back to --break-system-packages for Ubuntu 24.04+
        if ! pip3 install "$package" 2>/dev/null; then
            pip3 install --break-system-packages "$package" 2>/dev/null || {
                echo "ERROR: Failed to install $package"
                echo "Please install manually: pip3 install $package"
                exit 1
            }
        fi
        echo "  ✓ $package installed"
    else
        echo "  ✓ $package already installed"
    fi
}

install_pip_package numpy
install_pip_package onnx

echo "Python dependencies OK"
echo "=============================================="

# Clean if requested
if [ "$CLEAN_BUILD" = true ] && [ -d "$BUILD_DIR" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Run build script
echo ""
echo "Starting build..."
echo ""

"$ORT_ROOT/build.sh" \
    --config "$BUILD_TYPE" \
    --build_dir "$BUILD_DIR" \
    --parallel \
    --skip_tests \
    --use_webgpu \
    --build_wheel \
    --enable_pybind

echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Build output: $BUILD_DIR/$BUILD_TYPE"
echo ""
echo "To use in Python:"
echo "  export PYTHONPATH=$BUILD_DIR/$BUILD_TYPE"
echo "  python3 -c \"import onnxruntime; print(onnxruntime.get_available_providers())\""
echo ""
