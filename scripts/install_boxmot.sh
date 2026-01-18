#!/usr/bin/env bash
#
# BOXMOT Installation Script for OTVision
# ========================================
# This script installs BOXMOT multi-object tracking dependencies
# and optionally downloads ReID weights for appearance-based trackers.
#
# Usage:
#   ./scripts/install_boxmot.sh              # Install BOXMOT dependencies only
#   ./scripts/install_boxmot.sh --with-reid  # Also download ReID weights
#
# Requirements:
#   - Python 3.12
#   - uv package manager

set -e

# Resolve script directory (handles symlinks)
SOURCE=${BASH_SOURCE[0]}
while [ -L "$SOURCE" ]; do
  DIR=$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)
  SOURCE=$(readlink "$SOURCE")
  [[ $SOURCE != /* ]] && SOURCE=$DIR/$SOURCE
done
SCRIPT_DIR=$(cd -P "$(dirname "$SOURCE")" >/dev/null 2>&1 && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Parse command line arguments
WITH_REID=false
for arg in "$@"; do
    case $arg in
        --with-reid)
            WITH_REID=true
            shift
            ;;
        --help|-h)
            echo "BOXMOT Installation Script for OTVision"
            echo ""
            echo "Usage:"
            echo "  $0              Install BOXMOT dependencies only"
            echo "  $0 --with-reid  Also download ReID weights"
            echo "  $0 --help       Show this help message"
            exit 0
            ;;
    esac
done

print_header "BOXMOT Installation for OTVision"

cd "$PROJECT_DIR" || exit 1
print_info "Working directory: $PROJECT_DIR"

# Step 1: Check Python version
print_info "Checking Python version..."
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.12."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+')
if [[ "$PYTHON_VERSION" != "3.12" ]]; then
    print_warning "Python $PYTHON_VERSION detected. Python 3.12 is required."
    print_warning "BOXMOT may not work correctly with other versions."
else
    print_success "Python 3.12 detected"
fi

# Step 2: Check uv availability
print_info "Checking uv package manager..."
if ! command -v uv &>/dev/null; then
    print_warning "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"

    if ! command -v uv &>/dev/null; then
        print_error "Failed to install uv. Please install manually: https://github.com/astral-sh/uv"
        exit 1
    fi
fi
print_success "uv package manager is available"

# Step 3: Install BOXMOT dependencies
print_header "Installing BOXMOT Dependencies"
print_info "Running: uv pip install -e .[tracking_boxmot]"

if uv pip install -e ".[tracking_boxmot]"; then
    print_success "BOXMOT dependencies installed successfully"
else
    print_error "Failed to install BOXMOT dependencies"
    exit 1
fi

# Step 4: Verify installation
print_header "Verifying Installation"
print_info "Testing BOXMOT import..."

if $PYTHON_CMD -c "from boxmot import TRACKERS; print(f'Available trackers: {TRACKERS}')" 2>/dev/null; then
    print_success "BOXMOT is properly installed"
else
    print_error "BOXMOT import failed. Installation may be incomplete."
    exit 1
fi

print_info "Testing OTVision BOXMOT adapter..."
if $PYTHON_CMD -c "from OTVision.track.tracker.tracker_plugin_boxmot import BoxmotTrackerAdapter; print('BoxmotTrackerAdapter loaded successfully')" 2>/dev/null; then
    print_success "OTVision BOXMOT adapter is working"
else
    print_warning "OTVision BOXMOT adapter could not be loaded. Check for errors."
fi

# Step 5: Optionally download ReID weights
if [ "$WITH_REID" = true ]; then
    print_header "Downloading ReID Weights"

    WEIGHTS_DIR="$PROJECT_DIR/weights"
    mkdir -p "$WEIGHTS_DIR"

    REID_URL="https://github.com/mikel-brostrom/yolo_tracking/releases/download/v9.0/osnet_x0_25_msmt17.pt"
    REID_FILE="$WEIGHTS_DIR/osnet_x0_25_msmt17.pt"

    if [ -f "$REID_FILE" ]; then
        print_info "ReID weights already exist at: $REID_FILE"
    else
        print_info "Downloading OSNet ReID weights..."
        if curl -L -o "$REID_FILE" "$REID_URL"; then
            print_success "ReID weights downloaded to: $REID_FILE"
        else
            print_warning "Failed to download ReID weights. You can download manually from:"
            print_warning "  $REID_URL"
        fi
    fi

    echo ""
    print_info "To use appearance-based trackers (BotSORT, BoostTrack, etc.),"
    print_info "add to your config:"
    echo ""
    echo "TRACK:"
    echo "  BOXMOT:"
    echo "    ENABLED: true"
    echo "    TRACKER_TYPE: \"botsort\""
    echo "    REID_WEIGHTS: \"$REID_FILE\""
    echo ""
fi

# Print summary
print_header "Installation Complete"

echo "Available BOXMOT Trackers:"
echo ""
echo "  Motion-Only (fast, no ReID weights needed):"
echo "    - bytetrack  : High FPS, recommended for CPU"
echo "    - ocsort     : Alternative motion-only tracker"
echo ""
echo "  Appearance-Based (higher accuracy, requires ReID weights):"
echo "    - botsort    : Best overall accuracy"
echo "    - boosttrack : High identity consistency"
echo "    - strongsort : Balanced performance"
echo "    - deepocsort : Enhanced OcSORT with ReID"
echo "    - hybridsort : Hybrid motion-appearance"
echo ""
echo "Quick Start:"
echo "  1. Copy boxmot_config.example.yaml to user_config.otvision.yaml"
echo "  2. Enable BOXMOT by setting TRACK.BOXMOT.ENABLED: true"
echo "  3. Run tracking: uv run track.py --paths /path/to/*.otdet"
echo ""
echo "Documentation: BOXMOT_INTEGRATION.md"
echo ""

if [ "$WITH_REID" = false ]; then
    print_info "Tip: Run with --with-reid to download ReID weights for appearance trackers"
fi

print_success "BOXMOT installation completed successfully!"
