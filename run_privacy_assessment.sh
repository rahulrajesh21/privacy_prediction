#!/bin/bash

# Script to run privacy risk assessment

# Default values
EGO_ID="0"
DATA_DIR="facebook"
OUTPUT_DIR="privacy_reports"
MODEL_PATH=""  # Will be determined automatically based on ego_id

# Help function
function show_help {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -e, --ego-id EGO_ID         Ego network ID to assess (default: 0)"
    echo "  -d, --data-dir DIR          Path to Facebook dataset (default: facebook)"
    echo "  -o, --output-dir DIR        Output directory (default: privacy_reports)"
    echo "  -m, --model-path PATH       Path to trained model (default: auto-determined)"
    echo "  -t, --train                 Train model before assessment"
    echo "  -h, --help                  Show this help message"
    exit 0
}

# Parse command line arguments
TRAIN_MODEL=false
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -e|--ego-id)
            EGO_ID="$2"
            shift 2
            ;;
        -d|--data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -t|--train)
            TRAIN_MODEL=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Create output directories
mkdir -p output
mkdir -p "$OUTPUT_DIR"

echo "Privacy Risk Assessment"
echo "======================="
echo "Ego Network ID: $EGO_ID"
echo "Data Directory: $DATA_DIR"
echo "Output Directory: $OUTPUT_DIR"

# Train model if requested
if [ "$TRAIN_MODEL" = true ]; then
    echo -e "\nTraining privacy risk assessment model..."
    python train.py --data_dir "$DATA_DIR" --model_type explainable --conv_type GAT --epochs 20 --ego_id "$EGO_ID"
fi

# Determine model path if not provided
if [ -z "$MODEL_PATH" ]; then
    if [ -f "output/best_model_${EGO_ID}.pt" ]; then
        MODEL_PATH="output/best_model_${EGO_ID}.pt"
    elif [ -f "output/final_model_${EGO_ID}.pt" ]; then
        MODEL_PATH="output/final_model_${EGO_ID}.pt"
    fi
    echo "Model Path: $MODEL_PATH"
fi

# Run privacy assessment
echo -e "\nRunning privacy risk assessment..."
if [ -n "$MODEL_PATH" ]; then
    python privacy_assessment.py --ego_id "$EGO_ID" --data_dir "$DATA_DIR" --model_path "$MODEL_PATH" --output_dir "$OUTPUT_DIR"
else
    python privacy_assessment.py --ego_id "$EGO_ID" --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR"
fi

echo -e "\nPrivacy assessment complete. Reports are available in $OUTPUT_DIR directory." 