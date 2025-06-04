#!/bin/bash

# Run Multi-view Privacy Risk Assessment

# Default values
FB_EGO_ID="0"
TW_EGO_ID="104615636"
TW_SAMPLE_SIZE=3
CONV_TYPE="GAT"
HIDDEN_DIM=64
NUM_LAYERS=3
EPOCHS=30
OUTPUT_DIR="multiview_output"
REPORT_DIR="multiview_reports"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --fb_ego_id)
      FB_EGO_ID="$2"
      shift
      shift
      ;;
    --tw_ego_id)
      TW_EGO_ID="$2"
      shift
      shift
      ;;
    --tw_sample_size)
      TW_SAMPLE_SIZE="$2"
      shift
      shift
      ;;
    --conv_type)
      CONV_TYPE="$2"
      shift
      shift
      ;;
    --hidden_dim)
      HIDDEN_DIM="$2"
      shift
      shift
      ;;
    --num_layers)
      NUM_LAYERS="$2"
      shift
      shift
      ;;
    --epochs)
      EPOCHS="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --report_dir)
      REPORT_DIR="$2"
      shift
      shift
      ;;
    --help)
      echo "Usage: ./run_multiview.sh [options]"
      echo "Options:"
      echo "  --fb_ego_id       Facebook ego network ID (default: 0)"
      echo "  --tw_ego_id       Twitter ego network ID (default: 104615636)"
      echo "  --tw_sample_size  Number of Twitter networks to sample (default: 3)"
      echo "  --conv_type       GNN layer type: GCN, GAT, SAGE (default: GAT)"
      echo "  --hidden_dim      Hidden dimension size (default: 64)"
      echo "  --num_layers      Number of GNN layers (default: 3)"
      echo "  --epochs          Number of training epochs (default: 30)"
      echo "  --output_dir      Output directory for models (default: multiview_output)"
      echo "  --report_dir      Output directory for reports (default: multiview_reports)"
      echo "  --help            Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Create directories if they don't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$REPORT_DIR"

echo "============================================================"
echo "Multi-view Privacy Risk Assessment"
echo "============================================================"
echo "Facebook Ego ID: $FB_EGO_ID"
echo "Twitter Ego ID: $TW_EGO_ID"
echo "Twitter Sample Size: $TW_SAMPLE_SIZE"
echo "Model Configuration: $CONV_TYPE, $HIDDEN_DIM hidden dims, $NUM_LAYERS layers"
echo "Training for $EPOCHS epochs"
echo "Output directory: $OUTPUT_DIR"
echo "Report directory: $REPORT_DIR"
echo "============================================================"

# Step 1: Train the multi-view model
echo -e "\n[1/3] Training multi-view model..."
python multiview_train.py \
  --fb_ego_id "$FB_EGO_ID" \
  --tw_ego_id "$TW_EGO_ID" \
  --tw_sample_size "$TW_SAMPLE_SIZE" \
  --conv_type "$CONV_TYPE" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_layers "$NUM_LAYERS" \
  --epochs "$EPOCHS" \
  --output_dir "$OUTPUT_DIR"

# Check if training was successful
if [ $? -ne 0 ]; then
  echo "Error: Training failed. Exiting."
  exit 1
fi

# Step 2: Evaluate the model
echo -e "\n[2/3] Evaluating multi-view model..."
python multiview_evaluate.py \
  --fb_ego_id "$FB_EGO_ID" \
  --tw_ego_id "$TW_EGO_ID" \
  --conv_type "$CONV_TYPE" \
  --hidden_dim "$HIDDEN_DIM" \
  --num_layers "$NUM_LAYERS" \
  --output_dir "$REPORT_DIR"

# Check if evaluation was successful
if [ $? -ne 0 ]; then
  echo "Error: Evaluation failed. Exiting."
  exit 1
fi

# Step 3: Display summary of results
echo -e "\n[3/3] Summarizing results..."
echo "============================================================"
echo "Multi-view Privacy Risk Assessment Complete"
echo "============================================================"
echo "Results saved to:"
echo "- Models: $OUTPUT_DIR"
echo "- Reports: $REPORT_DIR"
echo "- Privacy Report: $REPORT_DIR/multiview_privacy_report_fb${FB_EGO_ID}_tw${TW_EGO_ID}.txt"
echo "- Visualizations: $REPORT_DIR/privacy_comparison_fb${FB_EGO_ID}_tw${TW_EGO_ID}.png"
echo "============================================================"
echo "To view the detailed privacy report, run:"
echo "cat $REPORT_DIR/multiview_privacy_report_fb${FB_EGO_ID}_tw${TW_EGO_ID}.txt"
echo "============================================================" 