#!/bin/bash
set -e

# Configuration
RESULTS_DIR="../gemma-12b-results"
OUTPUT_DIR="./output"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}BigBench Results Extraction${NC}"
echo "================================"
echo ""

# Check if Python script exists
if [ ! -f "$SCRIPT_DIR/parse_bigbench.py" ]; then
    echo "Error: parse_bigbench.py not found in $SCRIPT_DIR"
    exit 1
fi

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo "Error: Results directory '$RESULTS_DIR' not found"
    exit 1
fi

# Run the Python parser
echo -e "${GREEN}Running parser...${NC}"
python3 "$SCRIPT_DIR/parse_bigbench.py" "$RESULTS_DIR" "$OUTPUT_DIR"

# Display summary
if [ -f "$OUTPUT_DIR/bigbench_summary.csv" ]; then
    echo ""
    echo -e "${GREEN}Summary:${NC}"
    column -t -s ',' "$OUTPUT_DIR/bigbench_summary.csv" | head -20
fi

echo ""
echo -e "${GREEN}Output files:${NC}"
echo "  - $OUTPUT_DIR/bigbench_results.csv (detailed results)"
echo "  - $OUTPUT_DIR/bigbench_summary.csv (summary by model)"