#!/bin/bash

# Script to run the LLM-KG-extraction orchestrator
# Usage: ./run_orchestrator.sh <input_folder> <project_name> <llm_provider> [additional_args]

# Check if we have the minimum required arguments
if [ $# -lt 3 ]; then
    echo "Usage: $0 <input_folder> <project_name> <llm_provider> [additional_args]"
    echo ""
    echo "Required arguments:"
    echo "  input_folder    - Path to folder containing PDF documents"
    echo "  project_name    - Name of the project for output organization"
    echo "  llm_provider    - LLM provider (azure or vertexai)"
    echo ""
    echo "Optional arguments:"
    echo "  --main_model_name <model>           - Main LLM model name"
    echo "  --processing_mode <mode>            - Processing mode (page_based or document_aware)"
    echo "  --dump_page_kgs                     - Save intermediate page KGs"
    echo "  --construction_mode <mode>          - Construction mode (iterative or parallel)"
    echo "  --extraction_mode <mode>            - Extraction mode (text or multimodal)"
    echo "  --max_workers <num>                 - Number of parallel workers"
    echo ""
    echo "Examples:"
    echo "  $0 \"C:/PE/Teasers\" \"Airelle\" \"vertexai\""
    echo "  $0 \"C:/PE/Teasers\" \"Airelle\" \"vertexai\" --main_model_name \"gemini-2.5-flash\" --processing_mode \"document_aware\" --dump_page_kgs"
    echo "  $0 \"C:/PE/Teasers\" \"Airelle\" \"azure\" --main_model_name \"gpt-4o\" --main_azure_model_env_suffix \"GPT4O_AZURE\""
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "Error: Poetry is not installed or not in PATH"
    echo "Please install Poetry first: https://python-poetry.org/docs/#installation"
    exit 1
fi

# Check if the main orchestrator file exists
if [ ! -f "llm_kg_extraction/main_orchestrator.py" ]; then
    echo "Error: main_orchestrator.py not found in llm_kg_extraction/ directory"
    exit 1
fi

# Set environment variables for better Python path handling
export PYTHONPATH="$SCRIPT_DIR/llm_kg_extraction:$PYTHONPATH"

echo "Starting LLM-KG-Extraction Orchestrator..."
echo "Working directory: $SCRIPT_DIR"
echo "Python path: $PYTHONPATH"
echo ""

# Run the orchestrator with Poetry
poetry run python llm_kg_extraction/main_orchestrator.py "$@"

echo ""
echo "Orchestrator execution completed."
