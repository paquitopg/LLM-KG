# LLMKG - LLM Knowledge Graph Extraction Framework

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/paquitopg/LLMKG)](https://github.com/paquitopg/LLMKG)

A comprehensive framework for extracting structured knowledge graphs from PDF documents using Large Language Models (LLMs). The system supports both text-based and multimodal extraction, with sophisticated entity merging and ontology-aware processing.

## 🚀 Key Features

- **Multi-Modal Extraction**: Extract knowledge from both text and visual content in PDFs
- **Ontology-Driven**: Uses PEKG (Private Equity Knowledge Graph) ontology with 17+ entity types and rich relationships
- **Intelligent Document Classification**: Automatic document type identification with contextual summarization
- **Advanced Entity Merging**: PEKG-aware page-level and inter-document merging strategies
- **Multi-Document Processing**: Process entire project folders with intelligent cross-document entity resolution
- **Multiple LLM Providers**: Support for Azure OpenAI and Google Vertex AI
- **Interactive Visualizations**: Rich HTML visualizations with provenance tracking
- **Evaluation Framework**: Comprehensive KG quality assessment against ontology compliance

## 📁 Project Structure

```
LLM-KG-extraction/
├── configs/                        # Configuration files
│   ├── extraction_config.yaml     # Extraction & processing behavior settings
│   └── llm_config.yaml           # LLM provider & model configurations
├── ontologies/                    # Ontology definitions
│   └── pekg_ontology_teasers.yaml # PEKG ontology definition
├── llm_kg_extraction/             # Main package
│   ├── _1_document_ingestion/    # PDF processing and document classification
│   │   ├── pdf_parser.py         # PDF text/image extraction with PyMuPDF
│   │   └── document_classifier.py # LLM-based document classification
│   ├── _2_context_understanding/ # Document context preparation
│   │   └── document_context_preparer.py
│   ├── _3_knowledge_extraction/  # Core KG extraction logic
│   │   ├── page_llm_processor.py # Page-level entity/relation extraction
│   │   ├── kg_constructor_single_doc.py # Document-level KG construction
│   │   └── document_aware_extraction/ # Document-aware extraction methods
│   │       ├── chunk_llm_processor.py # Chunk-level LLM processing
│   │       ├── context_manager.py # Context management for chunks
│   │       ├── document_structure_analysis.py # Document structure analysis
│   │       ├── entity_tracker.py # Entity tracking across chunks
│   │       ├── hierarchical_processor.py # Hierarchical processing
│   │       ├── semantic_chunker.py # Semantic document chunking
│   │       └── two_pass_processor.py # Two-pass processing strategy
│   ├── _4_knowledge_graph_operations/ # KG merging and post-processing
│   │   ├── page_level_merger.py  # PEKG-aware entity merging
│   │   ├── inter_document_merger.py # Multi-document KG consolidation
│   │   ├── ontology_aware_merger.py # Ontology-aware merging strategies
│   │   └── common_kg_utils.py    # Entity similarity and utilities
│   ├── _5_transformations/       # KG transformation utilities
│   │   ├── transform_json.py     # KG transformation utilities
│   │   └── split_json_outputs.py # Output splitting utilities
│   ├── llm_integrations/         # LLM provider wrappers
│   │   ├── azure_llm.py         # Azure OpenAI integration
│   │   ├── vertex_llm.py        # Google Vertex AI integration
│   │   └── base_llm_wrapper.py  # Abstract LLM interface
│   ├── loaders/                  # Configuration and ontology loaders
│   │   ├── config_loader.py     # Configuration file loading
│   │   └── ontology_loader.py   # PEKG ontology processing
│   ├── visualization_tools/       # KG visualization
│   │   └── KG_visualizer.py     # Interactive HTML visualizations
│   ├── core_components/          # Core utilities
│   │   └── document_scanner.py  # PDF file discovery
│   ├── lib/                      # External library dependencies
│   │   ├── bindings/            # JavaScript bindings
│   │   ├── tom-select/          # Tom Select library
│   │   └── vis-9.1.2/           # Vis.js network visualization
│   ├── main_orchestrator.py     # Main pipeline orchestrator
│   └── merger_diagnostic_tool.py # Entity merging analysis
├── env_file_example              # Environment variables template
├── pyproject.toml               # Poetry project configuration
├── poetry.lock                   # Poetry dependency lock file
└── run_orchestrator.sh          # Shell script for running orchestrator
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Azure OpenAI or Google Vertex AI access

### Install Dependencies

```bash
# Core dependencies
pip install pypdf pymupdf pillow pyyaml python-dotenv

# Visualization and graph processing
pip install networkx matplotlib pyvis

# LLM integrations
pip install google-generativeai openai

# Additional utilities
pip install pathlib typing-extensions
```

### Environment Setup

Create a `.env` file in the project root with your LLM provider credentials. The framework supports both Azure OpenAI and Google Vertex AI.

#### Option 1: Azure OpenAI Setup

```env
# Main GPT-4o model
AZURE_OPENAI_ENDPOINT_gpt-4o=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY_gpt-4o=your_api_key_here
AZURE_DEPLOYMENT_NAME_gpt-4o=gpt-4o
AZURE_OPENAI_API_VERSION_gpt-4o=2024-02-15-preview

# GPT-4.1 model (alternative)
AZURE_OPENAI_ENDPOINT_gpt-4.1=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY_gpt-4.1=your_api_key_here
AZURE_DEPLOYMENT_NAME_gpt-4.1=gpt-4.1
AZURE_OPENAI_API_VERSION_gpt-4.1=2024-02-15-preview

# GPT-4.1-mini model (for classification/summarization)
AZURE_OPENAI_ENDPOINT_gpt-4.1-mini=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY_gpt-4.1-mini=your_api_key_here
AZURE_DEPLOYMENT_NAME_gpt-4.1-mini=gpt-4.1-mini
AZURE_OPENAI_API_VERSION_gpt-4.1-mini=2024-02-15-preview
```

#### Option 2: Google Vertex AI Setup

```env
# Google Cloud credentials
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1

# Model preferences
VERTEXAI_DEFAULT_MODEL=gemini-2.5-flash
VERTEXAI_ADVANCED_MODEL=gemini-2.5-pro
```

#### Environment Variable Naming Convention

For Azure OpenAI, the framework uses a suffix-based naming system:
- `AZURE_OPENAI_ENDPOINT_{model_name}`: API endpoint
- `AZURE_OPENAI_API_KEY_{model_name}`: API key
- `AZURE_DEPLOYMENT_NAME_{model_name}`: Deployment name
- `AZURE_OPENAI_API_VERSION_{model_name}`: API version

**Note**: Replace `{model_name}` with your actual model identifier (e.g., `gpt-4o`, `gpt-4.1`, etc.)

## 🎯 Quick Start

The examples below show how CLI arguments override config file defaults. All examples assume you have:
1. Created a `.env` file with your API credentials
2. Configured default settings in `configs/extraction_config.yaml`

### Example 1: Single Document Processing (Vertex AI)

**Uses config defaults + minimal CLI overrides:**

```bash
python main_orchestrator.py \
    /path/to/document.pdf \
    "MyProject" \
    vertexai \
    --main_model_name "gemini-2.5-pro" \
    --classification_model_name "gemini-2.5-flash" \
    --summary_model_name "gemini-2.5-flash"
```

**What this does:**
- Uses `construction_mode: "parallel"` from config
- Uses `extraction_mode: "multimodal"` from config  
- Uses `max_workers: 60` from config
- Overrides only the model names via CLI

### Example 2: Multi-Document Processing (Azure OpenAI)

**Uses config defaults + specific processing overrides:**

```bash
python main_orchestrator.py \
    /path/to/project/folder \
    "CompanyAnalysis" \
    azure \
    --main_model_name "gpt-4o" \
    --main_azure_model_env_suffix "GPT4O" \
    --classification_model_name "gpt-4.1-mini" \
    --classification_azure_model_env_suffix "GPT4_1_MINI" \
    --summary_model_name "gpt-4.1-mini" \
    --summary_azure_model_env_suffix "GPT4_1_MINI" \
    --construction_mode iterative \
    --extraction_mode text \
    --max_workers 8 \
    --dump_page_kgs \
    --transform_final_kg
```

**What this does:**
- Overrides `construction_mode` to `"iterative"` (sequential processing)
- Overrides `extraction_mode` to `"text"` (text-only extraction)
- Overrides `max_workers` to `8` (reduced parallelism)
- Enables intermediate KG dumping and final transformation

### Example 3: Document-Aware Processing

**Uses advanced document-aware mode:**

```bash
python main_orchestrator.py \
    /path/to/complex/document.pdf \
    "ComplexAnalysis" \
    azure \
    --main_model_name "gpt-4o" \
    --processing_mode document_aware \
    --chunk_size 3000 \
    --chunk_overlap 300 \
    --construction_mode iterative
```

**What this does:**
- Uses semantic chunking instead of page-based processing
- Customizes chunk size and overlap for better context
- Uses iterative construction for complex documents

### Configuration File Customization

To avoid repeating CLI arguments, customize your `configs/extraction_config.yaml`:

```yaml
processing_behavior:
  default_construction_mode: "iterative"  # Change from "parallel"
  default_extraction_mode: "text"         # Change from "multimodal"
  default_max_workers: 8                  # Change from 60
  default_dump_page_kgs: true             # Change from false
```

## 🧠 PEKG Ontology

The system uses a comprehensive Private Equity Knowledge Graph ontology with:

### Entity Types
- **Companies & Organizations**: Company, GovernmentBody, Advisor
- **People & Roles**: Person, Position, Shareholder
- **Financial Data**: FinancialMetric, OperationalKPI, Headcount
- **Products & Technology**: ProductOrService, Technology
- **Market Context**: MarketContext, UseCaseOrIndustry, Location
- **Transactions**: TransactionContext, HistoricalEvent

### Key Relationships
- `pekg:employs`, `pekg:hasShareholder`, `pekg:reportsFinancialMetric`
- `pekg:offers`, `pekg:operatesIn`, `pekg:advisedBy`
- `pekg:experiencedEvent`, `pekg:hasOfficeIn`

## ⚙️ Configuration System

The LLM-KG-Extraction framework uses a three-tier configuration system for maximum flexibility:

### 1. Configuration Files (`configs/` directory)

**Purpose**: Default settings and behavior configuration
**Location**: `configs/extraction_config.yaml`, `configs/llm_config.yaml`

#### Key Configuration Categories:

**Processing Behavior** (`extraction_config.yaml`):
- `default_construction_mode`: `"iterative"` (sequential) or `"parallel"` (concurrent)
- `default_extraction_mode`: `"text"` or `"multimodal"`
- `default_max_workers`: Number of parallel workers (default: 60)
- `default_dump_page_kgs`: Save intermediate KGs (default: false)
- `default_transform_final_kg`: Transform final KG (default: false)

**LLM Model Settings** (`llm_config.yaml`):
- Model-specific parameters (temperature, max_tokens, etc.)
- Default model selection for different tasks
- Rate limiting and error handling policies
- Provider-specific configurations

Note: Previously there was a `document_aware_config.yaml`. It is deprecated and no longer used. Document-aware parameters now live under `processing_behavior` in `extraction_config.yaml`.

### 2. Environment Variables (`.env` file)

**Purpose**: API credentials and sensitive information
**Location**: `.env` file in project root

#### Required Variables:

**For Azure OpenAI**:
```env
AZURE_OPENAI_ENDPOINT_gpt-4o=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY_gpt-4o=your_api_key_here
AZURE_DEPLOYMENT_NAME_gpt-4o=gpt-4o
AZURE_OPENAI_API_VERSION_gpt-4o=2024-02-15-preview
```

**For Google Vertex AI**:
```env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_CLOUD_PROJECT=your_project_id
GOOGLE_CLOUD_LOCATION=us-central1
VERTEXAI_DEFAULT_MODEL=gemini-2.5-flash
VERTEXAI_ADVANCED_MODEL=gemini-2.5-pro
```

### 3. Command Line Arguments

**Purpose**: Runtime overrides and specific execution parameters
**Usage**: Override config file defaults for specific runs

#### Core Arguments (Required):
- `input_folder_path`: Path to PDF documents
- `project_name`: Project identifier
- `llm_provider`: `"azure"` or `"vertexai"`

#### Model Selection Arguments:
- `--main_model_name`: Main extraction model (e.g., `"gpt-4o"`, `"gemini-2.5-flash"`)
- `--main_azure_model_env_suffix`: Azure model environment suffix
- `--classification_model_name`: Document classification model
- `--summary_model_name`: Document summarization model

#### Processing Control Arguments:
- `--construction_mode`: Override default construction mode
- `--extraction_mode`: Override default extraction mode
- `--max_workers`: Override default worker count
- `--processing_mode`: Choose `"page_based"` or `"document_aware"`

#### Output Control Arguments:
- `--dump_page_kgs`: Save intermediate page KGs
- `--transform_final_kg`: Apply transformations to final KG
- `--run_diagnostics`: Run merger diagnostics

### Configuration Priority

1. **CLI Arguments** (highest priority) - Override everything
2. **Config Files** (medium priority) - Default behavior
3. **Environment Variables** (lowest priority) - Credentials and basic settings

### Document Types
- `financial_teaser`, `financial_report`, `legal_contract`, `technical_documentation`

## 📊 Output Formats

### Standard KG Format
```json
{
  "entities": [
    {
      "id": "e1",
      "type": "pekg:Company",
      "name": "TechCorp",
      "industry": "Software",
      "foundedYear": 2020
    }
  ],
  "relationships": [
    {
      "source": "e1",
      "target": "e2",
      "type": "pekg:reportsMetric"
    }
  ]
}
```

### Multi-Document Format (with Provenance)
```json
{
  "entities": [
    {
      "id": "e1",
      "type": "pekg:Company",
      "name": [
        {"value": "TechCorp", "source_doc_id": "doc1"},
        {"value": "TechCorp Inc.", "source_doc_id": "doc2"}
      ],
      "_source_document_ids": ["doc1", "doc2"]
    }
  ]
}
```

## 🔧 Entity Merging Strategies

The system implements ontology-aware merging with different strategies per entity type:

- **Preserve Strategy**: Context entities (TransactionContext) - rarely merged
- **Strict Strategy**: Financial metrics - exact temporal/scope matching required
- **Liberal Strategy**: Reference entities (Company, Person) - broader similarity matching
- **Moderate Strategy**: Business entities - balanced approach

## 📈 Evaluation & Diagnostics

### KG Quality Evaluation
```bash
python KG_evaluator.py /path/to/kg.json /path/to/ontology.yaml /path/to/output.json
```

Evaluates:
- Ontology compliance (entity/relation type validity)
- Domain/range constraint adherence
- Graph connectivity and quality metrics
- Overall scoring (0-1 scale)

### Merger Diagnostics
```python
from tests.merger_diagnostic_tool import diagnose_merger_issues
diagnose_merger_issues(page_kgs, final_kg)
```

Provides detailed analysis of:
- Entity loss patterns by type
- Unexpected merges
- Similarity threshold recommendations

## 🎨 Visualization

The system generates interactive HTML visualizations with:

- Entity type-based color coding
- Hover tooltips with detailed entity information
- Relationship labels and directionality
- Multi-document provenance tracking
- Physics-based layout with clustering

## 🛠️ Troubleshooting

### Configuration Issues

#### Environment Variables Not Found
**Error**: `Environment variable AZURE_OPENAI_API_KEY_gpt-4o not found`

**Solutions**:
1. Check your `.env` file exists in the project root
2. Verify the variable name matches exactly (case-sensitive)
3. Ensure no extra spaces around the `=` sign
4. For Azure models, check the suffix matches your CLI argument

```bash
# If using --main_azure_model_env_suffix "GPT4O", your .env should have:
AZURE_OPENAI_API_KEY_GPT4O=your_key_here
# NOT:
AZURE_OPENAI_API_KEY_gpt-4o=your_key_here
```

#### Config File Not Loading
**Error**: `Configuration system not available. Using hardcoded defaults.`

**Solutions**:
1. Ensure `configs/extraction_config.yaml` exists
2. Check YAML syntax is valid (use online YAML validator)
3. Verify file permissions allow reading

#### Model Selection Conflicts
**Error**: `Model 'gpt-4o' not found in configuration`

**Solutions**:
1. Check `configs/llm_config.yaml` contains the model definition
2. Verify environment variables are set for the model
3. Use `--main_model_name` to specify exact model name

### Processing Issues

#### Empty KG Output
**Symptoms**: Final KG contains no entities or relationships

**Debugging Steps**:
1. Enable debug mode: `export KG_DEBUG=1`
2. Check document classification results
3. Verify LLM responses are being parsed correctly
4. Try with `--dump_page_kgs` to see intermediate results

#### Entity Over-merging
**Symptoms**: Distinct entities are incorrectly merged together

**Solutions**:
1. Adjust similarity thresholds in merger configuration
2. Use `--run_diagnostics` to analyze merging patterns
3. Consider using `--construction_mode iterative` for better context

#### Memory Issues
**Symptoms**: Process crashes with out-of-memory errors

**Solutions**:
1. Reduce `--max_workers` (try 4 or 8)
2. Use `--extraction_mode text` instead of `multimodal`
3. Process documents individually instead of batch processing

### Common Issues

- **Empty KG Output**: Check document classification results and LLM response parsing
- **Entity Over-merging**: Adjust similarity thresholds in merger configuration
- **Memory Issues**: Reduce `max_workers` for large documents
- **Visualization Errors**: Ensure PyVis dependencies are installed correctly

### Debug Mode

Enable detailed logging:
```bash
export KG_DEBUG=1
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For questions and support, please open an issue on the GitHub repository.

---
