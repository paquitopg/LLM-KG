import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from string import Template

class ExtractionConfigLoader:
    """
    Loads and manages extraction configuration settings.
    Provides centralized access to all extraction parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None, llm_config_path: Optional[str] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the extraction configuration file. If None, uses default.
            llm_config_path: Path to the LLM configuration file. If None, uses default.
        """
        if config_path is None:
            # Use default config path - look in root configs folder first, then package
            root_config_path = Path(__file__).parent.parent.parent / "configs" / "extraction_config.yaml"
            package_config_path = Path(__file__).parent / "extraction_config.yaml"
            
            if root_config_path.exists():
                config_path = root_config_path
            else:
                config_path = package_config_path
        
        if llm_config_path is None:
            # Use default LLM config path - look in root configs folder first, then package
            root_llm_config_path = Path(__file__).parent.parent.parent / "configs" / "llm_config.yaml"
            package_llm_config_path = Path(__file__).parent / "llm_config.yaml"
            
            if root_llm_config_path.exists():
                llm_config_path = root_llm_config_path
            else:
                llm_config_path = package_llm_config_path
        
        self.config_path = Path(config_path)
        self.llm_config_path = Path(llm_config_path)
        self.config = self._load_config()
        self.llm_config = self._load_llm_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load extraction configuration from YAML file with environment variable substitution."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Substitute environment variables
            config_content = Template(config_content).safe_substitute(os.environ)
            
            # Parse the substituted content
            config = yaml.safe_load(config_content)
            
            # Set defaults for missing values
            config = self._set_defaults(config)
            return config
            
        except FileNotFoundError:
            print(f"Warning: Configuration file not found at {self.config_path}")
            print("Using default configuration values.")
            return self._get_default_config()
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Using default configuration values.")
            return self._get_default_config()
    
    def _load_llm_config(self) -> Dict[str, Any]:
        """Load LLM configuration from YAML file with environment variable substitution."""
        try:
            with open(self.llm_config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            
            # Substitute environment variables
            config_content = Template(config_content).safe_substitute(os.environ)
            
            # Parse the substituted content
            config = yaml.safe_load(config_content)
            return config
            
        except FileNotFoundError:
            print(f"Warning: LLM configuration file not found at {self.llm_config_path}")
            print("Using default LLM configuration values.")
            return self._get_default_llm_config()
        except Exception as e:
            print(f"Error loading LLM configuration: {e}")
            print("Using default LLM configuration values.")
            return self._get_default_llm_config()
    
    def _set_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Set default values for missing configuration keys."""
        defaults = self._get_default_config()
        
        def merge_configs(current: Dict[str, Any], default: Dict[str, Any]) -> Dict[str, Any]:
            """Recursively merge configurations with defaults."""
            result = default.copy()
            
            for key, value in current.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_configs(value, result[key])
                else:
                    result[key] = value
            
            return result
        
        return merge_configs(config, defaults)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            "quality_control": {
                "relevance_threshold": 0.3,
                "enable_quality_filtering": True,
                "min_entity_score": 0.2,
                "prevent_over_extraction": True
            },
            "extraction_focus": {
                "prioritize_main_entity": True,
                "max_entities_per_page": 0,
                "max_relationships_per_page": 0,
                "priority_entity_types": [
                    "pekg:Company",
                    "pekg:FinancialMetric",
                    "pekg:ProductOrService",
                    "pekg:Person",
                    "pekg:Technology"
                ]
            },
            "prompt_engineering": {
                "use_focused_prompts": True,
                "emphasize_main_entity": True,
                "include_relevance_instructions": True,
                "use_ontology_examples": True,
                "extraction_temperature": 0.1
            },
            "entity_type_scores": {
                "pekg:Company": 0.8,
                "pekg:FinancialMetric": 0.9,
                "pekg:Person": 0.7,
                "pekg:ProductOrService": 0.8,
                "pekg:Technology": 0.7,
                "pekg:MarketContext": 0.6,
                "pekg:UseCaseOrIndustry": 0.6,
                "pekg:Location": 0.5,
                "pekg:Shareholder": 0.8,
                "pekg:Advisor": 0.4,
                "pekg:TransactionContext": 0.7,
                "pekg:HistoricalEvent": 0.6,
                "pekg:OperationalKPI": 0.8,
                "pekg:Headcount": 0.7,
                "pekg:GovernmentBody": 0.3
            },
            "output": {
                "include_relevance_scores": False,
                "include_filtering_stats": True,
                "max_final_entities": 0,
                "max_final_relationships": 0
            },
            "processing_behavior": {
                "default_construction_mode": "parallel",
                "default_extraction_mode": "text",
                "default_max_workers": 4,
                "default_dump_page_kgs": False,
                "default_transform_final_kg": False,
                "default_run_diagnostics": False,
                "default_processing_mode": "page_based",
                "default_chunk_size": 4000,
                "default_min_chunk_size": 500,
                "default_chunk_overlap": 200,
                "default_respect_sentence_boundaries": True,
                "default_detect_topic_shifts": True
            },
            "document_classification": {
                "default_categories": ["financial_teaser", "legal_contract", "press_release", "annual_report"]
            }
        }
    
    def _get_default_llm_config(self) -> Dict[str, Any]:
        """Get default LLM configuration values."""
        return {
            "azure_openai": {
                "api_type": "azure",
                "api_version": "2024-02-15-preview",
                "max_retries": 3,
                "timeout": 60
            },
            "vertex_ai": {
                "max_retries": 3,
                "timeout": 60
            },
            "model_selection": {
                "default_provider": "azure_openai",
                "fallback_provider": "vertex_ai"
            }
        }
    
    # LLM Configuration Methods
    def get_llm_config(self) -> Dict[str, Any]:
        """Get the complete LLM configuration."""
        return self.llm_config
    
    def get_azure_openai_config(self) -> Dict[str, Any]:
        """Get Azure OpenAI configuration."""
        return self.llm_config.get("azure_openai", {})
    
    def get_vertex_ai_config(self) -> Dict[str, Any]:
        """Get Vertex AI configuration."""
        return self.llm_config.get("vertex_ai", {})
    
    def get_model_selection_config(self) -> Dict[str, Any]:
        """Get model selection configuration."""
        return self.llm_config.get("model_selection", {})
    
    def get_task_model(self, task: str) -> Dict[str, str]:
        """Get the preferred model for a specific task."""
        task_models = self.llm_config.get("model_selection", {}).get("task_models", {})
        return task_models.get(task, {})
    
    def get_model_config(self, provider: str, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        provider_config = self.llm_config.get(provider, {})
        models = provider_config.get("models", {})
        return models.get(model_name, {})
    
    def get_rate_limiting_config(self, provider: str) -> Dict[str, Any]:
        """Get rate limiting configuration for a provider."""
        rate_limiting = self.llm_config.get("rate_limiting", {})
        return rate_limiting.get(provider, {})
    
    # Existing Configuration Methods
    def get_quality_control_config(self) -> Dict[str, Any]:
        """Get quality control configuration."""
        return self.config.get("quality_control", {})
    
    def get_extraction_focus_config(self) -> Dict[str, Any]:
        """Get extraction focus configuration."""
        return self.config.get("extraction_focus", {})
    
    def get_prompt_engineering_config(self) -> Dict[str, Any]:
        """Get prompt engineering configuration."""
        return self.config.get("prompt_engineering", {})
    
    def get_entity_type_scores(self) -> Dict[str, float]:
        """Get entity type scoring configuration."""
        return self.config.get("entity_type_scores", {})
    
    def get_document_type_config(self, doc_type: str) -> Dict[str, Any]:
        """Get document type specific configuration."""
        return self.config.get("document_types", {}).get(doc_type, {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get("output", {})
    
    def get_relevance_threshold(self) -> float:
        """Get the relevance threshold for quality filtering."""
        return self.get_quality_control_config().get("relevance_threshold", 0.3)
    
    def is_quality_filtering_enabled(self) -> bool:
        """Check if quality filtering is enabled."""
        return self.get_quality_control_config().get("enable_quality_filtering", True)
    
    def should_prioritize_main_entity(self) -> bool:
        """Check if main entity prioritization is enabled."""
        return self.get_extraction_focus_config().get("prioritize_main_entity", True)
    
    def get_extraction_temperature(self) -> float:
        """Get the extraction temperature setting."""
        return self.get_prompt_engineering_config().get("extraction_temperature", 0.1)
    
    def get_priority_entity_types(self) -> list:
        """Get priority entity types for extraction."""
        return self.get_extraction_focus_config().get("priority_entity_types", [])
    
    # NEW: Processing Behavior Configuration Methods
    def get_processing_behavior_config(self) -> Dict[str, Any]:
        """Get processing behavior configuration."""
        return self.config.get("processing_behavior", {})
    
    def get_document_classification_config(self) -> Dict[str, Any]:
        """Get document classification configuration."""
        return self.config.get("document_classification", {})
    
    def get_default_construction_mode(self) -> str:
        """Get default construction mode."""
        return self.get_processing_behavior_config().get("default_construction_mode", "parallel")
    
    def get_default_extraction_mode(self) -> str:
        """Get default extraction mode."""
        return self.get_processing_behavior_config().get("default_extraction_mode", "text")
    
    def get_default_max_workers(self) -> int:
        """Get default max workers."""
        return self.get_processing_behavior_config().get("default_max_workers", 4)
    
    def get_default_dump_page_kgs(self) -> bool:
        """Get default dump page KGs setting."""
        return self.get_processing_behavior_config().get("default_dump_page_kgs", False)
    
    def get_default_transform_final_kg(self) -> bool:
        """Get default transform final KG setting."""
        return self.get_processing_behavior_config().get("default_transform_final_kg", False)
    
    def get_default_run_diagnostics(self) -> bool:
        """Get default run diagnostics setting."""
        return self.get_processing_behavior_config().get("default_run_diagnostics", False)
    
    def get_default_processing_mode(self) -> str:
        """Get default processing mode."""
        return self.get_processing_behavior_config().get("default_processing_mode", "page_based")
    
    def get_default_chunk_size(self) -> int:
        """Get default chunk size."""
        return self.get_processing_behavior_config().get("default_chunk_size", 4000)
    
    def get_default_min_chunk_size(self) -> int:
        """Get default min chunk size."""
        return self.get_processing_behavior_config().get("default_min_chunk_size", 500)
    
    def get_default_chunk_overlap(self) -> int:
        """Get default chunk overlap."""
        return self.get_processing_behavior_config().get("default_chunk_overlap", 200)
    
    def get_default_respect_sentence_boundaries(self) -> bool:
        """Get default respect sentence boundaries setting."""
        return self.get_processing_behavior_config().get("default_respect_sentence_boundaries", True)
    
    def get_default_detect_topic_shifts(self) -> bool:
        """Get default detect topic shifts setting."""
        return self.get_processing_behavior_config().get("default_detect_topic_shifts", True)
    
    def get_default_categories(self) -> List[str]:
        """Get default document categories."""
        return self.get_document_classification_config().get("default_categories", ["financial_teaser", "legal_contract", "press_release", "annual_report"])
    
    # NEW: LLM Model Default Methods
    def get_default_models_for_provider(self, provider: str) -> Dict[str, str]:
        """Get default models for a specific provider."""
        default_models = self.llm_config.get("model_selection", {}).get("default_models", {})
        return default_models.get(provider, {})
    
    def get_default_model_for_provider(self, provider: str, model_type: str) -> str:
        """Get default model for a specific provider and model type."""
        provider_models = self.get_default_models_for_provider(provider)
        return provider_models.get(model_type, "")
    
    def get_azure_model_suffixes(self) -> Dict[str, str]:
        """Get Azure model environment suffixes."""
        azure_suffixes = self.llm_config.get("model_selection", {}).get("azure_model_suffixes", {})
        return azure_suffixes
    
    def get_azure_model_suffix(self, model_type: str) -> str:
        """Get Azure model environment suffix for a specific model type."""
        suffixes = self.get_azure_model_suffixes()
        return suffixes.get(model_type, "")
    
    def reload_config(self):
        """Reload configuration from files."""
        self.config = self._load_config()
        self.llm_config = self._load_llm_config()
        print(f"Configuration reloaded from {self.config_path} and {self.llm_config_path}")
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        def update_nested(current: Dict[str, Any], updates: Dict[str, Any]):
            """Recursively update nested configuration."""
            for key, value in updates.items():
                if key in current and isinstance(current[key], dict) and isinstance(value, dict):
                    update_nested(current[key], value)
                else:
                    current[key] = value
        
        update_nested(self.config, updates)
        print("Configuration updated with new values")
    
    def save_config(self, output_path: Optional[str] = None):
        """Save current configuration to file."""
        if output_path is None:
            output_path = self.config_path
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            print(f"Configuration saved to {output_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def print_config_summary(self):
        """Print a summary of current configuration."""
        print("\n" + "="*60)
        print("EXTRACTION CONFIGURATION SUMMARY")
        print("="*60)
        
        print(f"Quality Control:")
        qc = self.get_quality_control_config()
        print(f"  - Relevance Threshold: {qc.get('relevance_threshold', 0.3)}")
        print(f"  - Quality Filtering: {'Enabled' if qc.get('enable_quality_filtering', True) else 'Disabled'}")
        print(f"  - Over-extraction Prevention: {'Enabled' if qc.get('prevent_over_extraction', True) else 'Disabled'}")
        
        print(f"\nExtraction Focus:")
        ef = self.get_extraction_focus_config()
        print(f"  - Main Entity Prioritization: {'Enabled' if ef.get('prioritize_main_entity', True) else 'Disabled'}")
        print(f"  - Priority Entity Types: {len(ef.get('priority_entity_types', []))} types")
        
        print(f"\nPrompt Engineering:")
        pe = self.get_prompt_engineering_config()
        print(f"  - Focused Prompts: {'Enabled' if pe.get('use_focused_prompts', True) else 'Disabled'}")
        print(f"  - Extraction Temperature: {pe.get('extraction_temperature', 0.1)}")
        
        print(f"\nLLM Configuration:")
        print(f"  - Azure OpenAI Models: {len(self.llm_config.get('azure_openai', {}).get('models', {}))}")
        print(f"  - Vertex AI Models: {len(self.llm_config.get('vertex_ai', {}).get('models', {}))}")
        print(f"  - Default Provider: {self.llm_config.get('model_selection', {}).get('default_provider', 'unknown')}")
        
        print("="*60) 