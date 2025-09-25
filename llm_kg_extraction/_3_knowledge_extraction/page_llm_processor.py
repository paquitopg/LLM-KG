import json
import base64
from typing import Dict, Any, Optional, List

# Assuming LLM wrappers and Ontology loader are structured as previously discussed
# Adjust these import paths based on your final project structure
from llm_integrations.base_llm_wrapper import BaseLLMWrapper
from llm_integrations.azure_llm import AzureLLM # For type checking if needed for specific params
from llm_integrations.vertex_llm import VertexLLM # For type checking

from loaders.ontology_loader import PEKGOntology # Assuming PEKGOntology or a base class

# Semantic chunking configuration
config = {
    "chunk_size": 4000,
    "min_chunk_size": 500, 
    "chunk_overlap": 200,
    "respect_sentence_boundaries": True,
    "detect_topic_shifts": True
}

class PageLLMProcessor:
    """
    Processes a single document page (text or multimodal) using an LLM
    to extract a knowledge graph.
    """

    def __init__(self,
                 llm_client: BaseLLMWrapper,
                 ontology: PEKGOntology, 
                 extraction_mode: str,
                 use_ontology: bool = True): 
        """
        Initializes the PageLLMProcessor.

        Args:
            llm_client (BaseLLMWrapper): An instance of an LLM client wrapper.
            ontology (PEKGOntology): An instance of the ontology loader.
        """
        self.llm_client = llm_client
        self.ontology = ontology
        self.temperature = 0.1
        self.extraction_mode = extraction_mode
        self.use_ontology = use_ontology

    def _build_unified_extraction_prompt(self,
                                       page_text: str,
                                       page_number: str,
                                       document_context_info: Dict[str, Any],
                                       previous_graph_context: Optional[Dict[str, Any]] = None,
                                       construction_mode: str = "iterative",
                                       is_multimodal: bool = False
                                      ) -> str:
        """
        Unified prompt builder for both text and multimodal extraction.
        Eliminates code duplication and improves extraction quality.
        """
        if self.use_ontology:
            ontology_desc = self.ontology.format_for_prompt()
        else:
            ontology_desc = "No ontology is being used for this extraction. Entities and relations will be extracted based on general relevance."
            
        previous_graph_json_for_prompt = "{}"
        if construction_mode == "iterative" and previous_graph_context and previous_graph_context.get("entities"):
            previous_graph_json_for_prompt = json.dumps(previous_graph_context)

        # Extract key context information
        doc_type = document_context_info.get("identified_document_type", "document")
        main_entity_name = document_context_info.get("main_entity", "")
        document_summary = document_context_info.get("document_summary", "No summary provided")

        # Build focused context introduction
        context_specific_intro = self._build_focused_context_intro(
            doc_type, main_entity_name, document_summary
        )

        # Build extraction focus instructions
        extraction_focus = self._build_extraction_focus_instructions(
            doc_type, main_entity_name, is_multimodal
        )

        # Build the unified prompt
        prompt = f"""
        You are a specialized financial information extraction system focused on PEKG (Private Equity Knowledge Graph) entities.

        {context_specific_intro}

        {extraction_focus}

        **CRITICAL EXTRACTION RULES:**
        1. **BALANCED EXTRACTION**: Extract entities and relations explicitly supported by the page content. Include the main entity when relevant, but also capture relations between non-main entities (peers, sub-entities, metrics, people, products) when they are stated.
        2. **NO FORCED HUB**: Do NOT add or infer relations to the main entity unless they are explicitly stated on the page. Prefer forming local subgraphs between co-mentioned entities over connecting everything to the main entity.
        3. **ONTOLOGY COMPLIANCE**: Only extract entities and relations that match the PEKG ontology exactly
        4. **RELEVANCE FILTERING**: Skip generic information, industry overviews, or details not directly relevant to the page’s concrete facts
        5. **QUALITY OVER QUANTITY**: Extract fewer, high-quality entities rather than many irrelevant ones
        """

        # Add previous context if iterative mode
        if construction_mode == "iterative" and previous_graph_context and previous_graph_context.get("entities"):
            prompt += f"""
            **CONTEXT INTEGRATION:**
            Use the previous knowledge graph context for ID consistency. DO NOT re-extract entities from previous context unless they appear again on the current page.

            Previous knowledge graph context:
            {previous_graph_json_for_prompt}
            """

        # Add ontology information
        if self.use_ontology:
            prompt += f"""
            **PEKG ONTOLOGY (STRICT COMPLIANCE REQUIRED):**
            {ontology_desc}

            **ONLY extract entities and relations that EXACTLY match these types.**
            """
        
        # Add content to analyze
        content_section = self._build_content_analysis_section(page_text, page_number, is_multimodal)
        prompt += f"\n{content_section}"

        # Add specific extraction instructions
        prompt += self._build_specific_extraction_instructions(is_multimodal)

        # Add output format
        prompt += self._build_output_format_instructions()

        return prompt

    def _build_focused_context_intro(self, doc_type: str, main_entity_name: str, document_summary: str) -> str:
        """Build focused context introduction emphasizing the main entity."""
        
        if doc_type == "financial_teaser":
            return f"""
            **DOCUMENT CONTEXT:**
            This is a '{doc_type}' document focused on: **{main_entity_name}**
            
            **KEY FOCUS AREAS:**
            - Company: {main_entity_name} (the primary subject)
            - Financial metrics and performance data
            - Business model and key offerings
            - Market position and competitive advantages
            - Growth strategy and investment highlights
            
            **AVOID EXTRACTING:**
            - Generic industry information not specific to {main_entity_name}
            - Advisory firm details (unless directly managing {main_entity_name})
            - Project codenames or transaction codes
            - General market overviews
            """
        else:
            return f"""
            **DOCUMENT CONTEXT:**
            Document Type: {doc_type}
            Primary Subject: **{main_entity_name}**
            
            **FOCUS ON:**
            - Information directly related to {main_entity_name}
            - Specific facts, metrics, and relations involving {main_entity_name}
            - Business-relevant details about {main_entity_name}'s operations
            """

    def _build_extraction_focus_instructions(self, doc_type: str, main_entity_name: str, is_multimodal: bool) -> str:
        """Build instructions that focus extraction on relevant information while avoiding hub-and-spoke graphs."""
        
        mode_text = "text and visual content" if is_multimodal else "text content"
        
        return f"""
        **EXTRACTION TASK:**
        Extract ONLY entities and relations from the current page {mode_text} that are:
        1. **DIRECTLY SUPPORTED BY THE PAGE CONTENT** (text or visuals)
        2. **EXACTLY MATCHING** the PEKG ontology types
        3. **SPECIFIC AND ACTIONABLE** rather than generic or descriptive
        4. **NOT LIMITED TO THE MAIN ENTITY**. Capture relations among non-main entities when they are explicitly stated (e.g., metric-to-period, person-to-role, product-to-technology, partner-to-partner).

        **PRIORITIZE:**
        - Explicit relations among entities that co-occur on the page (including non-main-entity to non-main-entity links)
        - Financial metrics, KPIs, and their associations (periods, units, contexts)
        - Roles and affiliations among people, organizations, and products/services
        - Partnerships and competitive relations stated on the page

        **AVOID HUB BIAS:**
        - Prefer forming small local clusters/subgraphs among co-mentioned entities

        **SKIP:**
        - Generic industry descriptions
        - Vague or non-specific information
        """

    def _build_content_analysis_section(self, page_text: str, page_number: str, is_multimodal: bool) -> str:
        """Build the content analysis section."""
        
        if is_multimodal:
            return f"""
            **CURRENT PAGE {page_number} CONTENT TO ANALYZE:**
            
            Text Content:
            \"\"\"{page_text}\"\"\"
            
            Visual Content: [Image data provided separately]
            
            **ANALYZE BOTH text and visual elements together for comprehensive understanding.**
            """
        else:
            return f"""
            **CURRENT PAGE TEXT TO ANALYZE:**
            \"\"\"{page_text}\"\"\"
            """

    def _build_specific_extraction_instructions(self, is_multimodal: bool) -> str:
        """Build specific extraction instructions."""
        
        return f"""
        **EXTRACTION INSTRUCTIONS:**
        1. **IDENTIFY RELEVANT ENTITIES**: From the current page content, identify entities that:
           - Match PEKG ontology types exactly
           - Are directly related to the main entity
           - Have specific, extractable attributes
        
        2. **EXTRACT ATTRIBUTES**: For each relevant entity, extract ONLY attributes that:
           - Are explicitly stated in the content
           - Match the ontology specification
           - Are specific and factual (avoid interpretations)
        
        3. **CREATE RELATIONSHIPS**: Build relations that:
           - Connect entities based on explicit content
           - Use only PEKG relationship types
           - Are supported by clear evidence in the text
        
        4. **QUALITY CONTROL**: Before including an entity/relationship, ask:
           - Is this directly relevant to the main entity?
           - Does it match the ontology exactly?
           - Is the information specific and factual?
           - Would this be useful for business analysis?
        
        5. **AVOID OVER-EXTRACTION**: Do not extract:
           - Generic industry information
           - Unrelated company mentions
           - Vague or non-specific details
        """

    def _build_output_format_instructions(self) -> str:
        """Build output format instructions that discourage star-shaped graphs."""
        
        return f"""
        **OUTPUT FORMAT:**
        Return ONLY a valid JSON object with this structure:
        {{
            "entities": [
                {{"id": "e1", "type": "pekg:Company", "name": "CompanyName", "industry": "Software"}},
                {{"id": "e2", "type": "pekg:FinancialMetric", "metricName": "FY23 Revenue", "valueString": "€16.0m", "DateOrPeriod": "FY23"}}
            ],
            "relations": [
                {{"source": "e1", "target": "e2", "type": "pekg:reportsFinancialMetric"}}
            ]
        }}
        
        **CRITICAL:**
        - Include ONLY entities and relations from the CURRENT PAGE
        - Connect entities ONLY when the relationship is explicitly supported by the page content
        - Avoid making the main entity a hub unless the page explicitly creates those connections
        - Use ONLY PEKG ontology types
        - Focus on quality and relevance over quantity
        - No commentary, explanations, or markdown formatting
        """

    def _build_text_extraction_prompt(self,
                                      page_text: str,
                                      document_context_info: Dict[str, Any],
                                      previous_graph_context: Optional[Dict[str, Any]] = None,
                                      construction_mode: str = "iterative" 
                                     ) -> str:
        """
        Builds the detailed prompt for text-based knowledge graph extraction.
        Now uses the unified prompt builder for consistency.
        """
        return self._build_unified_extraction_prompt(
            page_text=page_text,
            page_number="",
            document_context_info=document_context_info,
            previous_graph_context=previous_graph_context,
            construction_mode=construction_mode,
            is_multimodal=False
        )

    def _build_multimodal_extraction_prompt(self,
                                           page_text: str,
                                           page_number: str,
                                           document_context_info: Dict[str, Any],
                                           previous_graph_context: Optional[Dict[str, Any]] = None,
                                           construction_mode: str = "iterative"
                                          ) -> str:
        """
        Builds the detailed prompt for multimodal (text + image) KG extraction.
        Now uses the unified prompt builder for consistency.
        """
        return self._build_unified_extraction_prompt(
            page_text=page_text,
            page_number=page_number,
            document_context_info=document_context_info,
            previous_graph_context=previous_graph_context,
            construction_mode=construction_mode,
            is_multimodal=True
        )


    def process_page(self,
                     page_data: Dict[str, Any],
                     document_context_info: Dict[str, Any],
                     construction_mode: str = "iterative",
                     previous_graph_context: Optional[Dict[str, Any]] = None
                    ) -> Dict[str, List[Any]]:
        """
        Method to process a single page to extract a knowledge graph.
        Handles both text and multimodal extraction based on self.extraction_mode.

        Args:
            page_data (Dict[str, Any]): Dict containing 'page_number', 'text' and optionally 'image_base64'.
            document_context_info (Dict[str, Any]): Contextual info about the document.
            construction_mode (str): How the overall KG is being built ("iterative", "parallel", "onego").
            previous_graph_context (Optional[Dict[str, Any]]): KG from previous pages.

        Returns:
            Dict[str, List[Any]]: The extracted knowledge graph for the page (entities and relations).
                                  Returns {"entities": [], "relations": []} on error or no content.
        """
        page_number = page_data.get("page_number", "N/A")
        page_text = page_data.get("text", "")
        page_image_base64 = page_data.get("image_base64")
        llm_response_content: Optional[str] = None
        
        # Validate input based on extraction mode
        if self.extraction_mode == "text":
            if not page_text.strip():
                print(f"Skipping page {page_number} for text analysis as it has no text content.")
                return {"entities": [], "relations": []}
        elif self.extraction_mode == "multimodal":
            if not page_image_base64:
                print(f"Warning: image_base64 not found for page {page_number} in multimodal mode.")
                if not page_text.strip():
                    print(f"No text content either. Skipping page {page_number}.")
                    return {"entities": [], "relations": []}
                print(f"Falling back to text-only processing for page {page_number}.")
                # Will process as text-only below
        
        try:
            # Determine processing approach
            use_multimodal = (self.extraction_mode == "multimodal" and page_image_base64)
            
            if use_multimodal:
                prompt_str = self._build_multimodal_extraction_prompt(
                    page_text=page_text,
                    page_number=page_number,
                    document_context_info=document_context_info,
                    previous_graph_context=previous_graph_context,
                    construction_mode=construction_mode
                )
                
                # Prepare multimodal input for LLM client
                if isinstance(self.llm_client, AzureLLM):
                    user_content_parts = [
                        {"type": "text", "text": prompt_str},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_image_base64}"}}
                    ]
                    messages = [
                        {"role": "system", "content": "You are a financial KG extraction assistant for multimodal inputs, outputting JSON."},
                        {"role": "user", "content": user_content_parts}
                    ]
                    llm_response_content = self.llm_client.chat_completion(
                        messages=messages,
                        temperature=self.temperature
                    )

                elif isinstance(self.llm_client, VertexLLM):
                    # Fix: Add type check to ensure page_image_base64 is not None
                    if page_image_base64 is not None:
                        image_bytes = base64.b64decode(page_image_base64)
                        vertex_parts = [
                            {'text': prompt_str},
                            {'inline_data': {'mime_type': 'image/png', 'data': image_bytes}}
                        ]
                        llm_response_content = self.llm_client.generate_content(
                            prompt=vertex_parts,
                            temperature=self.temperature,
                            response_mime_type="application/json"
                        )
                    else:
                        print(f"Warning: page_image_base64 is None for VertexLLM processing on page {page_number}")
                        use_multimodal = False
                else:
                    print(f"Multimodal extraction not configured for LLM type: {self.llm_client.__class__.__name__}. Falling back to text-only.")
                    use_multimodal = False
            
            # Text-only processing (either by design or fallback)
            if not use_multimodal:
                prompt_str = self._build_text_extraction_prompt(
                    page_text=page_text,
                    document_context_info=document_context_info,
                    previous_graph_context=previous_graph_context,
                    construction_mode=construction_mode
                )
                
                if isinstance(self.llm_client, AzureLLM):
                    messages = [
                        {"role": "system", "content": "You are a financial information extraction assistant designed to output JSON."},
                        {"role": "user", "content": prompt_str}
                    ]
                    llm_response_content = self.llm_client.chat_completion(
                        messages=messages,
                        temperature=self.temperature
                    )
                elif isinstance(self.llm_client, VertexLLM):
                    llm_response_content = self.llm_client.generate_content(
                        prompt=prompt_str,
                        temperature=self.temperature,
                        response_mime_type="application/json"
                    )
                else: # Generic fallback
                    messages = [
                        {"role": "system", "content": "You are an information extraction assistant designed to output JSON."},
                        {"role": "user", "content": prompt_str}
                    ]
                    llm_response_content = self.llm_client.chat_completion(messages=messages, temperature=self.temperature)

            # Parse response
            if not llm_response_content:
                print(f"Warning: Empty content received from LLM for page {page_number}. Returning empty graph.")
                return {"entities": [], "relations": []}

            # Clean potential markdown ```json
            clean_content = llm_response_content.strip()
            if clean_content.startswith("```json"):
                clean_content = clean_content.lstrip("```json").rstrip("```").strip()
            elif clean_content.startswith("```"):
                 clean_content = clean_content.lstrip("```").rstrip("```").strip()
                 if clean_content.startswith("json"):
                    clean_content = clean_content.lstrip("json").strip()
            
            if not clean_content:
                print(f"Warning: Content became empty after stripping markdown for page {page_number}. Raw: '{llm_response_content[:100]}...'")
                return {"entities": [], "relations": []}

            result = json.loads(clean_content)
            
            # Add page_number to result for tracking
            result["page_number"] = page_number

            return result

        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response from LLM ({self.llm_client.__class__.__name__}) for page {page_number}: {e}")
            print(f"Model: {self.llm_client.model_name}")
            raw_content_snippet = llm_response_content[:500] if llm_response_content else "N/A"
            print(f"Raw LLM Content that failed parsing (first 500 chars):\n---\n{raw_content_snippet}\n---")
            return {"entities": [], "relations": []}
        except Exception as e:
            if "response.text" in str(e) and "valid Part" in str(e) and isinstance(self.llm_client, VertexLLM):
                 print(f"ValueError accessing response.text for page {page_number} (Vertex), likely due to blocked content or no parts. Error: {e}")
                 return {"entities": [], "relations": []}
            print(f"Error during LLM page processing ({self.llm_client.__class__.__name__}, page {page_number}): {type(e).__name__} - {e}")
            print(f"Model: {self.llm_client.model_name}")
            raw_content_snippet = llm_response_content[:500] if llm_response_content else "N/A"
            print(f"Raw LLM Content (if available, first 500 chars):\n---\n{raw_content_snippet}\n---")
            return {"entities": [], "relations": []}
