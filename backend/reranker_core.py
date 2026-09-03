import json
import time
from typing import List
import openai
from progress_tracker import SCHEMA_CATEGORIES

SYSTEM_PROMPT = """You are an automated, high-precision Document Tagger serving as the orchestrator for a Large Knowledge Model (LKM) specialized in Additive Manufacturing and Bioprinting. Your objective is to classify a provided research paper (represented by its abstract and introduction) into one or more of the 30 official schema categories.

### THE CATEGORIES & DEFINITIONS

1. Bioprinting Modality (Core Technology)
- extrusion_based: Pneumatic, piston, or screw-driven deposition
- droplet_based_inkjet: Piezoelectric, thermal, or valve-based jetting
- light_based_vat: DLP, SLA, Stereolithography, Two-photon polymerization
- laser_assisted: Laser-induced forward transfer
- in_situ_bioprinting: Printing directly onto or inside a patient/tissue defect
- 4d_bioprinting: Smart/stimuli-responsive materials that change shape post-print

2. Biomaterial & Bioink Chemistry
- natural_polymers: Gelatin, Alginate, Collagen, Hyaluronic acid, Chitosan
- synthetic_polymers: Pluronic F127, PCL, PEG, PLA
- dmatrix_bioinks: Decellularized extracellular matrix / dECM
- nanomaterial_composite: Inks containing graphene, carbon nanotubes, nanocellulose
- hydrogel_rheology: Papers focusing on viscosity, shear-thinning, or crosslinking mechanics
- sacrificial_support: Temporary bath or matrix materials used for embedding/overhangs

3. Biological & Cellular Systems
- stem_cells: iPSCs, MSCs, embryonic stem cells
- primary_cells: HUVECs, fibroblasts, macrophages, hepatocytes
- multicellular_co_culture: Studies utilizing or mixing more than one cell type
- spheroids_organoids: Self-assembled high-density cellular configurations
- cell_viability_proliferation: Focusing primarily on biological survival post-print
- vascularization: Angiogenesis, channel fabrication, perfusable networks

4. Engineering Applications (Target Tissue/Organ)
- tissue_engineering: General scaffold fabrication for regenerative medicine
- disease_modeling: In vitro tumor microenvironments, pathological tissue replication
- drug_screening_discovery: High-throughput testing platforms for pharmaceuticals
- anatomical_models: Ear, bone grafts, structural implants matching patient data
- conductive_electronics: Biosensors, smart patches, cellular electronics

5. Computational, AI & Machine Learning
- process_calibration: Trial-and-error optimization, statistical Design of Experiments
- bayesian_optimization: Surrogate modeling for parameter space searching
- computer_vision_monitoring: Cameras, anomaly detection, real-time closed-loop control
- deep_learning_models: Neural networks, ResNet, YOLO, image regression
- large_language_models: RAG, LKM, semantic search, literature retrieval
- finite_element_analysis: Fluid dynamics simulation, bioink flow mechanics
- open_source_software: Custom slicing tools, algorithmic toolpaths, code repositories

### RULES FOR TAGGING
- YOU MUST EXTRACT EVERY SINGLE APPLICABLE TAG from the text. A paper usually covers multiple domains (e.g., the printing method + the biomaterial + the cells used + the target application).
- Do NOT stop at one tag. You must thoroughly analyze the paper and return a list of ALL relevant tags (e.g., `["extrusion_based", "natural_polymers", "stem_cells", "tissue_engineering"]`).
- DO NOT invent new tags. Only use exact matches from the list above.
- Be precise. If they don't explicitly focus on machine learning, don't tag `deep_learning_models`.

### STRICT OUTPUT FORMAT
CRITICAL: You MUST NOT output any explanations, reasoning, or step-by-step analysis.
Do not use lists. Do not say "Let me analyze".
Your ENTIRE response must be just the JSON array.
Example output:
["extrusion_based", "natural_polymers", "stem_cells", "tissue_engineering"]
"""

def create_client(api_key: str) -> openai.OpenAI:
    return openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        timeout=60.0,
    )

def _parse_llm_output(raw_text: str) -> List[str]:
    """Parse the LLM response into a list of tags and filter against the schema."""
    import re
    try:
        print(f"RAW LLM OUTPUT:\n{raw_text}\n---".encode('utf-8', 'ignore').decode('utf-8', 'ignore'))
    except Exception:
        pass
    
    # Strip markdown if present
    json_match = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', raw_text)
    if json_match:
        raw_text = json_match.group(1)
        
    valid_keys = list(SCHEMA_CATEGORIES.keys())
    valid_keys_lower = [k.lower() for k in valid_keys]
        
    try:
        tags = json.loads(raw_text.strip())
        if isinstance(tags, list):
            # Filter to only valid schema categories (case-insensitive)
            matched = []
            for t in tags:
                if t.lower() in valid_keys_lower:
                    idx = valid_keys_lower.index(t.lower())
                    matched.append(valid_keys[idx])
            if matched:
                return matched
    except json.JSONDecodeError:
        pass
        
    # Fallback: extract any string matching a category
    found = []
    raw_lower = raw_text.lower()
    for i, cat_lower in enumerate(valid_keys_lower):
        if cat_lower in raw_lower:
            found.append(valid_keys[i])
    return found

def classify_paper(client: openai.OpenAI, paper_text: str, model: str) -> List[str]:
    """Classify a paper's text using LLM, returning a list of categories."""
    # We only send the first ~10000 chars to save tokens (Abstract, Intro, Methods)
    truncated_text = paper_text[:10000]
    
    fallback_models = [
        "nvidia/nemotron-3-ultra-550b-a55b:free",
        "google/gemma-4-26b-a4b-it:free",
        "openai/gpt-oss-120b:free",
    ]
    
    models_to_try = [model] + [m for m in fallback_models if m != model]
    
    for current_model in models_to_try:
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": f"Analyze this paper text and assign tags:\n\n{truncated_text}"},
                    ],
                    temperature=0.0,
                    max_tokens=2000,
                    extra_headers={
                        "HTTP-Referer": "https://bioprint-paper-sorter.local",
                        "X-Title": "BioPrint Paper Sorter Dashboard",
                    },
                )
                raw_content = response.choices[0].message.content
                if not raw_content:
                    raise ValueError("Empty or None content from LLM")
                
                raw = raw_content.strip()
                tags = _parse_llm_output(raw)
                
                # If we got at least one valid tag, return it
                if tags:
                    return tags
                    
            except openai.RateLimitError:
                time.sleep(3.0)
                break
            except Exception as e:
                print(f"LLM Error for {current_model}: {e}")
                if attempt < 1:
                    time.sleep(1)
                    continue
                break
                
    # Ultimate fallback if everything fails
    return ["tissue_engineering"]
