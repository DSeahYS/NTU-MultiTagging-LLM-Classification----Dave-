import json
import os
from pathlib import Path
from typing import Dict, Any, List

# The Official Online Tagging Schema (30 Categories)
SCHEMA_CATEGORIES = {
    # 1. Bioprinting Modality (Core Technology)
    "extrusion_based": "Bioprinting Modality",
    "droplet_based_inkjet": "Bioprinting Modality",
    "light_based_vat": "Bioprinting Modality",
    "laser_assisted": "Bioprinting Modality",
    "in_situ_bioprinting": "Bioprinting Modality",
    "4d_bioprinting": "Bioprinting Modality",

    # 2. Biomaterial & Bioink Chemistry
    "natural_polymers": "Biomaterial & Bioink Chemistry",
    "synthetic_polymers": "Biomaterial & Bioink Chemistry",
    "dmatrix_bioinks": "Biomaterial & Bioink Chemistry",
    "nanomaterial_composite": "Biomaterial & Bioink Chemistry",
    "hydrogel_rheology": "Biomaterial & Bioink Chemistry",
    "sacrificial_support": "Biomaterial & Bioink Chemistry",

    # 3. Biological & Cellular Systems
    "stem_cells": "Biological & Cellular Systems",
    "primary_cells": "Biological & Cellular Systems",
    "multicellular_co_culture": "Biological & Cellular Systems",
    "spheroids_organoids": "Biological & Cellular Systems",
    "cell_viability_proliferation": "Biological & Cellular Systems",
    "vascularization": "Biological & Cellular Systems",

    # 4. Engineering Applications (Target Tissue/Organ)
    "tissue_engineering": "Engineering Applications",
    "disease_modeling": "Engineering Applications",
    "drug_screening_discovery": "Engineering Applications",
    "anatomical_models": "Engineering Applications",
    "conductive_electronics": "Engineering Applications",

    # 5. Computational, AI & Machine Learning
    "process_calibration": "Computational, AI & Machine Learning",
    "bayesian_optimization": "Computational, AI & Machine Learning",
    "computer_vision_monitoring": "Computational, AI & Machine Learning",
    "deep_learning_models": "Computational, AI & Machine Learning",
    "large_language_models": "Computational, AI & Machine Learning",
    "finite_element_analysis": "Computational, AI & Machine Learning",
    "open_source_software": "Computational, AI & Machine Learning"
}


class ProgressTracker:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.data_dir / "progress.json"
        
        # State: Dictionary mapping filename to a list of tags
        self.classified: Dict[str, List[str]] = {}
        
        self.load()

    def load(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.classified = data.get("classified", {})
            except Exception as e:
                print(f"Error loading progress: {e}")
                self.reset()
        else:
            self.reset()

    def save(self):
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump({
                "classified": self.classified
            }, f, ensure_ascii=False, indent=2)

    def reset(self):
        self.classified = {}
        self.save()

    def update_progress(self, filename: str, categories: List[str]):
        """Update the tags for a specific paper."""
        # Filter out invalid categories just in case
        valid_cats = [cat for cat in categories if cat in SCHEMA_CATEGORIES]
        self.classified[filename] = valid_cats
        self.save()

    def get_stats(self, total_papers: int) -> Dict[str, Any]:
        """Aggregate stats across all 30 categories."""
        counts = {cat: 0 for cat in SCHEMA_CATEGORIES.keys()}
        
        # Count occurrences of each tag across all papers
        for tags in self.classified.values():
            for tag in tags:
                if tag in counts:
                    counts[tag] += 1
                
        processed = len(self.classified)
        
        return {
            "total": total_papers,
            "processed": processed,
            "remaining": max(0, total_papers - processed),
            "counts": counts,
            "pct": round((processed / total_papers * 100) if total_papers > 0 else 0, 1)
        }
