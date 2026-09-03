"""
PDF text extraction module for the RAG Benchmark Q&A Generator.
Extracts and cleans text from bioprinting research PDFs, with
enhanced figure/table caption extraction for visual data conversion.
"""

import os
import re
from pathlib import Path
from typing import Optional

import pypdf
import pypdf._font
from pypdf.generic import DictionaryObject

# Monkeypatch pypdf to ignore "More than one /FontFile found" errors in malformed PDFs
original_parse_font_descriptor = pypdf._font.Font._parse_font_descriptor

def patched_parse_font_descriptor(font_descriptor_obj, *args, **kwargs):
    keys_found = []
    for key in ["/FontFile", "/FontFile2", "/FontFile3"]:
        if key in font_descriptor_obj:
            keys_found.append(key)
    if len(keys_found) > 1:
        new_obj = DictionaryObject()
        for k, v in font_descriptor_obj.items():
            if k in keys_found and k != keys_found[0]:
                continue
            new_obj[k] = v
        font_descriptor_obj = new_obj
    return original_parse_font_descriptor(font_descriptor_obj, *args, **kwargs)

pypdf._font.Font._parse_font_descriptor = staticmethod(patched_parse_font_descriptor)


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF file page by page.
    
    Returns a dict with:
        - filename: basename of the PDF
        - filepath: full path
        - page_count: number of pages
        - pages: list of {page_num, text} dicts
        - full_text: concatenated cleaned text
        - doi: extracted DOI if found
        - title_guess: best guess at paper title
    """
    path = Path(pdf_path)
    reader = pypdf.PdfReader(str(path))
    
    pages = []
    full_text_parts = []
    
    for idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = _clean_page_text(raw_text)
        pages.append({
            "page_num": idx + 1,
            "text": cleaned,
        })
        full_text_parts.append(cleaned)
    
    full_text = "\n\n".join(full_text_parts)
    doi = _extract_doi(full_text)
    title_guess = _guess_title(pages[0]["text"] if pages else "", path.stem)
    
    return {
        "filename": path.name,
        "filepath": str(path),
        "page_count": len(reader.pages),
        "pages": pages,
        "full_text": full_text,
        "doi": doi,
        "title_guess": title_guess,
    }


def _clean_page_text(text: str) -> str:
    """Clean extracted page text by removing common artifacts."""
    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove common header/footer patterns
    text = re.sub(r'(?m)^Downloaded from .+$', '', text)
    text = re.sub(r'(?m)^©\s*\d{4}.+$', '', text)
    # Remove page numbers standing alone
    text = re.sub(r'(?m)^\s*\d{1,3}\s*$', '', text)
    # Collapse multiple spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def _extract_doi(text: str) -> Optional[str]:
    """Try to extract a DOI from the text."""
    doi_pattern = r'(?:doi[:\s]*|https?://doi\.org/)?(10\.\d{4,}/[^\s,;"\'\]]+)'
    match = re.search(doi_pattern, text, re.IGNORECASE)
    if match:
        doi = match.group(1).rstrip('.')
        return doi
    return None


def _guess_title(first_page_text: str, filename_stem: str) -> str:
    """
    Best-effort title extraction from first page text.
    Falls back to cleaned filename.
    """
    lines = [l.strip() for l in first_page_text.split('\n') if l.strip()]
    
    # Heuristic: title is usually one of the first non-short lines
    for line in lines[:10]:
        # Skip lines that look like journal names, dates, DOIs
        if any(skip in line.lower() for skip in [
            'journal', 'volume', 'doi:', 'received', 'accepted',
            'published', '©', 'http', 'university', 'department',
            'abstract', 'keywords'
        ]):
            continue
        # Title lines tend to be 20-200 chars and not all-caps short
        if 20 <= len(line) <= 250:
            return line
    
    # Fallback: clean the filename
    title = filename_stem.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
    return title[:150]


def scan_pdf_folder(folder_path: str) -> list[dict]:
    """
    Scan a folder for PDF files and return basic info (no text extraction).
    """
    folder = Path(folder_path)
    pdfs = sorted(folder.glob("*.pdf"), key=lambda p: p.stat().st_mtime)
    
    results = []
    for pdf in pdfs:
        results.append({
            "filename": pdf.name,
            "filepath": str(pdf),
            "size_bytes": pdf.stat().st_size,
        })
    
    return results


# ── Figure/Table Caption Extraction ──────────────────────────────────

def _extract_figure_captions(text: str) -> list[str]:
    """
    Extract figure captions and their descriptive text from the full text.
    
    Matches patterns like:
      - "Fig. 1. Caption text..."
      - "Figure 1: Caption text..."  
      - "Fig 2a-c: Description..."
      - "FIGURE 3. Description..."
    """
    captions = []
    
    # Pattern for figure captions (Fig./Figure followed by number and caption text)
    # Captures the caption up to the next figure/table header, section header, or double newline
    fig_patterns = [
        # "Fig. 1." or "Fig. 1:" or "Figure 1." style
        r'(?:Fig(?:ure)?\.?\s*\d+[a-z]?(?:[-–]\w+)?)\s*[.:]\s*(.+?)(?=\n\s*\n|(?:Fig(?:ure)?\.?\s*\d)|(?:Table\s*\d)|$)',
        # Multi-line caption blocks after "Fig. X"
        r'(?:Fig(?:ure)?\.?\s*\d+[a-z]?(?:[-–]\w+)?)\s*[.:]\s*(.+?)(?=\n\s*(?:[A-Z][a-z]+\s){2})',
    ]
    
    for pattern in fig_patterns:
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        for match in matches:
            caption = re.sub(r'\s+', ' ', match.strip())
            if len(caption) > 20:  # Skip very short fragments
                captions.append(f"[Figure Description] {caption}")
    
    return captions


def _extract_table_captions(text: str) -> list[str]:
    """
    Extract table captions and headers from the full text.
    
    Matches patterns like:
      - "Table 1. Summary of..."
      - "Table 2: Mechanical properties..."
    """
    captions = []
    
    table_pattern = r'(?:Table\s*\d+)\s*[.:]\s*(.+?)(?=\n\s*\n|(?:Table\s*\d)|(?:Fig(?:ure)?\.?\s*\d)|$)'
    matches = re.findall(table_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        caption = re.sub(r'\s+', ' ', match.strip())
        if len(caption) > 15:
            captions.append(f"[Table Description] {caption}")
    
    return captions


def _extract_inline_figure_data(text: str) -> list[str]:
    """
    Extract sentences that contain inline figure/table references with data.
    
    These sentences often contain crucial quantitative data that accompanies
    figures the LLM cannot see, e.g.:
      "As shown in Fig. 3a, the compressive modulus increased from 12 to 45 kPa"
    """
    data_sentences = []
    
    # Find sentences containing figure references with quantitative data
    # Look for sentences with "Fig." or "Figure" or "Table" that also contain numbers
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        # Must reference a figure/table
        has_fig_ref = bool(re.search(r'(?:Fig(?:ure)?\.?\s*\d|Table\s*\d)', sentence, re.IGNORECASE))
        # Must contain quantitative data (numbers with units or percentages)
        has_data = bool(re.search(r'\d+\.?\d*\s*(?:%|mm|μm|nm|kPa|MPa|GPa|°C|mL|μL|mg|μg|g/|mol|wt|vol|cells|Pa·s|mW|Hz|N/m|kN|mN)', sentence, re.IGNORECASE))
        
        if has_fig_ref and has_data and len(sentence) > 30:
            # Clean the figure reference to make it more generic
            cleaned = re.sub(r'\(?(?:as\s+)?(?:shown|depicted|illustrated|displayed|presented)\s+in\s+', '(described in ', sentence, flags=re.IGNORECASE)
            data_sentences.append(f"[Visual Data] {cleaned}")
    
    return data_sentences


def _build_visual_data_section(text: str) -> str:
    """
    Build a consolidated section of all visual data descriptions
    extracted from figure captions, table captions, and inline references.
    """
    figure_captions = _extract_figure_captions(text)
    table_captions = _extract_table_captions(text)
    inline_data = _extract_inline_figure_data(text)
    
    all_visual = figure_captions + table_captions + inline_data
    
    if not all_visual:
        return ""
    
    # Deduplicate similar entries (within 80% overlap)
    unique_entries = []
    for entry in all_visual:
        is_duplicate = False
        entry_lower = entry.lower()
        for existing in unique_entries:
            if entry_lower in existing.lower() or existing.lower() in entry_lower:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_entries.append(entry)
    
    section = "\n\n## Visual Data Descriptions\n"
    section += "The following are textual descriptions of data presented in figures, tables, and micrographs:\n\n"
    for entry in unique_entries:
        section += f"- {entry}\n"
    
    return section


def get_text_for_qa(pdf_data: dict, max_chars: int = 25000) -> str:
    """
    Prepare text from a PDF for Q&A generation.
    Truncates to max_chars to fit within LLM context windows.
    Prioritizes abstract + introduction + methods + results + conclusion.
    
    For RAG benchmarking: appends a Visual Data Descriptions section
    consolidating all figure/table captions and their data.
    """
    full_text = pdf_data["full_text"]
    
    # Build visual data section from the full text BEFORE truncation
    visual_section = _build_visual_data_section(full_text)
    
    # Reserve space for visual section
    visual_len = len(visual_section)
    text_budget = max_chars - visual_len
    
    if text_budget < 5000:
        # If visual section is very large, cap it and give more to main text
        text_budget = max_chars - 3000
        visual_section = visual_section[:3000]
    
    if len(full_text) <= text_budget:
        main_text = full_text
    else:
        # Try to extract key sections
        sections = _extract_key_sections(full_text)
        
        if sections:
            prioritized = "\n\n".join(sections)
            if len(prioritized) <= text_budget:
                main_text = prioritized
            else:
                main_text = prioritized[:text_budget]
        else:
            # Fallback: just truncate
            main_text = full_text[:text_budget]
    
    # Append visual data section
    return main_text + visual_section


def _extract_key_sections(text: str) -> list[str]:
    """
    Try to extract key sections (abstract, intro, methods, results, conclusion).
    """
    section_patterns = [
        r'(?i)(abstract\s*\n[\s\S]*?)(?=\n\s*(?:1\.|introduction|keywords))',
        r'(?i)((?:1\.\s*)?introduction[\s\S]*?)(?=\n\s*(?:2\.|materials|methods|experimental))',
        r'(?i)((?:2\.\s*)?(?:materials and methods|methods|experimental)[\s\S]*?)(?=\n\s*(?:3\.|results))',
        r'(?i)((?:3\.\s*)?(?:results|results and discussion)[\s\S]*?)(?=\n\s*(?:4\.|conclusion|discussion|acknowledgment))',
        r'(?i)((?:4\.\s*)?(?:conclusion|conclusions|summary)[\s\S]*?)(?=\n\s*(?:acknowledgment|references|funding|conflict))',
    ]
    
    sections = []
    for pattern in section_patterns:
        match = re.search(pattern, text)
        if match:
            section_text = match.group(1).strip()
            if len(section_text) > 50:
                sections.append(section_text)
    
    return sections
