import pdfplumber
import pytesseract
import re
import json
import logging
import os
import time
import unicodedata
from shutil import which
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageOps
from pytesseract import Output

logger = logging.getLogger(__name__)

# ── Tesseract path resolution (Windows) ──────────────────────────────────────
# Checks env var, PATH, and common install locations in one pass.
def _resolve_tesseract() -> str | None:
    candidates = [
        os.getenv("TESSERACT_CMD"),
        which("tesseract"),
        *[
            os.path.join(base, "Tesseract-OCR", "tesseract.exe")
            for key in ("LOCALAPPDATA", "ProgramFiles", "ProgramFiles(x86)")
            if (base := os.getenv(key))
        ],
        os.path.join(os.getenv("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
    ]
    return next((c for c in candidates if c and os.path.isfile(c)), None)

if cmd := _resolve_tesseract():
    pytesseract.pytesseract.tesseract_cmd = cmd


class OrangeExtractor:
    """
    Hybrid PDF extraction — strategy per document type:

    | Document       | PDF type | Strategy                          |
    |----------------|----------|-----------------------------------|
    | Fiche          | Digital  | pdfplumber + regex + table        |
    | Autorisation   | Scanned  | OCR → regex (ref) + LLM (address) |
    | Mandat         | Scanned  | OCR → LLM entirely                |

    Fiche never uses the LLM (fully structured form — regex is instant).
    LLM only runs for prose fields where layout varies too much for regex.
    """

    # ── Compiled patterns ─────────────────────────────────────────────────────
    FICHE_PERMIT = re.compile(r"\b((?:PC|PA|DP)\d{5,}[A-Z0-9]*)\b", re.I)
    # Bare ref without prefix — some consultants omit PC/PA/DP on the fiche.
    # Format: DDD DDD DD X DDDD or DDDDDDDDXDDDD (13 chars after stripping).
    # Only matched on the "Référence Autorisation Urbanisme" label line.
    FICHE_PERMIT_BARE = re.compile(
        r"r[eé]f[eé]rence\s+autorisation\s+urbanisme\s*[:\-]?\s*([0-9][A-Z0-9\s\-/]{10,17})",
        re.I,
    )
    FICHE_DLPI   = re.compile(
        r"DATE\s+DE\s+LIVRAISON\s+DU\s+PROJET\s*[:\-]?\s*(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        re.I,
    )
    # ── Permit ref patterns ──────────────────────────────────────────────────
    # Primary: strict French format PP DDD DDD DD X DDDD
    # Suffix first char [A-Z0-9]: accepts digit because OCR misreads letters
    # as digits — most common: Z→2, B→8, O→0. _clean_permit_ref corrects this.
    AU_PERMIT = re.compile(
        r"(?<![A-Z0-9])"
        r"((?:PC|PA|DP|CU)"
        r"[\s\-/]*\d{2,3}"           # dept: 2 or 3 digits
        r"[\s\-/]*\d{3}"             # commune
        r"[\s\-/]*\d{2}"             # year
        r"[\s\-/]*[A-Z0-9]\d{4})"   # suffix: letter OR digit (OCR: Z→2)
        r"(?![A-Z0-9])",
        re.IGNORECASE,
    )
    AU_PERMIT_FALLBACK = re.compile(
        r"\b(?:P\s*C|P\s*A|D\s*P|C\s*U)\b(?:[\s\-/]*[A-Z0-9]{2,6}){2,5}",
        re.I,
    )

    # OCR letter↔digit confusion map at suffix position (char 10, 0-indexed)
    # These letters are systematically misread as digits by Tesseract
    _OCR_DIGIT_TO_LETTER = {"2": "Z", "8": "B", "0": "O", "1": "I", "6": "G", "5": "S"}

    @classmethod
    def _clean_permit_ref(cls, raw: str) -> str:
        """
        Normalize a raw permit reference to the standard 15-char format.

        Three normalizations:
        1. Remove non-alphanumeric, uppercase
        2. Zero-pad 2-digit dept: PC6448324B0041 (14) → PC06448324B0041 (15)
        3. Truncate OCR noise suffix: PC06448324B0041ST → PC06448324B0041
        4. Fix OCR digit→letter at suffix position (char index 10):
           PC03320024 20041 → PC03320024Z0041 (2 was Z misread by OCR)

        WHY position 10?
        French permit refs: PP(2) + DDD(3) + DDD(3) + DD(2) + X(1) + DDDD(4) = 15
        Position 10 (0-indexed) is always a LETTER (the type indicator).
        OCR frequently misreads this letter as a visually similar digit.
        """
        clean = re.sub(r"[^A-Z0-9]", "", raw.upper())
        prefix = clean[:2] if len(clean) >= 2 else ""
        if prefix in ("PC", "PA", "DP", "CU"):
            if len(clean) == 14:           # 2-digit dept — zero-pad
                clean = clean[:2] + "0" + clean[2:]
            if len(clean) > 15:            # OCR noise — truncate
                clean = clean[:15]
            # Fix OCR digit at suffix letter position (index 10)
            if len(clean) == 15 and clean[10].isdigit():
                letter = cls._OCR_DIGIT_TO_LETTER.get(clean[10])
                if letter:
                    clean = clean[:10] + letter + clean[11:]
        return clean

    # ── Document type keywords ────────────────────────────────────────────────
    # "urbanisme" removed — too generic, matches plan filenames.
    # Plans use compound check (both words required) — see _detect_type.
    FICHE_KW        = {"fiche", "renseignement"}
    MANDAT_KW       = {"mandat"}
    AUTORISATION_KW = {"autorisation", "permis", "arrêté", "arrete"}
    CERTIFICAT_KW   = {"certificat", "adressage"}

    # ── Table column keyword groups ───────────────────────────────────────────
    TABLE_COL_KEYWORDS = {
        "adresse":  ["adresse", "voie", "rue", "address"],
        "nb_resid": ["résidentiel", "residentiel", "logts", "logement"],
        "nb_pro":   ["professionnel", "locaux", "pro", "cellule", "cellules"],
        "el_resid": ["el residentiel", "el logement", "el logt"],
        "el_pro":   ["el pro", "el professionnel", "el locaux", "el cellules"],
        "nb_lots":  ["lot", "lots"],
        "nb_macrolots": ["macrolot", "macrolots"],
    }

    # ── Fiche text patterns: (key, [patterns]) ───────────────────────────────
    # Tried in order — first match wins. Merged from the two former methods.
    FICHE_COUNT_PATTERNS = {
        "nb_logements_residentiels": [
            r"nb\s*(?:de\s*)?(?:logts?|logements?)\s*(?:r[ée]sidentiels?)?\s*[:=]?\s*(\d+)",
            r"nb\s*total\s*de\s*logements(?!\s*/\s*locaux?\s*/\s*lots?)(?:\s|[/\\]|$|[^\w])+[:=]?\s*(\d+)",
            r"logements?(?!\s*/\s*locaux?\s*/\s*lots?)(?:\s*(?:r[ée]sidentiels?)?)?\s*[:=]?\s*(\d+)",
        ],
        "nb_locaux_pros": [
            r"nb\s*(?:de\s*)?(?:locaux?|cellules?)\s*(?:professionnels?|pros?)\s*[:=]?\s*(\d+)",
            r"nb\s*de\s*cellules?\s*[:=]?\s*(\d+)",
            r"nb\s*total\s*de\s*(?:locaux|cellules)(?:\s|[/\\]|$|[^\w])+[:=]?\s*(\d+)",
            r"locaux?\s*(?:professionnels?|pros?)?\s*[:=]?\s*(\d+)",
        ],
        "nb_lots": [
            r"nb\s*(?:de\s*)?(?:lots?|parcelles?)\s*[:=]?\s*(\d+)",
            r"nombre\s+de\s+lots\s*[:=]?\s*(\d+)",
        ],
        "nb_macrolots": [
            r"nb\s*(?:de\s*)?(?:macrolots?|macroparcelles?)\s*[:=]?\s*(\d+)",
            r"nombre\s+de\s+macrolots\s*[:=]?\s*(\d+)",
        ],
    }
    FICHE_TOTAL_PATTERN = re.compile(
        r"nb\s*total\s*de\s*logements\s*/\s*locaux\s*/\s*lots\s*[:=]?\s*(\d+)", re.I
    )
    FICHE_LOGT_PRO_PAIR = re.compile(
        r"\b(\d{1,4})\s*logements?\s*(?:et|\+|&|,)\s*(\d{1,4})\s*(?:locaux|local|cellules?)(?:\s+(?:pro\w*|professionnel\w*))?",
        re.I,
    )
    FICHE_PRO_LOGT_PAIR = re.compile(
        r"\b(\d{1,4})\s*(?:locaux|local|cellules?)(?:\s+(?:pro\w*|professionnel\w*))?\s*(?:et|\+|&|,)\s+(\d{1,4})\s*logements?",
        re.I,
    )
    # OCR variant: "62 LOGEMENTS ET 5 LOCAUX PRO" style (all caps, extra space)
    FICHE_PAIR_OCR_STRICT = re.compile(
        r"(?:^|\s|[^a-z])(\d{1,4})\s*LOGEMENTS?[\s]*(?:ET|E\s*T)[\s]*(\d{1,4})\s*LOCAUX?[\s]*PRO",
        re.IGNORECASE,
    )
    # Direct digit extraction with flexible separator: "62 logements, 5 locaux profs"
    FICHE_PAIR_FLEXIBLE = re.compile(
        r"(\d{1,4})\s*logements?[\s,;:\-]*(?:et|and|&|\+|,)\s*(\d{1,4})\s*(?:locaux?|local|cellule|cell)",
        re.I,
    )
    FICHE_NB_CELLULES = re.compile(
        r"\bnb\s*(?:de\s*)?cellules?\s*[:=]?\s*(\d+)\b",
        re.I,
    )

    def __init__(self) -> None:
        # Language is fixed by project requirement: French only.
        self._ocr_lang = "fra"

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def extract(self, pdf_path: str, file_name: str = "") -> dict[str, Any]:
        start  = time.perf_counter()
        # Use a minimal pre-detection base — type-specific fields added below
        result: dict[str, Any] = self._empty_result()
        try:
            text, is_scanned = self._get_text(pdf_path)

            if not text.strip():
                return result

            doc_type = self._detect_type(file_name.lower(), text)
            result["document_type"] = doc_type

            # Rebuild result with only the fields relevant to this document type
            typed = self._empty_result(doc_type)
            typed.update(result)   # carry over scanned/error flags
            result = typed

            if   doc_type == "fiche":        result.update(self._extract_fiche(pdf_path, text, is_scanned))
            elif doc_type == "autorisation": result.update(self._extract_autorisation(text))
            elif doc_type == "mandat":       result.update(self._extract_mandat(text))
            elif doc_type == "unknown":
                logger.info("Document type could not be classified: %s", file_name)

            self._post_validate_result(result)
            result["present"] = any(v is not None for k, v in result.items() if k not in {"document_type", "present"})

            logger.info("Done %.2fs | %s | type=%s | scanned=%s",
                        time.perf_counter() - start, file_name,
                        doc_type, is_scanned)
        except Exception as e:
            logger.exception("Extraction failed: %s", pdf_path)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # TEXT LAYER
    # ══════════════════════════════════════════════════════════════════════════

    def _get_text(self, pdf_path: str) -> tuple[str, bool]:
        """Digital PDF → pdfplumber (instant). Scanned → Tesseract OCR."""
        if Path(pdf_path).suffix.lower() in {".png", ".jpg", ".jpeg"}:
            with Image.open(pdf_path) as image:
                return self._ocr_to_string(self._prepare_ocr_image(image)), True

        with pdfplumber.open(pdf_path) as pdf:
            pages = list(pdf.pages)
            page_texts = [p.extract_text() or "" for p in pages]
            digital = "\n--- PAGE ---\n".join(page_texts)
            pages_with_text = sum(1 for t in page_texts if len(t.strip()) >= 20)
            if pages_with_text >= 1:
                return digital, False

            logger.info("Scanned PDF detected — running OCR")
            ocr_config = self._build_ocr_config(psm_default=6)
            pages = [
                self._ocr_to_string(
                    self._prepare_ocr_image(p.to_image(resolution=300).original),
                    config=ocr_config,
                )
                for p in pages
            ]
        return "\n--- PAGE ---\n".join(pages), True

    # ══════════════════════════════════════════════════════════════════════════
    # DOCUMENT TYPE DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_type(self, file_lower: str, text: str) -> str:
        """
        Detection strategy — FILENAME FIRST, content only for ambiguous names.

        WHY filename first?
        Orange follows a strict naming convention:
          PFxxxxxxxx_Fiche-de-renseignement_N.pdf
          PFxxxxxxxx_Autorisation-d-urbanisme_PAR-N-N_N.pdf
          PFxxxxxxxx_Mandat_N.pdf
          PFxxxxxxxx_Plan-de-situation_N.pdf / Plan-de-masse_N.pdf

        The filename is the most reliable signal — it's chosen explicitly.
        Content detection runs ONLY when the filename gives no clear signal
        (e.g. "Autre", "document_1", "PAR-3-1").

        WHY the previous content-first approach was wrong:
        Every fiche contains "Autorisation d'Urbanisme : OUI/NON" and a permit
        ref code (PC...) in its text. Content-first detection matched these as
        autorisation markers and returned the wrong type for fiches.
        """
        # ── Priority 1: filename compound plan check ──────────────────────────
        # Plans require BOTH words to avoid "fiche_plan_projet.pdf" → plan
        if "plan" in file_lower and "situation" in file_lower:
            return "plan_situation"
        if "plan" in file_lower and "masse" in file_lower:
            return "plan_masse"

        # ── Priority 2: unambiguous filename keywords ─────────────────────────
        # Each keyword is EXCLUSIVE to one document type in Orange's naming.
        if "fiche" in file_lower or "renseignement" in file_lower:
            return "fiche"
        if "mandat" in file_lower:
            return "mandat"
        if "certificat" in file_lower or "adressage" in file_lower:
            return "certificat"
        if "autorisation" in file_lower or "permis" in file_lower:
            return "autorisation"

        # ── Priority 3: content detection — AMBIGUOUS filenames only ─────────
        # Reached only for names like "Autre", "document_1", "PAR-3-1".
        # Uses EXCLUSIVE structural markers — present in ONE type only.
        # Checked in order: most specific → least specific.
        t = re.sub(r"\s+", " ", text[:3000]).lower()

        # Plans — check structural phrases unique to plan documents
        if "plan de situation" in t:
            return "plan_situation"
        if "plan de masse" in t or "plan masse" in t:
            return "plan_masse"

        # Mandat — "mandant" + "mandataire" co-occurring is unique to mandats
        # A fiche only says "mandat de représentation" — not mandant/mandataire pair
        if re.search(r"mandant.{0,300}mandataire", t, re.S):
            return "mandat"
        if re.search(r"mandat\s+de\s+signature", t):
            return "mandat"

        # Fiche — Orange-specific headings that NEVER appear in other docs
        # "FICHE DE RENSEIGNEMENTS" is the clearest possible signal
        if re.search(r"fiche\s+de\s+renseignements?", t):
            return "fiche"
        if re.search(r"date\s+de\s+livraison\s+du\s+projet", t):
            return "fiche"
        if re.search(r"ma[iî]tre\s+d.ouvrage\s*/\s*propri[eé]taire", t):
            return "fiche"
        if re.search(r"description\s+de\s+l.op[eé]ration", t):
            return "fiche"

        # Autorisation — permit-specific vocabulary and structure
        # NOTE: "permis" and "autorisation" appear in fiche text too, so we
        # require more specific phrases like "PERMIS DE CONSTRUIRE" heading
        if re.search(r"permis\s+de\s+construire|d[eé]claration\s+pr[eé]alable", t):
            return "autorisation"
        if re.search(r"arrêt[eé]\s*\s*:", t):
            return "autorisation"
        if re.search(r"article\s+1\s*\s*:", t):
            return "autorisation"

        # Default — fiche is the most common document in this workflow
        return "fiche"


    def _extract_fiche(self, pdf_path: str, text: str, is_scanned: bool = False) -> dict:
        norm = re.sub(r"\s+", " ", text)
        m    = self.FICHE_PERMIT.search(norm)
        ref  = m.group(1).replace(" ", "").upper() if m else None
        # Fallback: bare ref without PC/PA/DP prefix typed by consultant
        if not ref:
            m2 = self.FICHE_PERMIT_BARE.search(norm)
            if m2:
                candidate = re.sub(r"[^A-Z0-9]", "", m2.group(1).upper())
                # Accept if it looks like a valid ref body (10-13 alphanum chars)
                if 10 <= len(candidate) <= 13:
                    ref = candidate
        out  = {
            "ref_urbanisme": ref,
            "dlpi":          None,
        }
        m = self.FICHE_DLPI.search(norm)
        if m:
            out["dlpi"] = re.sub(r"\s*/\s*", "/", m.group(1))

        compact = re.sub(r"\s+", " ", text)

        # ── Layer 1: pdfplumber structured table (AUTHORITATIVE) ─────────────
        # The detail table (Table 2) is always checked first because it has
        # explicit column headers — "Nb de logts résidentiel" and
        # "Nb de locaux professionnel" — that leave no ambiguity.
        # Pair extraction (summary text) runs AFTER and only fills gaps.
        if Path(pdf_path).suffix.lower() == ".pdf":
            table_result = self._extract_fiche_table(pdf_path)
            out.update({k: v for k, v in table_result.items() if v is not None})

        # ── Layer 1.5: pair extraction — fills gaps left by table ─────────────
        # "62 logements et 5 locaux profs" — explicit combined summary phrase.
        # Only fires for fields the table did NOT populate.
        pair_found = False
        if out.get("nb_logements_residentiels") is None or out.get("nb_locaux_pros") is None:
            pair = self._extract_logements_locaux_pair(compact)
            if pair:
                pair_found = True
                if out.get("nb_logements_residentiels") is None:
                    out["nb_logements_residentiels"] = pair[0]
                if out.get("nb_locaux_pros") is None:
                    out["nb_locaux_pros"] = pair[1]
                logger.info("Pair filled gaps: res=%s pro=%s",
                            out.get("nb_logements_residentiels"), out.get("nb_locaux_pros"))

        # ── Layer 2: regex pattern matching on digital text ────────────────────
        if not pair_found and out.get("nb_logements_residentiels") is None and out.get("nb_locaux_pros") is None:
            self._fill_fiche_counts(out, compact)

        # Explicit "Nb de cellules" value is authoritative for professional units.
        explicit_pro = self._extract_explicit_cellules_count(compact)
        if explicit_pro is not None:
            out["nb_locaux_pros"] = explicit_pro
            # If residential was only inferred from an ambiguous total and equals
            # the explicit pro count, clear it so later layers can recover it.
            if out.get("nb_logements_residentiels") == explicit_pro:
                has_explicit_res = re.search(
                    r"\bnb\s*(?:de\s*)?(?:logts?|logements?)\s*(?:r[ée]sidentiels?)?\s*[:=]\s*\d+\b",
                    compact,
                    re.I,
                )
                if not has_explicit_res:
                    out["nb_logements_residentiels"] = None

        # ── Layer 3: OCR image fallback — handles screenshot tables, bad scan,
        #    cropped or non-standard templates where pdfplumber finds nothing ──
        counts_missing = (
            out.get("nb_logements_residentiels") is None or
            out.get("nb_locaux_pros") is None
        )
        # Trigger OCR refinement if:
        # 1. We found any counts but they include suspicious values (0/1 pro with many logements)
        # 2. Or counts are completely missing
        # Suspicious only when res >> pro AND we had no table evidence for pro.
        # pro=0 from the detail table is VALID (the building has 0 pro units).
        # pro=0 from regex/pair on a large building (res≥10) is suspicious.
        suspicious_pro_count = (
            pair_found is False
            and out.get("nb_locaux_pros") == 0
            and (out.get("nb_logements_residentiels") or 0) >= 10
            and not out.get("_pro_from_table", False)  # set below when table found pro col
        )
        if counts_missing or suspicious_pro_count:
            # Scanned files already use OCR in _get_text, so only re-OCR digital files.
            source_text = compact
            if not is_scanned:
                logger.info("Triggering OCR fallback for refinement")
                ocr_text = self._ocr_fiche_images(pdf_path)
                if ocr_text:
                    source_text = f"{source_text} {ocr_text}"

            # Aggressive pair extraction with all variants
            pair = self._extract_logements_locaux_pair_aggressive(source_text)
            if pair:
                out["nb_logements_residentiels"], out["nb_locaux_pros"] = pair
                logger.info("OCR fallback pair extraction successful: %d logements, %d locaux", pair[0], pair[1])

            if not pair:  # If aggressive extraction still fails, try pattern matching
                self._fill_fiche_counts(out, source_text)

            has_explicit_numeric_counts = (
                pair is not None
                or bool(
                    re.search(
                        r"\b(?:nb\s*(?:de\s*)?(?:logts?|logements?|locaux?|cellules?)\s*[:=]?\s*\d+|\d{1,4}\s*logements?.{0,20}\d{1,4}\s*(?:locaux|cellules?))\b",
                        source_text,
                        re.I,
                    )
                )
            )
            el_results = self._extract_el_columns(source_text)
            # EL fields are booleans, not hard counts. Use only as weak fallback
            # when no explicit numeric counts were found in the text.
            if not has_explicit_numeric_counts:
                for key in ["nb_logements_residentiels", "nb_locaux_pros"]:
                    if out.get(key) is None and el_results.get(key) is not None:
                        out[key] = el_results[key]

            still_missing = (
                out.get("nb_logements_residentiels") is None or
                out.get("nb_locaux_pros") is None
            )
            if still_missing and source_text.strip():
                llm_fields = [
                    f for f in ["nb_logements_residentiels", "nb_locaux_pros"]
                    if out.get(f) is None
                ]
                llm_out = self._llm_extract(
                    text=source_text[:1500],
                    fields=llm_fields,
                    instructions=(
                        """### ROLE
You are a French Urbanism Data Auditor specializing in Orange "Fiche de Renseignement" forms. You have high attention to detail and never confuse land lots with housing units.

### ANALYSIS RULES
1. **Identify the Building Table**: Look for rows containing "RUE", "Bâtiment", or "Commune". 
2. **Extract with Logic**:
   - **Residential Units (nb_logements_residentiels)**: Look for "Nb", "EL résidentiel", or "Logements". If the text says "62 logements et 5 locaux", the answer is 62.
   - **Professional Units (nb_locaux_pros)**: Look for "Nb de cellules", "Locaux pros", or "Commerces". If the text says "1 CELLULE", the answer is 1.
3. **DO NOT CONFUSE**:
   - **Land vs. Housing**: "Nb total de lots" in the "Lotissement" section refers to land. Ignore it unless it specifically says "logements".
   - **Booleans vs. Counts**: "EL Résidentiel: OUI" is a checkbox, not a number. Do not extract "1" just because you see "OUI".
   - **The "Total" Trap**: If the document says "Total: 67" but specifies "62 logts + 5 locaux", you MUST return 62 and 5 separately.

### WORKSPACE (Chain of Thought)
Before providing the JSON, identify each building/row you found in the text and the numbers associated with them.

### OUTPUT FORMAT
Return ONLY a JSON object. No prose. No markdown blocks.
{
  "evidence": "List the buildings and counts found here to justify your math",
  "nb_logements_residentiels": <integer or null>,
  "nb_locaux_pros": <integer or null>
}
                    """),
                )
                for key in llm_fields:
                    if llm_out.get(key) is not None:
                        out[key] = llm_out[key]

        # ── Layer 4: infer pro units from declared total ───────────────────────
        if out.get("nb_locaux_pros") is None and out.get("nb_logements_residentiels") is not None:
            hit = self.FICHE_TOTAL_PATTERN.search(compact)
            if hit:
                total = int(hit.group(1))
                if total >= out["nb_logements_residentiels"]:
                    out["nb_locaux_pros"] = total - out["nb_logements_residentiels"]

        return out

    def _ocr_fiche_images(self, pdf_path: str) -> str:
        """
        Rasterise every page of the fiche at high resolution and OCR it.
        Used when the table is an embedded screenshot or poor-quality scan
        that pdfplumber cannot parse as structured text.

        Resolution 300 dpi — enough to read digits clearly from a screenshot.
        Returns the concatenated OCR text of all pages, empty string on failure.
        """
        try:
            pages_text = []
            ocr_config = self._build_ocr_config(psm_default=6)
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    img = self._prepare_ocr_image(page.to_image(resolution=300).original)
                    pages_text.append(
                        self._ocr_to_string(img, config=ocr_config)
                    )
            result = "\n".join(pages_text).strip()
            if result:
                logger.info("OCR image fallback produced %d chars", len(result))
            return result
        except Exception as e:
            logger.warning("OCR image fallback failed: %s", e)
            return ""

    def _extract_fiche_table(self, pdf_path: str) -> dict:
        result = {"adresse": None, "nb_logements_residentiels": None, "nb_locaux_pros": None}

        with pdfplumber.open(pdf_path) as pdf:
            all_tables = [t for page in pdf.pages for t in (page.extract_tables() or [])]

        # Pass 1 — precise header matching, summing ALL data rows.
        # Multi-building fiches have one row per building — counts must be
        # accumulated across every row, not just the first one.
        for table in all_tables:
            if not table or len(table) < 2:
                continue
            # Collapse internal newlines — pdfplumber sometimes splits
            # multi-line header cells with "\n" inside the string.
            header = [re.sub(r"\s+", " ", str(c or "")).lower().strip()
                      for c in table[0]]
            col = {}
            for i, cell in enumerate(header):
                is_ambiguous_total = (
                    "nb total" in cell
                    and "logement" in cell
                    and "locaux" in cell
                )
                # Residential: match "résidentiel/residentiel" that isn't also pro
                if (
                    not is_ambiguous_total
                    and ("résidentiel" in cell or "residentiel" in cell)
                    and not ("professionnel" in cell or "pro" in cell)
                    and not ("locaux" in cell or "lots" in cell)
                    and not re.match(r"^el\b", cell)  # Exclude EL (boolean) column
                ):
                    col["nb_resid"] = i
                # Professional: numeric count columns only (avoid EL yes/no columns)
                # Prefer explicit professional headers or any "cellule" count header.
                elif (
                    not is_ambiguous_total
                    and (
                        "cellule" in cell
                        or (("locaux" in cell) and ("pro" in cell or "professionnel" in cell))
                    )
                    and not re.match(r"^el\b", cell)  # Exclude EL (boolean) column
                ):
                    col["nb_pro"] = i
                elif any(k in cell for k in ["adresse", "voie", "rue", "address"]):
                    col["adresse"] = i

            # Bare "Nb" column mapping:
            # Case A: "Nb de cellules" is present → "Nb" = residential (existing logic)
            # Case B: No pro column at all but table has structural markers
            #         (commune/synergie/voie/nom/dlpi) → "Nb" = residential.
            #         This covers single-unit fiches (PDF3 template) where
            #         pro column is simply absent.
            _STRUCTURAL_MARKERS = {"commune", "synergie", "voie", "nom",
                                   "dlpi", "numéro voie", "numero voie",
                                   "complément n° voie", "complement n voie"}
            _has_structural = len(set(header) & _STRUCTURAL_MARKERS) >= 2
            if "nb_resid" not in col and ("nb_pro" in col or _has_structural):
                for i, cell in enumerate(header):
                    if re.fullmatch(r"nb", cell):
                        col["nb_resid"] = i
                        break

            if "nb_resid" not in col and "nb_pro" not in col:
                continue

            total_resid = 0
            total_pro   = 0
            found_resid = False
            found_pro   = False
            addr_set    = False

            for row in table[1:]:
                if not row or all(c is None or str(c).strip() == "" for c in row):
                    continue
                # Grab address from first valid row only
                if not addr_set and "adresse" in col and col["adresse"] < len(row):
                    val = str(row[col["adresse"]] or "").strip()
                    if val:
                        result["adresse"] = val
                        addr_set = True
                # Accumulate residential count
                if "nb_resid" in col and col["nb_resid"] < len(row):
                    parsed = self._to_count(str(row[col["nb_resid"]] or "").strip().upper())
                    if parsed is not None:
                        total_resid += parsed
                        found_resid  = True
                # Accumulate professional count
                if "nb_pro" in col and col["nb_pro"] < len(row):
                    parsed = self._to_count(str(row[col["nb_pro"]] or "").strip().upper())
                    if parsed is not None:
                        total_pro += parsed
                        found_pro  = True

            if found_resid:
                result["nb_logements_residentiels"] = total_resid
            if found_pro:
                result["nb_locaux_pros"] = total_pro
                result["_pro_from_table"] = True   # suppress false suspicious flag
            elif "nb_pro" in col:
                # Pro column existed but all rows were empty/None → treat as 0
                result["nb_locaux_pros"] = 0
                result["_pro_from_table"] = True

            # Trust this table immediately when it contains a professional column,
            # or when it is an address-based detail table with both count columns.
            if result["nb_locaux_pros"] is not None:
                return result
            if (
                result["nb_logements_residentiels"] is not None
                and result["adresse"]
                and "nb_pro" in col
            ):
                return result

        # Pass 2 — generic header fallback, also summing all rows
        key_map = {"adresse": "adresse", "nb_resid": "nb_logements_residentiels", "nb_pro": "nb_locaux_pros"}
        for table in all_tables:
            if not table:
                continue
            hi = self._find_table_header(table)
            if hi is None:
                continue
            col_map = self._build_col_map(table[hi])
            totals: dict[str, int] = {}
            addr_set = False
            for row in table[hi + 1:]:
                if not row or all(c is None or str(c).strip() == "" for c in row):
                    continue
                for internal, rkey in key_map.items():
                    idx = col_map.get(internal)
                    if idx is None or idx >= len(row):
                        continue
                    val = str(row[idx] or "").strip()
                    if not val:
                        continue
                    if internal.startswith("nb_"):
                        n = self._safe_int(val)
                        if n is not None:
                            totals[rkey] = totals.get(rkey, 0) + n
                    elif not addr_set:
                        result[rkey] = val
                        addr_set = True
            for rkey, total in totals.items():
                result[rkey] = total

        return result

    def _extract_el_columns(self, text: str) -> dict:
        """
        Extract EL résidentiel and EL pro columns from table text using a simple approach:
        1. Find the position of "EL résidentiel" and "EL pro" 
        2. Extract the first OUI/NON/0/1 value that follows each
        
        Returns: {"nb_logements_residentiels": int or None, "nb_locaux_pros": int or None}
        """
        result = {"nb_logements_residentiels": None, "nb_locaux_pros": None}
        for marker, key in [
            (r"el\s+r[ée]sidentiels?", "nb_logements_residentiels"),
            (r"el\s+pros?", "nb_locaux_pros"),
        ]:
            match = re.search(marker, text, re.I)
            if not match:
                continue
            window = text[match.end():match.end() + 500]
            val_match = re.search(r"\b(oui|non|yes|no|0|1)\b", window, re.I)
            if val_match:
                result[key] = self._to_count(val_match.group(1))
        
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # AUTORISATION
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_au_ref(self, text: str) -> str | None:
        """
        Extract and clean permit reference from autorisation text.
        Handles spaced formats, 2-digit depts, OCR noise, and Z→2 misreads.
        """
        hit = self.AU_PERMIT.search(text)
        if hit:
            return self._clean_permit_ref(hit.group(1))

        # Fallback: scan lines with permit keywords
        for line in text.splitlines():
            if not re.search(r"dossier|permis|arr[êe]t|n[°o]\s*p[cad]", line, re.I):
                continue
            cand = self.AU_PERMIT_FALLBACK.search(line)
            if cand:
                cleaned = self._clean_permit_ref(cand.group(0))
                if re.match(r"^(PC|PA|DP|CU)[0-9A-Z]{13}$", cleaned):
                    return cleaned
        return None

    def _extract_autorisation(self, text: str) -> dict:
        out = {
            "ref_urbanisme": self._extract_au_ref(text),
        }
        llm_out = self._llm_extract(
            text=text[:1200],
            fields=["adresse"],
            instructions=(
                "Extract the project address — the physical location of the building or land "
                "the permit refers to. Include street number, street name, postal code and city. "
                "Do NOT return the applicant's home address."
                "Extract number of logements if you can find it with certainty, but do NOT guess or infer from context — we just want what is explicitly stated in the text. "
            ),
        )
        out["adresse"] = llm_out.get("adresse")
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # MANDAT
    # ══════════════════════════════════════════════════════════════════════════

    # Anchor present in every Orange mandat page regardless of template
    MANDAT_PAGE_ANCHOR = re.compile(
        r"mandat\s+de\s+signature|mandataire|le\s+mandant.*le\s+mandataire", re.I | re.S
    )

    def _extract_mandat(self, text: str) -> dict:
        fields = ["orange_representant_nom",
                  "orange_representant_mobile", "orange_representant_email"]

        # The mandat section can be on ANY page — often the last one when the
        # PDF bundles devis + CGV + mandat together. Scan pages in reverse
        # (mandat is almost always last) and pass only the matching page to the
        # LLM so relevant content isn't buried under boilerplate.
        pages = text.split("\n--- PAGE ---\n")
        mandat_chunk = next(
            (p for p in reversed(pages) if self.MANDAT_PAGE_ANCHOR.search(p)),
            text[-3000:],   # fallback: last 3000 chars of unsplit text
        )

        llm_out = self._llm_extract(
            text=mandat_chunk[:1600],
            fields=fields,
            instructions=(
                "This is a French FTTH mandate (Mandat de signature). "
                "Orange is always the SECOND party — the \'Mandataire\'. "
                "Extract ONLY the Orange representative\'s details — NOT the client (Mandant). "
                "orange_representant_nom: full name. "
                "orange_representant_mobile: phone number. "
                "orange_representant_email: @orange.com email."
            ),
        )
        return llm_out

    # ══════════════════════════════════════════════════════════════════════════
    # LLM
    # ══════════════════════════════════════════════════════════════════════════

    def _llm_extract(self, text: str, fields: list[str], instructions: str) -> dict:
        """Focused extraction for fields regex can't handle. Never raises."""
        import ollama
        prompt = (
            f"You are a data extraction expert for French real estate documents.\n"
            f"{instructions}\n\n"
            f"Return ONLY a valid JSON object with these exact keys: {json.dumps(fields)}\n"
            f"For missing values: return 0 for numeric count fields and null for other fields.\n"
            f"No markdown, no explanation, no text outside the JSON.\n\n"
            f"Document text:\n{text}"
        )
        model_name = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
        options = {
            "temperature": 0,
            "top_k": 10,
            "top_p": 0.9,
            # CPU: smaller context = faster prefill
            "num_ctx": 1024,
            # Use physical cores only (avoid hyper-threading contention)
            "num_thread": int(os.getenv("OLLAMA_THREADS",
                              str(max(1, (os.cpu_count() or 4) // 2)))),
        }

        try:
            raw = ollama.generate(model=model_name, prompt=prompt, options=options).get("response", "")
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end <= 0:
                logger.warning("LLM returned no JSON block")
                return {}
            payload = json.loads(raw[start:end])
            return {f: self._coerce_field_value(f, payload.get(f)) for f in fields}
        except json.JSONDecodeError as e:
            logger.warning("LLM JSON parse error: %s", e)
            return {}
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            return {}

    # ══════════════════════════════════════════════════════════════════════════
    # HELPERS
    # ══════════════════════════════════════════════════════════════════════════

    # ── Per-type result schemas — only relevant fields, no cross-type nulls ────
    _BASE = {"document_type": None, "present": False}

    _INT_FIELDS = {
        "nb_logements_residentiels",
        "nb_locaux_pros",
    }
    _STR_FIELDS = {
        "ref_urbanisme",
        "dlpi",
        "adresse",
        "orange_representant_nom",
        "orange_representant_mobile",
        "orange_representant_email",
    }

    _SCHEMA: dict[str, dict] = {
        "fiche": {
            "ref_urbanisme":            None,
            "dlpi":                     None,
            "adresse":                  None,   # adresse_tableau mapped here
            "nb_logements_residentiels": None,
            "nb_locaux_pros":           None,
        },
        "autorisation": {
            "ref_urbanisme": None,
            "adresse":       None,
        },
        "mandat": {
            "orange_representant_nom":    None,
            "orange_representant_mobile": None,
            "orange_representant_email":  None,
        },
        # Plans and certificat carry no extractable fields — presence only
        "plan_situation": {},
        "plan_masse":     {},
        "certificat":     {},
        "unknown":        {},
    }

    def _empty_result(self, doc_type: str | None = None) -> dict[str, Any]:
        """
        Return a base result dict with only the fields relevant to doc_type.
        If doc_type is not yet known (pre-detection), return just the base.
        """
        base = dict(self._BASE)
        if doc_type and doc_type in self._SCHEMA:
            base.update(self._SCHEMA[doc_type])
        return base

    def _coerce_field_value(self, field: str, value: Any) -> Any:
        """Normalize LLM values to expected per-field types."""
        if field in self._INT_FIELDS:
            if value is None:
                return 0
            parsed = self._safe_int(value)
            return parsed if parsed is not None else 0
        if value is None:
            return None
        if field in self._STR_FIELDS:
            text = str(value).strip()
            if not text or text in {"0", "null", "none", "n/a", "na"}:
                return None
            return text
        return value

    def _post_validate_result(self, result: dict[str, Any]) -> None:
        """Apply lightweight field normalization and validation in-place."""
        result.pop("_pro_from_table", None)  # remove internal flag

        # Basic type/value validation
        for field in self._INT_FIELDS:
            if field not in result:
                continue
            parsed = self._safe_int(result.get(field))
            if parsed is None:
                result[field] = 0
                continue
            if parsed < 0:
                result[field] = 0
                continue
            result[field] = parsed

        ref = result.get("ref_urbanisme")
        if ref is not None:
            clean_ref = self._clean_permit_ref(str(ref))
            if re.match(r"^(PC|PA|DP|CU)[0-9A-Z]{13}$", clean_ref):
                result["ref_urbanisme"] = clean_ref
            else:
                result["ref_urbanisme"] = None

        dlpi = result.get("dlpi")
        if dlpi is not None:
            dlpi_norm = re.sub(r"\s*/\s*", "/", str(dlpi).strip())
            if re.match(r"^\d{1,2}/\d{1,2}/\d{4}$", dlpi_norm):
                result["dlpi"] = dlpi_norm
            else:
                result["dlpi"] = None

        email = result.get("orange_representant_email")
        if email is not None:
            email_text = str(email).strip()
            if re.match(r"^[^@\s]+@orange\.com$", email_text, re.I):
                result["orange_representant_email"] = email_text
            else:
                result["orange_representant_email"] = None

        mobile = result.get("orange_representant_mobile")
        if mobile is not None:
            mobile_text = str(mobile).strip()
            digits = re.sub(r"\D", "", mobile_text)
            if 10 <= len(digits) <= 15:
                result["orange_representant_mobile"] = mobile_text
            else:
                result["orange_representant_mobile"] = None

        for field in self._STR_FIELDS:
            if field not in result:
                continue
            if result.get(field) is None:
                continue
            text = str(result[field]).strip()
            if not text:
                result[field] = None
            else:
                result[field] = text

    def _safe_int(self, value: Any) -> int | None:
        s = re.sub(r"[^0-9\-]", "", str(value or "").strip())
        try:
            return int(s) if s else None
        except ValueError:
            return None

    def _build_ocr_config(self, psm_default: int = 6) -> str:
        """Build OCR config with environment overrides for fast runtime tuning."""
        try:
            psm = int(os.getenv("OCR_PSM", str(psm_default)))
        except ValueError:
            psm = psm_default
        try:
            oem = int(os.getenv("OCR_OEM", "1"))
        except ValueError:
            oem = 1
        # Clamp to common valid ranges to avoid passing invalid Tesseract args.
        psm = psm if 0 <= psm <= 13 else psm_default
        oem = oem if 0 <= oem <= 3 else 1
        return f"--oem {oem} --psm {psm}"

    def _ocr_to_string(self, image: Image.Image, config: str | None = None) -> str:
        """Run OCR in French only (fra)."""
        if config is None:
            config = self._build_ocr_config(psm_default=6)
        kwargs: dict[str, Any] = {"lang": self._ocr_lang}
        if config:
            kwargs["config"] = config
        try:
            return pytesseract.image_to_string(image, **kwargs)
        except pytesseract.TesseractError as e:
            logger.warning("OCR failed for lang=%s (%s)", self._ocr_lang, e)
        except Exception as e:
            logger.warning("OCR unexpected failure for lang=%s (%s)", self._ocr_lang, e)
        return ""

    def _prepare_ocr_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocessing pipeline tuned for CPU-based OCR on French real-estate forms.

        Order matters:
          1. EXIF rotation first — work in correct orientation from the start
          2. Greyscale conversion — all subsequent ops are single-channel
          3. Upscale if small — Tesseract needs ≥150 px/char-height to be reliable
          4. Auto-rotate via OSD — fix 90/180/270 degree scans
          5. Contrast normalisation — pull weak ink/shadow into a usable range
          6. Sauvola local binarization — handles uneven lighting & table borders
             without blurring digits (avoids the MedianFilter+Sharpen trap that
             smears thin strokes before thresholding)
        """
        # Step 1 — honour EXIF rotation embedded by scanners/phones
        img = ImageOps.exif_transpose(image)

        # Step 2 — greyscale
        if img.mode != "L":
            img = img.convert("L")

        # Step 3 — upscale small images (min dimension < 1800 px)
        if min(img.size) < 1800:
            scale = max(2, 1800 // min(img.size))
            img = img.resize(
                (img.width * scale, img.height * scale),
                Image.Resampling.LANCZOS,
            )

        # Step 4 — OSD auto-rotation (before heavy processing)
        img = self._autorotate_for_ocr(img)

        # Step 5 — contrast normalisation (stretch histogram, no clipping)
        img = ImageOps.autocontrast(img, cutoff=1)

        # Step 6 — Sauvola local binarization via numpy
        #  Sauvola threshold: T = mean * (1 + k * (std/R - 1))
        #  k=0.25, R=128  → aggressive enough for low-contrast forms
        return self._sauvola_binarize(img)

    def _autorotate_for_ocr(self, image: Image.Image) -> Image.Image:
        """Use Tesseract OSD to rotate pages into a readable orientation."""
        try:
            osd = pytesseract.image_to_osd(image, output_type=Output.DICT)
            rotate = int(osd.get("rotate", 0) or 0)
            confidence = float(osd.get("orientation_conf", 0.0) or 0.0)
            if rotate in (90, 180, 270) and confidence >= 3.0:
                return image.rotate(-rotate, expand=True, fillcolor=255)
        except Exception as e:
            logger.debug("OSD auto-rotation skipped: %s", e)
        return image

    def _sauvola_binarize(self, gray: Image.Image, k: float = 0.25,
                          window: int = 51, R: float = 128.0) -> Image.Image:
        """
        Sauvola local binarization — handles uneven illumination, shadow gradients,
        and table-border interference without blurring digit strokes.

        window: local neighbourhood size in pixels (odd number).
                Larger → better for big text; 51px works well at 300 dpi.
        k:      sensitivity (0.2–0.5); higher = more aggressive foreground capture.
        R:      dynamic range of standard deviation (128 for 8-bit greyscale).

        Falls back to Otsu-style global threshold on numpy failure.
        """
        try:
            if window < 3:
                window = 3
            if window % 2 == 0:
                window += 1

            arr = np.array(gray, dtype=np.float32)

            # Reflect-pad then use integral images with an extra zero border,
            # yielding true per-pixel local window sums in O(1).
            pad = window // 2
            padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
            padded_sq = padded ** 2

            ii = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
            ii2 = np.pad(padded_sq, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)

            s = ii[window:, window:] - ii[:-window, window:] - ii[window:, :-window] + ii[:-window, :-window]
            s2 = ii2[window:, window:] - ii2[:-window, window:] - ii2[window:, :-window] + ii2[:-window, :-window]

            win_area = float(window * window)
            mean = s / win_area
            var  = np.maximum(s2 / win_area - mean ** 2, 0)
            std  = np.sqrt(var)

            threshold = mean * (1.0 + k * (std / R - 1.0))
            binary = (arr >= threshold).astype(np.uint8) * 255
            return Image.fromarray(binary, mode="L")
        except Exception as e:
            logger.debug("Sauvola binarize fallback used: %s", e)
            # Simple Otsu-style global fallback
            arr = np.array(gray, dtype=np.uint8)
            thr = int(arr.mean())
            return Image.fromarray(np.where(arr >= thr, 255, 0).astype(np.uint8), mode="L")

    def _to_count(self, value: Any) -> int | None:
        v = str(value or "").strip().upper()
        if v in ("OUI", "YES", "1"):
            return 1
        if v in ("NON", "NO", "0"):
            return 0
        return self._safe_int(v)

    def _fill_fiche_counts(self, out: dict, text: str) -> None:
        for key, patterns in self.FICHE_COUNT_PATTERNS.items():
            if out.get(key) is not None:
                continue
            for pat in patterns:
                hit = re.search(pat, text, re.I)
                if not hit:
                    continue
                parsed = self._to_count(hit.group(1))
                if parsed is not None:
                    out[key] = parsed
                    break

    def _extract_explicit_cellules_count(self, text: str) -> int | None:
        """Return explicit professional-unit count from 'Nb de cellules' labels."""
        m = self.FICHE_NB_CELLULES.search(text)
        if not m:
            return None
        return self._safe_int(m.group(1))

    def _extract_logements_locaux_pair(self, text: str) -> tuple[int, int] | None:
        """Extract logements and locaux pros using primary patterns."""
        m = self.FICHE_LOGT_PRO_PAIR.search(text)
        if m:
            logements, locaux = int(m.group(1)), int(m.group(2))
            if logements > 0 or locaux > 0:  # Accept at least one non-zero value
                return logements, locaux
        m = self.FICHE_PRO_LOGT_PAIR.search(text)
        if m:
            locaux, logements = int(m.group(1)), int(m.group(2))
            if logements > 0 or locaux > 0:
                return logements, locaux
        return None

    def _extract_logements_locaux_pair_aggressive(self, text: str) -> tuple[int, int] | None:
        """Aggressive extraction with OCR-variant patterns for refinement pass."""
        patterns = [
            (self.FICHE_LOGT_PRO_PAIR, lambda m: (int(m.group(1)), int(m.group(2)))),
            (self.FICHE_PRO_LOGT_PAIR, lambda m: (int(m.group(2)), int(m.group(1)))),
            (self.FICHE_PAIR_OCR_STRICT, lambda m: (int(m.group(1)), int(m.group(2)))),
            (self.FICHE_PAIR_FLEXIBLE, lambda m: (int(m.group(1)), int(m.group(2)))),
        ]
        for pattern, extractor in patterns:
            m = pattern.search(text)
            if m:
                try:
                    logements, locaux = extractor(m)
                    if logements > 0 or locaux > 0:
                        return logements, locaux
                except (ValueError, IndexError):
                    continue
        return None

    def _find_table_header(self, table) -> int | None:
        addr_kw  = {"adresse", "bâtiment", "batiment"}
        units_kw = {"résidentiel", "residentiel", "professionnel", "locaux", "cellule", "nb de cellules"}
        for i, row in enumerate(table):
            if not row:
                continue
            rt = " ".join(str(c or "").lower() for c in row)
            if any(kw in rt for kw in addr_kw) and any(kw in rt for kw in units_kw):
                return i
        return None

    def _build_col_map(self, header_row) -> dict:
        header = [str(c or "").lower() for c in header_row]
        return {
            col_key: next((i for i, cell in enumerate(header) if any(kw in cell for kw in kws)), None)
            for col_key, kws in self.TABLE_COL_KEYWORDS.items()
        }