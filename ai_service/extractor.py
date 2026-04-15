import pdfplumber
import pytesseract
import re
import json
import logging
import os
import time
from shutil import which
from pathlib import Path
from typing import Any
from PIL import Image, ImageOps, ImageFilter, ImageEnhance, ImageMath
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
    FICHE_DLPI   = re.compile(
        r"DATE\s+DE\s+LIVRAISON\s+DU\s+PROJET\s*[:\-]?\s*(\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4})",
        re.I,
    )
    AU_PERMIT = re.compile(
        r"\b((?:P\s*C|P\s*A|D\s*P|C\s*U)[\s\-/]*(?:"
        r"\d{5}[\s\-/]*\d{2}[\s\-/]*(?:\d{5}|[A-Z0-9]{4,6})(?:[\s\-/]*[A-Z0-9]{2,4})?|"
        r"\d{3}[\s\-/]*\d{3}[\s\-/]*\d{2}[\s\-/]*[A-Z0-9]{5,6}|"
        r"(?:\d[\s\-/]*){10,14}[A-Z0-9]{0,4}"
        r"))\b",
        re.I,
    )
    AU_PERMIT_FALLBACK = re.compile(
        r"\b(?:P\s*C|P\s*A|D\s*P|C\s*U)\b(?:[\s\-/]*[A-Z0-9]{2,6}){2,5}",
        re.I,
    )

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
        "nb_pro":   ["professionnel", "locaux", "pro"],
        "nb_lots":  ["lot", "lots"],
        "nb_macrolots": ["macrolot", "macrolots"],
    }

    # ── Fiche text patterns: (key, [patterns]) ───────────────────────────────
    # Tried in order — first match wins. Merged from the two former methods.
    FICHE_COUNT_PATTERNS = {
        "nb_logements_residentiels": [
            r"nb\s*(?:de\s*)?(?:logts?|logements?)\s*(?:r[ée]sidentiels?)?\s*[:=]?\s*(\d+)",
            r"nb\s*total\s*de\s*logements(?:\s|[/\\]|$|[^\w])+[:=]?\s*(\d+)",
            r"logements?\s*(?:r[ée]sidentiels?)?\s*[:=]?\s*(\d+)",
            r"el\s+r[ée]sidentiels?[^a-z]*?(oui|non|0|1|yes|no|\d{1,3})(?:\s|,|\n|$)",
        ],
        "nb_locaux_pros": [
            r"nb\s*(?:de\s*)?(?:locaux?|cellules?)\s*(?:professionnels?|pros?)\s*[:=]?\s*(\d+)",
            r"nb\s*total\s*de\s*(?:locaux|cellules)(?:\s|[/\\]|$|[^\w])+[:=]?\s*(\d+)",
            r"nombre\s+de\s+lots\s*[:=]?\s*(\d+)",
            r"locaux?\s*(?:professionnels?|pros?)?\s*[:=]?\s*(\d+)",
            r"el\s+pros?[^a-z]*?(oui|non|0|1|yes|no|\d{1,3})(?:\s|,|\n|$)",
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
        r"(\d{1,4})\s*logements?[\s,;:\-]*(?:et|and|and|&|\+|,)\s*(\d{1,4})\s*(?:locaux?|local|cellule|cell)",
        re.I,
    )

    def __init__(self) -> None:
        self._tesseract_langs = self._detect_tesseract_languages()

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ══════════════════════════════════════════════════════════════════════════

    def extract(self, pdf_path: str, file_name: str = "") -> dict[str, Any]:
        start  = time.perf_counter()
        # Use a minimal pre-detection base — type-specific fields added below
        result: dict[str, Any] = dict(self._BASE)
        try:
            text, is_scanned = self._get_text(pdf_path)
            result["scanned"] = is_scanned

            if not text.strip():
                result["error"] = "No text extracted — blank or corrupt PDF"
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

            if not result.get("error"):
                result["present"] = True

            logger.info("Done %.2fs | %s | type=%s | scanned=%s | llm=%s",
                        time.perf_counter() - start, file_name,
                        doc_type, is_scanned, result.get("llm_used"))
        except Exception as e:
            logger.exception("Extraction failed: %s", pdf_path)
            result["error"] = str(e)
        return result

    # ══════════════════════════════════════════════════════════════════════════
    # TEXT LAYER
    # ══════════════════════════════════════════════════════════════════════════

    def _get_text(self, pdf_path: str) -> tuple[str, bool]:
        """Digital PDF → pdfplumber (instant). Scanned → Tesseract OCR."""
        if Path(pdf_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}:
            with Image.open(pdf_path) as image:
                return self._ocr_to_string(self._prepare_ocr_image(image)), True

        with pdfplumber.open(pdf_path) as pdf:
            pages = list(pdf.pages)
            digital = "\n--- PAGE ---\n".join(p.extract_text() or "" for p in pages)
            if len(digital.strip()) >= 50:
                return digital, False

            logger.info("Scanned PDF detected — running OCR")
            pages = [
                self._ocr_to_string(self._prepare_ocr_image(p.to_image(resolution=200).original))
                for p in pages
            ]
        return "\n--- PAGE ---\n".join(pages), True

    # ══════════════════════════════════════════════════════════════════════════
    # DOCUMENT TYPE DETECTION
    # ══════════════════════════════════════════════════════════════════════════

    def _detect_type(self, file_lower: str, text: str) -> str:
        """
        Detection priority:
        1. Compound keyword check for plans (most specific — prevents false matches)
        2. Single keyword checks ordered specific → generic
        3. Content fallback (first 1500 chars)
        4. Default fiche

        WHY compound for plans?
        "plan" + "situation" or "plan" + "masse" — requiring BOTH words prevents
        a file like "fiche_plan_projet.pdf" from being misclassified as a plan.

        WHY "urbanisme" removed from autorisation keywords?
        "autorisation-d-urbanisme" is the correct signal, but "urbanisme" alone
        appears in plan filenames too ("plan-masse-urbanisme.pdf"). Using only
        "autorisation" and "permis" is precise enough.
        """
        # Priority 1 — compound plan checks (most specific)
        if "plan" in file_lower and "situation" in file_lower:
            return "plan_situation"
        if "plan" in file_lower and "masse" in file_lower:
            return "plan_masse"

        # Priority 2 — single keyword checks
        if any(kw in file_lower for kw in self.FICHE_KW):
            return "fiche"
        if any(kw in file_lower for kw in self.MANDAT_KW):
            return "mandat"
        if any(kw in file_lower for kw in self.CERTIFICAT_KW):
            return "certificat"
        if any(kw in file_lower for kw in self.AUTORISATION_KW):
            return "autorisation"

        # Priority 3 — content fallback
        early = text[:1500].lower()
        if "plan de situation" in early:                          return "plan_situation"
        if "plan de masse" in early or "plan masse" in early:    return "plan_masse"
        if re.search(r"\bfiche\b|\brenseignement\b", early): return "fiche"
        if re.search(r"\bmandat\b", early):                    return "mandat"
        if re.search(r"\bpermis\b|\barr[êe]t[ée]\b|\bautorisation\b", early):
            return "autorisation"

        return "fiche"

    # ══════════════════════════════════════════════════════════════════════════
    # FICHE
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_fiche(self, pdf_path: str, text: str, is_scanned: bool = False) -> dict:
        norm = re.sub(r"\s+", " ", text)
        m    = self.FICHE_PERMIT.search(norm)
        out  = {
            "ref_urbanisme": m.group(1).replace(" ", "").upper() if m else None,
            "dlpi":          None,
        }
        m = self.FICHE_DLPI.search(norm)
        if m:
            out["dlpi"] = re.sub(r"\s*/\s*", "/", m.group(1))

        # ── Layer 0.5: HIGH-PRIORITY PAIR EXTRACTION (before table parsing) ────
        # If we find a confident logements/locaux phrase, mark it as trusted source
        compact = re.sub(r"\s+", " ", text)
        pair = self._extract_logements_locaux_pair(compact)
        pair_found = pair is not None
        if pair:
            out["nb_logements_residentiels"], out["nb_locaux_pros"] = pair
            logger.info("High-priority pair found: %d logements, %d locaux", pair[0], pair[1])

        # ── Layer 1: pdfplumber structured table ──────────────────────────────
        if Path(pdf_path).suffix.lower() == ".pdf":
            table_result = self._extract_fiche_table(pdf_path)
            # Only use table results if pair extraction wasn't already successful
            if not pair_found:
                out.update(table_result)
            else:
                # Merge address only (never override count fields from pair)
                if table_result.get("adresse") and not out.get("adresse"):
                    out["adresse"] = table_result["adresse"]

        # ── Layer 2: regex pattern matching on digital text ────────────────────
        if not pair_found:
            self._fill_fiche_counts(out, compact)

        # ── Layer 3: OCR image fallback — handles screenshot tables, bad scan,
        #    cropped or non-standard templates where pdfplumber finds nothing ──
        counts_missing = (
            out.get("nb_logements_residentiels") is None or
            out.get("nb_locaux_pros") is None
        )
        # Trigger OCR refinement if:
        # 1. We found any counts but they include suspicious values (0/1 pro with many logements)
        # 2. Or counts are completely missing
        suspicious_pro_count = (
            pair_found is False and  # Was pair extraction even attempted?
            out.get("nb_locaux_pros") in (0, 1) and
            (out.get("nb_logements_residentiels") or 0) >= 5  # Lower threshold from 10 to 5
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

            el_results = self._extract_el_columns(source_text)
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
                    text=source_text[:3000],
                    fields=llm_fields,
                    instructions=(
                        "This is a French real estate form (Fiche de renseignement). "
                        "The table lists properties with columns for residential units "
                        "(logements résidentiels) and professional premises (locaux pros). "
                        "Extract the TOTAL count across ALL rows for each field. "
                        "nb_logements_residentiels: total number of residential units (integer). "
                        "nb_locaux_pros: total number of professional premises (integer). "
                        "Use null — NOT 0 — if you genuinely cannot find the value."
                    ),
                )
                for key in llm_fields:
                    if llm_out.get(key) is not None:
                        out[key] = llm_out[key]
                if llm_out:
                    out["llm_used"] = True

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
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    img = self._prepare_ocr_image(page.to_image(resolution=300).original)
                    pages_text.append(
                        self._ocr_to_string(img, config="--psm 6")
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
            header = [str(c or "").lower().strip() for c in table[0]]
            col = {}
            for i, cell in enumerate(header):
                # Residential: match "résidentiel/residentiel" that isn't also pro
                if (
                    ("résidentiel" in cell or "residentiel" in cell)
                    and not ("professionnel" in cell or "pro" in cell)
                    and not re.match(r"^el\b", cell)  # Exclude EL (boolean) column
                ):
                    col["nb_resid"] = i
                # Professional: numeric count columns only (avoid EL yes/no columns)
                # Must have both "locaux"/"cellule" AND ("pro"/"professionnel")
                elif (
                    ("locaux" in cell or "cellule" in cell)
                    and ("professionnel" in cell or "pro" in cell)
                    and not re.match(r"^el\b", cell)  # Exclude EL (boolean) column
                ):
                    col["nb_pro"] = i
                elif any(k in cell for k in ["adresse", "voie", "rue", "address"]):
                    col["adresse"] = i

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

            if result["nb_logements_residentiels"] is not None or result["nb_locaux_pros"] is not None:
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

        # Pass 3 — last resort: scan first table as flat text
        if result["nb_logements_residentiels"] is None and all_tables:
            flat = " ".join(str(c or "") for row in all_tables[0] for c in row)
            m    = re.search(r"Nb\s+total\s+de\s+logements[^\d:]*[:=]\s*(\d+)(?:\s|$|[^\w])", flat, re.I)
            if m:
                result["nb_logements_residentiels"] = int(m.group(1))

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
        hit = self.AU_PERMIT.search(text)
        if hit:
            return re.sub(r"[^A-Z0-9]", "", hit.group(1).upper())

        for line in text.splitlines():
            if not re.search(r"dossier|permis|arr[êe]t", line, re.I):
                continue
            cand = self.AU_PERMIT_FALLBACK.search(line)
            if cand:
                cleaned = re.sub(r"[^A-Z0-9]", "", cand.group(0).upper())
                if re.match(r"^(PC|PA|DP|CU)[0-9A-Z]{8,}$", cleaned):
                    return cleaned
        return None

    def _extract_autorisation(self, text: str) -> dict:
        out = {
            "ref_urbanisme": self._extract_au_ref(text),
            "llm_used":      True,
        }
        out["adresse"] = self._llm_extract(
            text=text[:2000],
            fields=["adresse"],
            instructions=(
                "Extract the project address — the physical location of the building or land "
                "the permit refers to. Include street number, street name, postal code and city. "
                "Do NOT return the applicant's home address."
                "Extract number of logements if you can find it with certainty, but do NOT guess or infer from context — we just want what is explicitly stated in the text. "
            ),
        ).get("adresse")
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

        return {
            **self._llm_extract(
                text=mandat_chunk[:3000],
                fields=fields,
                instructions=(
                    "This is a French FTTH mandate (Mandat de signature). "
                    "Orange is always the SECOND party — the \'Mandataire\'. "
                    "Extract ONLY the Orange representative\'s details — NOT the client (Mandant). "
                    "orange_representant_nom: full name. "
                    "orange_representant_fonction: job title at Orange. "
                    "orange_representant_mobile: phone number. "
                    "orange_representant_email: @orange.com email."
                ),
            ),
            "llm_used": True,
        }

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
            f"return 0 for any field you cannot find with certainty.\n"
            f"No markdown, no explanation, no text outside the JSON.\n\n"
            f"Document text:\n{text}"
        )
        try:
            raw = ollama.generate(model="qwen2.5:7b", prompt=prompt, options={"temperature": 0, "top_k": 10}).get("response", "")
            start = raw.find("{")
            end   = raw.rfind("}") + 1
            if start == -1 or end <= 0:
                logger.warning("LLM returned no JSON block")
                return {}
            payload = json.loads(raw[start:end])
            return {f: payload.get(f) for f in fields}
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
    _BASE = {"document_type": None, "scanned": False, "llm_used": False,
             "error": None, "present": False}

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

    def _safe_int(self, value: Any) -> int | None:
        s = re.sub(r"[^0-9\-]", "", str(value or "").strip())
        try:
            return int(s) if s else None
        except ValueError:
            return None

    def _detect_tesseract_languages(self) -> set[str]:
        """Read installed OCR languages once and cache in the extractor instance."""
        try:
            langs = set(pytesseract.get_languages(config=""))
            logger.info("Tesseract languages detected: %s", ", ".join(sorted(langs)) or "none")
            return langs
        except Exception as e:
            logger.warning("Could not query Tesseract languages: %s", e)
            return set()

    def _ocr_lang_candidates(self) -> list[str | None]:
        """Prefer French, then English, then default engine language."""
        langs = self._tesseract_langs
        candidates: list[str | None] = []
        if {"fra", "eng"}.issubset(langs):
            candidates.append("fra+eng")
        if "fra" in langs:
            candidates.append("fra")
        if "eng" in langs:
            candidates.append("eng")
        candidates.append(None)
        return candidates

    def _ocr_to_string(self, image: Image.Image, config: str | None = None) -> str:
        """Run OCR with language fallbacks so missing traineddata does not crash extraction."""
        for lang in self._ocr_lang_candidates():
            kwargs: dict[str, Any] = {}
            if lang:
                kwargs["lang"] = lang
            if config:
                kwargs["config"] = config
            try:
                return pytesseract.image_to_string(image, **kwargs)
            except pytesseract.TesseractError as e:
                logger.warning("OCR failed for lang=%s (%s)", lang or "default", e)
            except Exception as e:
                logger.warning("OCR unexpected failure for lang=%s (%s)", lang or "default", e)
        return ""

    def _prepare_ocr_image(self, image: Image.Image) -> Image.Image:
        """Normalize, auto-rotate, and binarize images before OCR."""
        prepared = ImageOps.exif_transpose(image)
        if prepared.mode not in ("L", "RGB"):
            prepared = prepared.convert("RGB")

        if prepared.mode != "L":
            prepared = ImageOps.grayscale(prepared)

        # Upscale small images so glyphs have more pixels to work with.
        if min(prepared.size) < 1600:
            prepared = prepared.resize(
                (prepared.width * 2, prepared.height * 2),
                Image.Resampling.LANCZOS,
            )

        prepared = ImageOps.autocontrast(prepared)
        prepared = prepared.filter(ImageFilter.MedianFilter(size=3))
        prepared = prepared.filter(ImageFilter.SHARPEN)
        prepared = ImageEnhance.Sharpness(prepared).enhance(1.8)
        prepared = ImageEnhance.Contrast(prepared).enhance(1.4)

        # Fix common 90/180/270 orientation issues before OCR.
        prepared = self._autorotate_for_ocr(prepared)

        # Adaptive thresholding preserves thin text better than a fixed cutoff.
        return self._adaptive_binarize(prepared)

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

    def _adaptive_binarize(self, gray: Image.Image) -> Image.Image:
        """Apply local adaptive thresholding, with a safe global fallback."""
        try:
            local_bg = gray.filter(ImageFilter.GaussianBlur(radius=8))
            binary = ImageMath.eval(
                "convert(a > (b - offset), '1')",
                a=gray,
                b=local_bg,
                offset=12,
            )
            return binary.convert("L")
        except Exception as e:
            logger.debug("Adaptive threshold fallback used: %s", e)
            return gray.point(lambda p: 255 if p > 185 else 0)

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
        units_kw = {"résidentiel", "residentiel", "professionnel"}
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