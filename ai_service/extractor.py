import pdfplumber
import re
import json
import logging
import ollama
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

class OrangeExtractor:
    """
    Hybrid Extraction Strategy:
    1. Regex (Fast/Cheap) for structured IDs and Dates.
    2. Table Parser (Precise) for the project summary table.
    3. LLM Fallback (Llama3) for unstructured mandates or messy scans.
    """

    # Matches permit references like: PC 123 456 78 A1234 (with loose separators)
    PERMIT_PATTERN = re.compile(
        r"\b((?:PC|PA|DP)[\s\-_/]*\d{3}[\s\-_/]*\d{3}[\s\-_/]*\d{2}[\s\-_/]*[A-Z]?[\s\-_/]*\d{4})\b",
        re.IGNORECASE
    )

    # Preferred when fiche contains a dedicated urbanism reference label.
    REF_URBANISME_PATTERN = re.compile(
        r"(?:R[ÉE]F(?:[ÉE]RENCE)?\s*URBANISME|N[°O]\s*PERMIS)\s*[:\-]?\s*"
        r"((?:PC|PA|DP)[\s\-_/]*\d{3}[\s\-_/]*\d{3}[\s\-_/]*\d{2}[\s\-_/]*[A-Z]?[\s\-_/]*\d{4})",
        re.IGNORECASE,
    )

    # Matches Date de Livraison
    DLPI_PATTERN = re.compile(
        r"(?:DATE\s+DE\s+LIVRAISON|LIVRAISON\s+DU\s+PROJET)\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{4})",
        re.IGNORECASE
    )

    FICHE_LLM_FALLBACK_ENABLED = os.getenv("FICHE_LLM_FALLBACK", "0") == "1"

    # Document type detection keywords
    MANDAT_KEYWORDS = {"mandat", "mandate"}
    AUTORISATION_KEYWORDS = {"autorisation", "authorization", "permit", "arrêté"}
    FICHE_KEYWORDS = {"fiche", "renseignement", "information sheet"}

    # Field definitions per document type
    FIELDS_BY_DOCTYPE = {
        "mandat": {
            "fields": ["syndic_or_promoter", "owner", "address", "is_authorized"],
            "prompt_suffix": (
                "Mandat is a legal document delegating authority. "
                "Extract: syndic or promoter name, owner name, property address, authorization status."
            ),
        },
        "autorisation": {
            "fields": ["permit_type", "applicant", "project_address", "decision", "decision_date"],
            "prompt_suffix": (
                "Autorisation d'urbanisme is a permit decision document. "
                "Extract: permit type (PC/PA/DP), applicant name, project address, decision (approved/rejected), decision date."
            ),
        },
        "fiche": {
            "fields": ["ref_urbanisme", "dlpi", "adresse_tableau", "nb_logements_residentiels"],
            "prompt_suffix": (
                "Fiche de renseignement is a structured property info sheet. "
                "Extract: urbanism reference number, delivery date, address from table, residential unit count."
            ),
        },
    }

    def extract(self, pdf_path: str, file_name: str = "") -> dict[str, Any]:
        start = time.perf_counter()
        result = self._empty_result()
        file_lower = file_name.lower()

        try:
            with pdfplumber.open(pdf_path) as pdf:
                if not pdf.pages:
                    result["error"] = "Empty PDF document"
                    return result

                # Zonal Header Extraction (Top 30%)
                first_page = pdf.pages[0]
                header_box = (0, 0, first_page.width, first_page.height * 0.3)
                header_text = first_page.within_bbox(header_box).extract_text() or ""
                full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)

                # Parse tables while PDF handle is still open to avoid a second full file pass.
                table_data = self._parse_orange_table(pdf.pages)

            # Detect document type from filename and content
            doc_type = self._detect_document_type(file_lower, header_text, full_text)
            result["document_type"] = doc_type

            # Strategy 1: Mandat (Unstructured, LLM Intelligence)
            if doc_type == "mandat":
                logger.info("Extracting mandat using LLM intelligence: %s", file_name)
                llm_fields = self.FIELDS_BY_DOCTYPE["mandat"]["fields"]
                llm_result = self._extract_via_llm(full_text[:3000], llm_fields, doc_type)
                result.update(llm_result)
                result["llm_used"] = True
                return result

            # Strategy 2: Autorisation d'urbanisme (Unstructured, LLM Intelligence)
            if doc_type == "autorisation":
                logger.info("Extracting autorisation using LLM intelligence: %s", file_name)
                llm_fields = self.FIELDS_BY_DOCTYPE["autorisation"]["fields"]
                llm_result = self._extract_via_llm(full_text[:3000], llm_fields, doc_type)
                result.update(llm_result)
                result["llm_used"] = True
                return result

            # Strategy 3: Fiche (Structured: Regex + Table, optional LLM fallback)
            scan_text = self._normalize_text(f"{header_text}\n{full_text[:8000]}")
            result["ref_urbanisme"] = self._extract_ref_urbanisme(scan_text)
            result["dlpi"] = self._extract_dlpi(scan_text)

            # Table Pass (Address and unit counts)
            result.update(table_data)

            # Optional LLM fallback for missing critical fields
            if self.FICHE_LLM_FALLBACK_ENABLED and (not result["ref_urbanisme"] or not result["dlpi"]):
                logger.info("Fiche regex incomplete, using LLM fallback: %s", file_name)
                llm_fields = self.FIELDS_BY_DOCTYPE["fiche"]["fields"]
                llm_data = self._extract_via_llm(full_text[:2000], llm_fields, doc_type)
                result.update({k: v for k, v in llm_data.items() if v})
                result["llm_used"] = True

            logger.info(
                "Extraction completed in %.2fs | file=%s | type=%s | llm=%s",
                time.perf_counter() - start,
                file_name,
                result.get("document_type"),
                result.get("llm_used"),
            )

            return result

        except Exception as e:
            logger.exception("Extraction failed for %s", pdf_path)
            result["error"] = str(e)
            return result

    def _detect_document_type(self, file_lower: str, header_text: str, full_text: str) -> str:
        """
        Detect document type with priority hierarchy:
        1. Filename-based detection (primary, most reliable)
        2. Header text patterns (secondary, for generic filenames)
        3. Default to fiche (fallback for ambiguous cases)
        
        This avoids false positives from content mentions (e.g., "mandat" in description).
        """
        # Priority 1: EXPLICIT FILENAME DETECTION (highly reliable signal)
        # These filename checks have highest confidence
        if any(kw in file_lower for kw in self.FICHE_KEYWORDS):
            logger.debug("Document type: fiche (detected from filename)")
            return "fiche"
        
        if any(kw in file_lower for kw in self.MANDAT_KEYWORDS):
            logger.debug("Document type: mandat (detected from filename)")
            return "mandat"
        
        if any(kw in file_lower for kw in self.AUTORISATION_KEYWORDS):
            logger.debug("Document type: autorisation (detected from filename)")
            return "autorisation"

        # Priority 2: HEADER/EARLY CONTENT DETECTION (secondary fallback)
        # Only scan first 1500 chars to minimize noise from mentions in content
        header_lower = header_text.lower()
        early_content = full_text[:1500].lower() if full_text else ""
        header_and_start = f"{header_lower} {early_content}"

        # Check for strong structural indicators (whole-word matches preferred)
        if any(self._contains_word(header_and_start, kw) for kw in ["fiche", "renseignement"]):
            logger.debug("Document type: fiche (detected from header/early content)")
            return "fiche"

        if any(self._contains_word(header_and_start, kw) for kw in self.MANDAT_KEYWORDS):
            logger.debug("Document type: mandat (detected from header/early content)")
            return "mandat"

        if any(self._contains_word(header_and_start, kw) for kw in self.AUTORISATION_KEYWORDS):
            logger.debug("Document type: autorisation (detected from header/early content)")
            return "autorisation"

        # Priority 3: DEFAULT FALLBACK
        logger.debug("Document type: fiche (default fallback)")
        return "fiche"

    def _contains_word(self, text: str, word: str) -> bool:
        """Check if word exists as a whole word (not substring) in text."""
        return re.search(rf"\b{re.escape(word)}\b", text, re.IGNORECASE) is not None

    def _parse_orange_table(self, pages: list[Any]) -> dict[str, Any]:
        """Specifically targets the 'Description Détaillée' table"""
        extracted = {"adresse_tableau": None, "nb_logements_residentiels": None}
        for page in pages:
            tables = page.extract_tables() or []
            for table in tables:
                # Look for address/lot rows in Orange summary table
                for row in table or []:
                    if not row:
                        continue
                    row_str = " ".join(str(c) for c in row if c).lower()
                    if "ruelles" in row_str or "lots" in row_str:
                        extracted["adresse_tableau"] = row[1] if len(row) > 1 else None
                        extracted["nb_logements_residentiels"] = row[2] if len(row) > 2 else None
                        return extracted
        return extracted

    def _extract_ref_urbanisme(self, text: str) -> str | None:
        labeled = self.REF_URBANISME_PATTERN.search(text)
        if labeled:
            return self._compact_permit(labeled.group(1))

        plain = self.PERMIT_PATTERN.search(text)
        if plain:
            return self._compact_permit(plain.group(1))
        return None

    def _extract_dlpi(self, text: str) -> str | None:
        match = self.DLPI_PATTERN.search(text)
        if not match:
            return None
        return match.group(1)

    def _compact_permit(self, value: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", value).upper()

    def _normalize_text(self, value: str) -> str:
        # Flatten repeated whitespace/newlines so regex behaves consistently.
        return re.sub(r"\s+", " ", value)

    def _extract_via_llm(self, text: str, fields: list[str], doc_type: str) -> dict[str, Any]:
        """
        Extract fields from unstructured French documents using Llama3 with context-aware prompts.
        
        Prompts are tailored to document type (mandat, autorisation, fiche) for higher accuracy.
        Returns JSON with requested fields; missing fields default to None.
        """
        doc_config = self.FIELDS_BY_DOCTYPE.get(doc_type, {})
        prompt_suffix = doc_config.get("prompt_suffix", "")

        prompt = (
            f"You are a document extraction expert. Extract ONLY the following JSON fields from this French {doc_type} document:\n"
            f"Fields: {json.dumps(fields)}\n\n"
            f"Context: {prompt_suffix}\n\n"
            f"Return ONLY a valid JSON object with these exact field names (use null for missing). "
            f"Do NOT include markdown, explanations, or extra text.\n\n"
            f"Document text:\n{text}"
        )

        try:
            logger.debug("LLM extraction for %s | fields=%s", doc_type, fields)
            resp = ollama.generate(
                model="llama3",
                prompt=prompt,
                options={"temperature": 0, "top_k": 10, "top_p": 0.9}
            )
            raw = resp.get("response", "")

            # Find JSON boundaries robustly
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                logger.warning("No valid JSON found in LLM response for %s", doc_type)
                return {}

            json_str = raw[start : end + 1]
            parsed = json.loads(json_str)

            # Ensure all requested fields exist (null for missing)
            result = {field: parsed.get(field) for field in fields}
            logger.info("LLM extracted %d/%d fields for %s", len([v for v in result.values() if v]), len(fields), doc_type)
            return result

        except json.JSONDecodeError as e:
            logger.warning("JSON parse error in LLM response: %s", e)
            return {}
        except Exception as e:
            logger.exception("LLM extraction failed for doc_type=%s", doc_type)
            return {}

    def _empty_result(self) -> dict[str, Any]:
        """Initialize result with all possible fields across all document types."""
        return {
            # Metadata
            "document_type": None,
            "llm_used": False,
            "error": None,
            # Fiche fields
            "ref_urbanisme": None,
            "dlpi": None,
            "adresse_tableau": None,
            "nb_logements_residentiels": None,
            # Mandat fields
            "syndic_or_promoter": None,
            "owner": None,
            "address": None,
            "is_authorized": None,
            # Autorisation fields
            "permit_type": None,
            "applicant": None,
            "project_address": None,
            "decision": None,
            "decision_date": None,
        }