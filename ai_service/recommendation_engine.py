"""
Recommendation Engine — PF Folder Completeness Analysis

Architecture:
    Step 1: Rules engine  → finds all issues (deterministic, instant)
    Step 2: LLM           → writes professional French reasons (language only)

The LLM never makes decisions — it only phrases what the rules engine found.
This guarantees consistency and prevents hallucination on the verdict itself.

Key design principles:
  - BLOQUANT  → folder cannot be validated, verdict = INCOMPLET
  - AVERTISSEMENT → informational warning, does NOT affect verdict
  - "Document absent"  vs  "Document présent mais illisible" are distinct issues
  - Ref urbanisme comparison only runs when BOTH documents have the ref extracted
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class FolderDocuments:
    """
    Everything extracted from a PF folder's documents.

    extraction_failures: tracks fields that were NOT extracted from a document
    that IS present — so the rules engine can say "unreadable" instead of
    "missing document".

    Format:  { "fiche": ["nb_logements_residentiels", "adresse"], ... }
    """
    # ── Document presence ─────────────────────────────────────────────────────
    fiche_present:          bool = False
    plan_situation_present: bool = False
    plan_masse_present:     bool = False
    autorisation_present:   bool = False
    mandat_present:         bool = False   # optional

    # ── Fiche fields ──────────────────────────────────────────────────────────
    fiche_ref_urbanisme:    str | None = None
    fiche_dlpi:             str | None = None
    fiche_nb_logements_res: int | None = None
    fiche_nb_locaux_pro:    int | None = None
    fiche_adresse:          str | None = None

    # ── Autorisation fields ───────────────────────────────────────────────────
    autorisation_ref_urbanisme: str | None = None
    autorisation_adresse:       str | None = None

    # ── Mandat fields (optional document) ────────────────────────────────────
    mandat_orange_rep_nom:      str | None = None
    mandat_orange_rep_mobile:   str | None = None
    mandat_orange_rep_email:    str | None = None

    # ── Extraction failure tracking ───────────────────────────────────────────
    # Populated by _build_folder_documents in main.py.
    # Key = doc_type, Value = list of field names that returned None on a
    # present document. Lets the rules engine distinguish:
    #   • field is None because document was not uploaded  → "document absent"
    #   • field is None because extractor couldn't read it → "illisible"
    extraction_failures: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Issue:
    """
    A single problem found by the rules engine.

    code     : machine-readable identifier for the Angular frontend
    severity : BLOQUANT = blocks validation | AVERTISSEMENT = informational only
    details  : context dict passed to the LLM to phrase the reason
    """
    code:     str
    severity: str   # "BLOQUANT" | "AVERTISSEMENT"
    details:  dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    """
    Final output returned to Spring Boot → Angular.

    verdict  : "COMPLET" | "INCOMPLET"  (driven by BLOQUANT issues only)
    issues   : full list of Issue objects (BLOQUANT + AVERTISSEMENT)
    reasons  : professional French sentences — one per issue
    """
    verdict:  str
    issues:   list[Issue]
    reasons:  list[str]


# ══════════════════════════════════════════════════════════════════════════════
# RULES ENGINE — deterministic, no LLM
# ══════════════════════════════════════════════════════════════════════════════

class RulesEngine:
    """
    All checks are pure logic — no LLM, no I/O.
    Adding a new rule = one new _check_* method + call it from run().
    """

    REQUIRED_DOCUMENTS = [
        ("fiche_present",          "FICHE_MANQUANTE",         "Fiche de renseignement"),
        ("plan_situation_present", "PLAN_SITUATION_MANQUANT", "Plan de situation"),
        ("plan_masse_present",     "PLAN_MASSE_MANQUANT",     "Plan de masse"),
        ("autorisation_present",   "AUTORISATION_MANQUANTE",  "Autorisation d'urbanisme"),
    ]

    def run(self, docs: FolderDocuments) -> list[Issue]:
        """
        Run every check. BLOQUANT issues determine the verdict; AVERTISSEMENT
        issues are informational and do NOT flip the verdict to INCOMPLET.

        Order:
          1. Required document presence
          2. Cross-document ref consistency (only when both refs are available)
          3. Field-level completeness on present documents
        """
        issues = []

        # 1 ── Required document presence
        for attr, code, label in self.REQUIRED_DOCUMENTS:
            issue = self._check_document_present(docs, attr, code, label)
            if issue:
                issues.append(issue)

        # 2 ── Cross-document ref consistency
        #      Only runs when BOTH documents are present — no point checking
        #      refs if a document is already flagged as absent above.
        if docs.fiche_present and docs.autorisation_present:
            issue = self._check_permit_ref_match(docs)
            if issue:
                issues.append(issue)

        # 3 ── Field-level checks on each present document
        if docs.fiche_present:
            issues.extend(self._check_fiche_fields(docs))

        if docs.autorisation_present:
            issues.extend(self._check_autorisation_fields(docs))

        return issues

    # ── Individual checks ─────────────────────────────────────────────────────

    def _check_document_present(
        self, docs: FolderDocuments, attr: str, code: str, label: str
    ) -> Issue | None:
        if not getattr(docs, attr, False):
            return Issue(code=code, severity="BLOQUANT", details={"document": label})
        return None

    def _check_permit_ref_match(self, docs: FolderDocuments) -> Issue | None:
        """
        Compare permit refs only when BOTH were successfully extracted.

        Cases:
          • Both extracted + match      → no issue
          • Both extracted + mismatch   → BLOQUANT (real conflict)
          • One or both missing/unreadable → already caught by field checks
            below (AVERTISSEMENT) — no duplicate issue raised here
        """
        ref_fiche = self._normalise_ref(docs.fiche_ref_urbanisme)
        ref_au    = self._normalise_ref(docs.autorisation_ref_urbanisme)

        # Only compare when both are present
        if ref_fiche and ref_au and ref_fiche != ref_au:
            return Issue(
                code="REF_URBANISME_DISCORDANTE",
                severity="BLOQUANT",
                details={
                    "ref_fiche":        docs.fiche_ref_urbanisme,
                    "ref_autorisation": docs.autorisation_ref_urbanisme,
                },
            )
        return None

    def _check_fiche_fields(self, docs: FolderDocuments) -> list[Issue]:
        """
        For each expected fiche field, distinguish between:
          - Extracted = None because doc was not uploaded  → already a BLOQUANT above
          - Extracted = None because extractor failed      → AVERTISSEMENT "illisible"
          - Extracted = None because field absent on doc   → AVERTISSEMENT "absent"

        We detect the "extractor failed" case via extraction_failures.
        """
        issues = []
        extraction_failures = getattr(docs, "extraction_failures", {}) or {}
        fiche_failures = extraction_failures.get("fiche", [])

        checks = [
            ("fiche_nb_logements_res", "nb_logements_residentiels",
             "NB_LOGEMENTS_ILLISIBLE",   "NB_LOGEMENTS_MANQUANT",
             "nombre de logements résidentiels"),

            ("fiche_dlpi",             "dlpi",
             "DLPI_ILLISIBLE",          "DLPI_MANQUANTE",
             "date de livraison prévisionnelle (DLPI)"),

            ("fiche_adresse",          "adresse",
             "ADRESSE_ILLISIBLE",       "ADRESSE_MANQUANTE",
             "adresse du projet"),
        ]

        for attr, extractor_key, code_illisible, code_absent, label in checks:
            if getattr(docs, attr) is None:
                if extractor_key in fiche_failures:
                    # Document present, field found but unreadable by extractor
                    issues.append(Issue(
                        code=code_illisible,
                        severity="AVERTISSEMENT",
                        details={"champ": label, "document": "fiche de renseignement"},
                    ))
                else:
                    # Field simply absent from the document
                    issues.append(Issue(
                        code=code_absent,
                        severity="AVERTISSEMENT",
                        details={"champ": label, "document": "fiche de renseignement"},
                    ))

        # Ref urbanisme on fiche: only warn if autorisation has one (so comparison
        # was possible but fiche was missing it). If autorisation also lacks it,
        # that's caught in _check_autorisation_fields.
        if docs.fiche_ref_urbanisme is None and docs.autorisation_ref_urbanisme is not None:
            code = ("REF_URBANISME_ILLISIBLE_FICHE"
                    if "ref_urbanisme" in fiche_failures
                    else "REF_URBANISME_ABSENTE_FICHE")
            issues.append(Issue(
                code=code,
                severity="AVERTISSEMENT",
                details={"document": "fiche de renseignement"},
            ))

        return issues

    def _check_autorisation_fields(self, docs: FolderDocuments) -> list[Issue]:
        """
        Autorisation ref is required for folder validation.
        If it's present but unreadable → BLOQUANT with a clear "illisible" code.
        If it's simply absent from the document → BLOQUANT "manquante".
        """
        issues = []
        extraction_failures = getattr(docs, "extraction_failures", {}) or {}
        au_failures = extraction_failures.get("autorisation", [])

        if docs.autorisation_ref_urbanisme is None:
            code = ("REF_URBANISME_ILLISIBLE_AUTORISATION"
                    if "ref_urbanisme" in au_failures
                    else "REF_URBANISME_MANQUANTE_AUTORISATION")
            issues.append(Issue(
                code=code,
                severity="BLOQUANT",
                details={"document": "autorisation d'urbanisme"},
            ))

        return issues

    # OCR commonly misreads the suffix letter (position 10) as a digit.
    # Same map as OrangeExtractor._OCR_DIGIT_TO_LETTER.
    _OCR_DIGIT_TO_LETTER = {"2": "Z", "8": "B", "0": "O", "1": "I", "6": "G", "5": "S"}

    @classmethod
    def _normalise_ref(cls, ref: str | None) -> str | None:
        """
        Normalise permit references for comparison — handles 3 real-world issues:

        1. Spaces/dashes between groups: "PC 033 200 24 Z0041" → "PC03320024Z0041"
        2. 2-digit dept (zero-pad):      "PC6448324B0041"      → "PC06448324B0041"
        3. OCR noise suffix:             "PC06448324B0041ST"    → "PC06448324B0041"
        4. OCR digit at suffix position: "PC03320024 20041"     → "PC03320024Z0041"
           (Z misread as 2 — very common with Tesseract on scanned docs)

        WHY this matters for comparison:
        The fiche ref is extracted from a digital PDF (pdfplumber, perfect text).
        The autorisation ref is extracted via OCR from a scan (Tesseract, imperfect).
        Without normalization, identical refs like PC03320024Z0041 and PC0332002420041
        are treated as different, incorrectly flagging the folder as INCOMPLET.
        """
        if not ref:
            return None
        clean = re.sub(r"[^A-Z0-9]", "", ref.upper())
        prefix = clean[:2] if len(clean) >= 2 else ""
        if prefix in ("PC", "PA", "DP", "CU"):
            if len(clean) == 14:           # 2-digit dept — zero-pad
                clean = clean[:2] + "0" + clean[2:]
            if len(clean) > 15:            # OCR noise — truncate to 15
                clean = clean[:15]
            # Fix OCR digit at suffix letter position (always index 10)
            if len(clean) == 15 and clean[10].isdigit():
                letter = cls._OCR_DIGIT_TO_LETTER.get(clean[10])
                if letter:
                    clean = clean[:10] + letter + clean[11:]
        return clean


# ══════════════════════════════════════════════════════════════════════════════
# LLM REASONER — language only, no decisions
# ══════════════════════════════════════════════════════════════════════════════

class LLMReasoner:
    """
    Converts Issue objects into professional French sentences.
    Never decides verdict — that is the rules engine's job.
    """

    # Fallback templates used when LLM is unavailable.
    # They are also injected into the LLM prompt as context so it understands
    # exactly what each code means.
    ISSUE_DESCRIPTIONS = {
        # ── Document absent ───────────────────────────────────────────────────
        "FICHE_MANQUANTE":                        "La fiche de renseignement est absente du dossier.",
        "PLAN_SITUATION_MANQUANT":                "Le plan de situation est absent du dossier.",
        "PLAN_MASSE_MANQUANT":                    "Le plan de masse est absent du dossier.",
        "AUTORISATION_MANQUANTE":                 "L'autorisation d'urbanisme est absente du dossier.",
        # ── Ref urbanisme ─────────────────────────────────────────────────────
        "REF_URBANISME_DISCORDANTE":              "La référence du permis ne correspond pas entre la fiche et l'autorisation.",
        "REF_URBANISME_MANQUANTE_AUTORISATION":   "La référence d'urbanisme est absente de l'autorisation d'urbanisme.",
        "REF_URBANISME_ILLISIBLE_AUTORISATION":   "La référence d'urbanisme est présente sur l'autorisation mais n'a pas pu être lue par le système.",
        "REF_URBANISME_ABSENTE_FICHE":            "La référence d'urbanisme n'est pas renseignée sur la fiche (non bloquant).",
        "REF_URBANISME_ILLISIBLE_FICHE":          "La référence d'urbanisme est présente sur la fiche mais n'a pas pu être lue par le système (non bloquant).",
        # ── Fiche fields — absent ─────────────────────────────────────────────
        "NB_LOGEMENTS_MANQUANT":                  "Le nombre de logements n'est pas renseigné sur la fiche de renseignement.",
        "DLPI_MANQUANTE":                         "La date de livraison prévisionnelle (DLPI) n'est pas renseignée sur la fiche.",
        "ADRESSE_MANQUANTE":                      "L'adresse du projet n'est pas renseignée sur la fiche.",
        # ── Fiche fields — present but unreadable ────────────────────────────
        "NB_LOGEMENTS_ILLISIBLE":                 "Le nombre de logements est présent sur la fiche mais n'a pas pu être lu (tableau illisible ou en image).",
        "DLPI_ILLISIBLE":                         "La date de livraison (DLPI) est présente sur la fiche mais n'a pas pu être lue par le système.",
        "ADRESSE_ILLISIBLE":                      "L'adresse du projet est présente sur la fiche mais n'a pas pu être lue par le système.",
    }

    def write_reasons(self, issues: list[Issue]) -> list[str]:
        """
        Returns one reason string per issue.
        Falls back to template descriptions if LLM unavailable.
        """
        if not issues:
            return []
        try:
            return self._llm_write_reasons(issues)
        except Exception as e:
            logger.warning("LLM unavailable, using template reasons: %s", e)
            return self._template_reasons(issues)

    def _llm_write_reasons(self, issues: list[Issue]) -> list[str]:
        import os
        from openai import OpenAI

        issue_lines = []
        for issue in issues:
            desc = self.ISSUE_DESCRIPTIONS.get(issue.code, issue.code)
            if issue.details:
                details_str = ", ".join(f"{k}={v}" for k, v in issue.details.items())
                desc = f"{desc} ({details_str})"
            issue_lines.append(f"- [{issue.severity}] {desc}")

        prompt = f"""Tu es un conseiller immobilier Orange expert.
Pour chaque problème listé ci-dessous, rédige UNE phrase courte et professionnelle en français.

Règles:
- Une phrase par problème, dans le même ordre
- Distingue clairement "document absent du dossier" de "document présent mais illisible"
- Pour les AVERTISSEMENT, précise que ce n'est pas bloquant
- Ton factuel, pas de formules de politesse
- Réponds UNIQUEMENT avec un tableau JSON de chaînes de caractères, une entrée par problème

Problèmes ({len(issues)}):
{chr(10).join(issue_lines)}

Format attendu (exactement {len(issues)} entrées):
{json.dumps(["Raison " + str(i+1) for i in range(len(issues))])}"""

        client = OpenAI(
            api_key=os.environ["GROQ_API_KEY"],
            base_url="https://api.groq.com/openai/v1",
        )
        chat = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw   = chat.choices[0].message.content or ""
        start = raw.find("[")
        end   = raw.rfind("]") + 1

        if start == -1 or end <= 0:
            logger.warning("LLM returned no JSON array")
            return self._template_reasons(issues)

        parsed = json.loads(raw[start:end])
        # Guard: LLM sometimes returns fewer items than expected
        if len(parsed) < len(issues):
            parsed += self._template_reasons(issues[len(parsed):])
        return parsed

    def _template_reasons(self, issues: list[Issue]) -> list[str]:
        return [self.ISSUE_DESCRIPTIONS.get(i.code, i.code) for i in issues]


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDATION ENGINE — orchestrates rules + LLM
# ══════════════════════════════════════════════════════════════════════════════

class RecommendationEngine:

    def __init__(self):
        self.rules    = RulesEngine()
        self.reasoner = LLMReasoner()

    def analyze(self, docs: FolderDocuments) -> Recommendation:
        """
        Step 1: rules engine finds all issues.
        Step 2: verdict = INCOMPLET only when at least one BLOQUANT exists.
        Step 3: LLM writes one French sentence per issue.
        """
        issues   = self.rules.run(docs)
        bloquants = [i for i in issues if i.severity == "BLOQUANT"]
        verdict  = "COMPLET" if not bloquants else "INCOMPLET"
        reasons  = self.reasoner.write_reasons(issues) if issues else []

        logger.info(
            "Verdict: %s | total=%d bloquants=%d avertissements=%d",
            verdict, len(issues), len(bloquants),
            len(issues) - len(bloquants),
        )
        return Recommendation(verdict=verdict, issues=issues, reasons=reasons)

    def to_dict(self, rec: Recommendation) -> dict:
        return {
            "verdict": rec.verdict,
            "issues": [
                {"code": i.code, "severity": i.severity, "details": i.details}
                for i in rec.issues
            ],
            "reasons": rec.reasons,
        }