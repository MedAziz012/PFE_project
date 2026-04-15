"""
Demand validation and recommendation system.

Validates that a development demand contains all required documents,
critical fields are consistent across documents, and provides consultant
recommendations with suggested corrective actions.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class DemandValidator:
    """
    Validates Orange PF development demands and provides recommendations.

    Required documents:
    1. Fiche de renseignement (project description)
    2. Plan de situation (location map)
    3. Plan de masse (site plan)
    4. Autorisation d'urbanisme (building permit)

    Optional documents:
    - Mandat (required only for collectif projects; not universally required)

    Cross-validation rules:
    - Permit reference (ref_urbanisme) must be identical in fiche and autorisation
    - Number of logements must match between fiche and autorisation
    """

    REQUIRED_DOCUMENT_TYPES = {"fiche", "plan_situation", "plan_masse", "autorisation"}

    def validate_demand(
        self, extractions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Validate a complete demand across multiple extracted documents.

        Args:
            extractions: dict mapping document type → extraction result dict
                Example:
                {
                    "fiche": {"ref_urbanisme": "PC...", "nb_logements_residentiels": 5},
                    "autorisation": {"ref_urbanisme": "PC...", "nb_logements_residentiels": 5},
                    "plan_situation": {...},
                    "plan_masse": {...},
                    "mandat": {...},  # optional
                }

        Returns:
            {
                "complete": bool,
                "missing_documents": [list of missing required types],
                "consistency_errors": [list of field mismatches],
                "is_valid": bool (complete AND no consistency errors),
                "recommendation": {
                    "verdict": str ("COMPLETE", "INCOMPLETE", "WARNING"),
                    "reasons": [list of issues],
                    "corrective_actions": [list of suggested actions],
                    "summary": str (human-readable one-liner)
                },
                "details": {...}
            }
        """
        result = {
            "complete": False,
            "missing_documents": [],
            "consistency_errors": [],
            "is_valid": False,
            "recommendation": {
                "verdict": "INCOMPLETE",
                "reasons": [],
                "corrective_actions": [],
                "summary": "",
            },
            "details": {},
        }

        # Step 1: Check completeness
        provided = set(k for k in extractions.keys() if extractions[k] is not None)
        missing = self.REQUIRED_DOCUMENT_TYPES - provided
        result["missing_documents"] = sorted(list(missing))
        result["complete"] = len(missing) == 0

        # Track reasons and corrective actions
        reasons = result["recommendation"]["reasons"]
        actions = result["recommendation"]["corrective_actions"]

        if missing:
            reasons.append(f"Missing required documents: {', '.join(sorted(missing))}")
            for doc in sorted(missing):
                actions.append(f"Request {doc.replace('_', ' ').title()} from applicant")

        # Step 2: Cross-validate fiche ↔ autorisation
        if "fiche" in extractions and "autorisation" in extractions:
            fiche = extractions.get("fiche", {})
            autorisation = extractions.get("autorisation", {})

            if fiche and autorisation:
                consistency = self._validate_consistency(fiche, autorisation)
                result["consistency_errors"] = consistency["errors"]
                result["details"]["fiche_autorisation_match"] = consistency

                if consistency["errors"]:
                    for error in consistency["errors"]:
                        reasons.append(error)

                    # Add corrective actions for specific inconsistencies
                    if not consistency["permit_match"]:
                        actions.append(
                            "Verify permit number (ref_urbanisme) matches between fiche and autorisation"
                        )
                    if not consistency["logement_match"]:
                        actions.append(
                            "Verify logement count matches between fiche and autorisation"
                        )

        # Step 3: Determine verdict and overall validity
        result["is_valid"] = result["complete"] and len(result["consistency_errors"]) == 0

        if result["is_valid"]:
            result["recommendation"]["verdict"] = "COMPLETE"
            result["recommendation"]["summary"] = (
                "✓ Demand is complete and valid. All required documents present "
                "and critical fields are consistent."
            )
        elif result["complete"] and result["consistency_errors"]:
            result["recommendation"]["verdict"] = "WARNING"
            result["recommendation"]["summary"] = (
                "⚠ Demand has all required documents but contains field inconsistencies. "
                "Review and correct mismatches before submission."
            )
        else:
            result["recommendation"]["verdict"] = "INCOMPLETE"
            result["recommendation"]["summary"] = (
                "✗ Demand is incomplete. Missing documents and/or field inconsistencies detected. "
                "Address all issues before submission."
            )

        return result

    def _validate_consistency(
        self, fiche: dict[str, Any], autorisation: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Cross-validate critical fields between fiche and autorisation.

        Returns:
            {
                "permit_match": bool,
                "logement_match": bool,
                "errors": [list of specific mismatches]
            }
        """
        errors = []
        checks = {
            "permit_match": False,
            "logement_match": False,
        }

        # Check 1: Permit reference must match
        fiche_permit = fiche.get("ref_urbanisme")
        auth_permit = autorisation.get("ref_urbanisme")

        if fiche_permit and auth_permit:
            if fiche_permit == auth_permit:
                checks["permit_match"] = True
            else:
                errors.append(
                    f"Permit number mismatch: fiche={fiche_permit}, autorisation={auth_permit}"
                )
        elif fiche_permit or auth_permit:
            errors.append(
                f"Permit number incomplete: fiche={fiche_permit}, autorisation={auth_permit}"
            )

        # Check 2: Number of logements must match
        fiche_logements = fiche.get("nb_logements_residentiels")
        auth_logements = autorisation.get("nb_logements_residentiels")

        if fiche_logements is not None and auth_logements is not None:
            if fiche_logements == auth_logements:
                checks["logement_match"] = True
            else:
                errors.append(
                    f"Logement count mismatch: fiche={fiche_logements}, "
                    f"autorisation={auth_logements}"
                )
        elif fiche_logements is not None or auth_logements is not None:
            logger.debug(
                "Logement count present in one document only: "
                "fiche=%s, autorisation=%s",
                fiche_logements,
                auth_logements,
            )

        return {
            "permit_match": checks["permit_match"],
            "logement_match": checks["logement_match"],
            "errors": errors,
        }

    def compare_extractions(
        self, extractions: dict[str, dict[str, Any]]
    ) -> dict[str, Any]:
        """
        Compare multiple extracted documents side-by-side.

        Args:
            extractions: dict mapping file identifier → extraction result dict
                Example:
                {
                    "file_0.pdf": {"ref_urbanisme": "PC...", "dlpi": "01/08/2026", ...},
                    "file_1.pdf": {"ref_urbanisme": "PC...", "dlpi": "01/08/2026", ...},
                }

        Returns:
            {
                "summary": {
                    "total_files": int,
                    "matching_fields": [list of fields that match across all files],
                    "mismatching_fields": [list of fields that differ],
                    "critical_inconsistencies": [high-priority mismatches],
                },
                "field_comparison": {
                    "field_name": {
                        "values": [list of values per file],
                        "match": bool,
                        "files_with_value": [list of files that have this field],
                        "mismatches": [{"file": "...", "value": ...}]
                    }
                },
                "extractions": {...}  (full extractions passed in)
            }
        """
        if not extractions:
            return {
                "summary": {
                    "total_files": 0,
                    "matching_fields": [],
                    "mismatching_fields": [],
                    "critical_inconsistencies": [],
                },
                "field_comparison": {},
                "extractions": extractions,
            }

        filenames = list(extractions.keys())
        total_files = len(filenames)

        # Collect all unique field names across extractions
        all_fields = set()
        for extraction in extractions.values():
            if extraction:
                all_fields.update(extraction.keys())

        # Critical fields to prioritize in mismatch detection
        critical_fields = {
            "ref_urbanisme",
            "nb_logements_residentiels",
            "nb_locaux_pros",
            "dlpi",
            "adresse",
        }

        field_comparison = {}
        matching_fields = []
        mismatching_fields = []
        critical_inconsistencies = []

        for field in sorted(all_fields):
            if field in ("error", "scanned", "llm_used", "document_type"):
                # Skip metadata fields
                continue

            values = []
            files_with_value = []
            mismatches = []

            for filename in filenames:
                extraction = extractions[filename]
                if extraction and field in extraction:
                    value = extraction[field]
                    values.append(value)
                    if value is not None:
                        files_with_value.append(filename)
                    if mismatches or (values and values[-1] != values[0]):
                        if value is not None and value != values[0]:
                            mismatches.append(
                                {"file": filename, "value": value}
                            )

            # Determine if field matches across files (ignoring nulls)
            non_null_values = [v for v in values if v is not None]
            field_matches = (
                len(set(str(v) for v in non_null_values)) <= 1
                if non_null_values
                else True
            )

            field_comparison[field] = {
                "values": values,
                "match": field_matches,
                "files_with_value": files_with_value,
            }
            if mismatches:
                field_comparison[field]["mismatches"] = mismatches

            # Categorize fields
            if field_matches:
                matching_fields.append(field)
            else:
                mismatching_fields.append(field)
                if field in critical_fields and mismatches:
                    critical_inconsistencies.append(
                        f"{field}: {', '.join(f'{m['file']}={m['value']}' for m in mismatches)}"
                    )

        return {
            "summary": {
                "total_files": total_files,
                "matching_fields": matching_fields,
                "mismatching_fields": mismatching_fields,
                "critical_inconsistencies": critical_inconsistencies,
            },
            "field_comparison": field_comparison,
            "extractions": extractions,
        }
