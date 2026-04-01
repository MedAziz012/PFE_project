#!/usr/bin/env python3
"""Extended detection tests covering edge cases and real-world scenarios."""

from ai_service.extractor import OrangeExtractor

ex = OrangeExtractor()

# Extended test suite with edge cases
test_cases = [
    # Standard cases
    ("fiche_de_renseignement_2024.pdf", "FICHE", "", "fiche", "Standard fiche filename"),
    ("mandat_syndic.pdf", "", "", "mandat", "Standard mandat filename"),
    ("autorisation_urbanisme_PC123.pdf", "", "", "autorisation", "Standard autorisation filename"),
    
    # Edge case: generic filename with clear header
    ("document_001.pdf", "FICHE DE RENSEIGNEMENT", "", "fiche", "Generic name, clear fiche header"),
    ("doc.pdf", "MANDAT DE VENTE", "", "mandat", "Generic name, clear mandat header"),
    ("permit.pdf", "ARRÊTÉ MUNICIPAL", "", "autorisation", "Generic name, arrêté header"),
    
    # Critical bug case: fiche with "mandat" mention in content
    ("fiche_property.pdf", "FICHE", "Ce bien était précédemment géré par mandat du syndic.", "fiche",
     "FICHE with 'mandat' word in content - must NOT be misclassified"),
    
    # Another bug case: generic file, header suggests fiche, but content has "mandat"
    ("property_info.pdf", "FICHE DE RENSEIGNEMENT IMMOBILIÈRE", 
     "L'ancien mandat du propriétaire précédent expire le 31/12/2025...", "fiche",
     "Generic filename, fiche header, 'mandat' in middle of content"),
    
    # Case sensitivity and variation
    ("fiche_2024.pdf", "", "", "fiche", "Fiche with capital F (lowercased by caller)"),
    ("mandat.pdf", "", "", "mandat", "MANDAT all caps (lowercased by caller)"),
    ("autorisation.pdf", "", "", "autorisation", "Autorisation capital A (lowercased by caller)"),
    
    # Content-heavy file
    ("unknown.pdf", "FICHE", 
     "Ce document contient des informations sur un mandat précédent, "
     "des détails d'autorisation antérieurs, n'importe quel texte...", "fiche",
     "Long content with multiple doc-type keywords, filename generic, header determines type"),
]

print("=" * 120)
print("Extended Detection Test Suite (Real-World Scenarios):\n")
failed = 0
for fname, header, content, expected, desc in test_cases:
    detected = ex._detect_document_type(fname, header, content)
    status = "✓ PASS" if detected == expected else "✗ FAIL"
    if detected != expected:
        failed += 1
    print(f"{status:8} | {fname:35} | Expected: {expected:15} | Got: {detected}")
    print(f"         | {desc}\n")

print("=" * 120)
print(f"Results: {len(test_cases) - failed}/{len(test_cases)} tests passed")
if failed == 0:
    print("✓ All detection logic is robust and working correctly!\n")
else:
    print(f"⚠ {failed} test(s) failed\n")
