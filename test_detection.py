#!/usr/bin/env python3
"""Test document type detection to identify the bug."""

from ai_service.extractor import OrangeExtractor

ex = OrangeExtractor()

# Test cases to reveal the bug
test_cases = [
    # (filename, header_snippet, content_snippet, expected_type, description)
    ("fiche_de_renseignement.pdf", "FICHE", "Une fiche contenant les details...", "fiche", "Classic fiche name"),
    ("fiche_123.pdf", "Property Info", "The property details...", "fiche", "Simple fiche"),
    ("mandat.pdf", "", "Authorization document", "mandat", "Obvious mandat"),
    ("mandat_syndic.pdf", "", "", "mandat", "Syndic mandat"),
    ("autorisation_urbanisme.pdf", "", "", "autorisation", "Urbanism permit"),
    # Edge case: fiche with word "mandat" in content (the bug!)
    ("FicheRenseignement_2024.pdf", "FICHE DE RENSEIGNEMENT", 
     "The property was under mandat of the previous owner. Details follow...", "fiche", "FICHE with 'mandat' in content"),
    ("prop_info.pdf", "FICHE", 
     "Ce mandat a été exécuté par le syndic selon les termes du contrat...", "fiche", "Generic name, fiche header, 'mandat' in content"),
]

print("=" * 100)
print("Testing detection logic to identify bugs:\n")
failed = 0
for fname, header, content, expected, desc in test_cases:
    detected = ex._detect_document_type(fname, header, content)
    status = "✓ PASS" if detected == expected else "✗ FAIL"
    if detected != expected:
        failed += 1
    print(f"{status:8} | {fname:30} | Expected: {expected:15} | Got: {detected:15}")
    print(f"         | Description: {desc}\n")

print("=" * 100)
print(f"Results: {len(test_cases) - failed}/{len(test_cases)} tests passed")
if failed > 0:
    print(f"⚠ {failed} detection bug(s) found - filename-based detection needs priority!\n")
