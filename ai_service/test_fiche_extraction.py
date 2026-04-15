#!/usr/bin/env python3
"""Test script to debug fiche table extraction."""
import pdfplumber
import re
import json

# Path to the test PDF
pdf_path = "c:\\Users\\azizmohamed.miladi_a\\Desktop\\test_fiche.pdf"

try:
    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")
        
        for page_idx, page in enumerate(pdf.pages):
            print(f"=== PAGE {page_idx + 1} ===")
            tables = page.extract_tables()
            
            if not tables:
                print("No tables found on this page")
                continue
            
            print(f"Found {len(tables)} table(s)\n")
            
            for table_idx, table in enumerate(tables):
                print(f"\n--- TABLE {table_idx + 1} ---")
                print(f"Dimensions: {len(table)} rows x {len(table[0]) if table else 0} cols")
                
                # Print first 5 rows
                for row_idx, row in enumerate(table[:5]):
                    print(f"Row {row_idx}: {row}")
                
                if len(table) > 5:
                    print(f"... and {len(table) - 5} more rows")
                
                # Try to parse logement info from table text
                full_text = " ".join(str(cell or "") for row in table for cell in row)
                
                # Pattern 1: Direct "Nb total de logements/locaux/lots : X"
                match = re.search(r"Nb total de logements[^\d]*(\d+)", full_text, re.IGNORECASE)
                if match:
                    print(f"\n✓ Found logement count (pattern 1): {match.group(1)}")
                
                # Pattern 2: Look for "de logements" in header + number in data
                if "logements" in full_text.lower():
                    print(f"✓ Logements keyword found in table")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
