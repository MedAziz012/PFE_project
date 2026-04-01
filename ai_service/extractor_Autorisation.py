import pdfplumber
import re

class AutorisationExtractor:
    def __init__(self):
        # The 'fingerprint' of a permit number
        self.PERMIT_REGEX = r"\b(PC\s*\d{3}\s*\d{3}\s*\d{2}\s*[A-Z]\d{4})\b"
    def extract_autorisation(self, pdf_path: str):
        print(f"\n--- Processing: {pdf_path} ---", flush=True)
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[0]
        # Top 30% crop
        bbox = (0, 0, page.width, page.height * 0.3)
        header_text = page.within_bbox(bbox).extract_text() or ""
        
        # PRINT THE CROP CONTENT
        print(f"DEBUG HEADER TEXT:\n{header_text}", flush=True)
        
        match = re.search(self.PERMIT_REGEX, header_text, re.IGNORECASE)
        if match:
            ref = re.sub(r"\s+", "", match.group(1)).upper()
            print(f"MATCH FOUND: {ref}", flush=True)
            return {"ref_urbanisme": ref}
        
        print("NO MATCH FOUND IN HEADER", flush=True)
        return {"ref_urbanisme": None}

    def extract_autorisation(self, pdf_path: str):
        try:
            with pdfplumber.open(pdf_path) as pdf:
                page = pdf.pages[0]
                
                # --- THE OPTIMIZATION ---
                # Define a box: (left, top, right, bottom)
                # We take 100% of the width and only the top 30% of the height
                bounding_box = (0, 0, page.width, page.height * 0.3)
                
                # 'Crop' the page virtually
                header_area = page.within_bbox(bounding_box)
                header_text = header_area.extract_text() or ""
                
                # Search only in the cropped text
                match = re.search(self.PERMIT_REGEX, header_text, re.IGNORECASE)
                
                if match:
                    # Clean it up: "PC 044 168 23 T0014" -> "PC04416823T0014"
                    clean_ref = re.sub(r"\s+", "", match.group(1)).upper()
                    return {"ref_urbanisme": clean_ref, "status": "found_in_header"}
                
                return {"ref_urbanisme": None, "status": "not_in_header"}
                
        except Exception as e:
            return {"error": str(e)}
   
