import fitz

doc = fitz.open('report/main.pdf')
print(f"Total pages: {len(doc)}")
for i, page in enumerate(doc):
    text = page.get_text()
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    first_line = lines[0] if lines else "EMPTY"
    last_line = lines[-1] if lines else "EMPTY"
    has_refs = "References" in text
    has_conclusion = "Conclusion" in text
    print(f"Page {i+1}: First='{first_line[:40]}' | Last='{last_line[:40]}' | HasRefs={has_refs} | HasConclusion={has_conclusion}")
