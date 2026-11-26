import fitz

doc = fitz.open('report/main.pdf')
for i, page in enumerate(doc):
    text = page.get_text()
    headings = [l for l in text.split('\n') if any(l.strip().startswith(prefix) for prefix in ['1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', 'Q1', 'Q2', 'Q3', 'Q4', 'Table ', 'Figure '])]
    print(f"=== Page {i+1} ===")
    for h in headings[:8]:
        print("  ", h.strip())
