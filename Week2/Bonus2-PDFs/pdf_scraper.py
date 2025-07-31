import pytesseract
from pdf2image import convert_from_path
import os

pdf_path = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Bonus2-PDFs/ceaselessly.pdf"
output_dir = "/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Bonus2-PDFs/pdf_ocr"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_txt_path = os.path.join(output_dir, "ceaselessly.txt")

images = convert_from_path(pdf_path)

text = ""
for page_num, img in enumerate(images, start=1):
    text += f"\n\n=== PAGE {page_num} ===\n\n"
    text += pytesseract.image_to_string(img)

with open(output_txt_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"text saved to {output_txt_path}")
