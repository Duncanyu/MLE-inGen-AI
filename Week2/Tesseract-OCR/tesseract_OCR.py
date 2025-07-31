from PIL import Image
import pytesseract

handwriting = Image.open('/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Tesseract-OCR/handwriting.png')
typewriting = Image.open('/Users/duncanyu/Documents/GitHub/DuncanYu-HW/Week2/Tesseract-OCR/typewriter.jpg')

print(pytesseract.image_to_string(handwriting))
print("=====")
print(pytesseract.image_to_string(typewriting))

# wow the text recognition is pretty bad huh