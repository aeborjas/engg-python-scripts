try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import sys

# print(sys.argv)
path = sys.argv[1]

text = pytesseract.image_to_string(Image.open(path))
print(text)
