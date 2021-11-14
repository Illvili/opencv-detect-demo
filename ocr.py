import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('preview.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
result = image.copy()

# boxes = pytesseract.pytesseract.image_to_boxes(image, config='-c tessedit_char_whitelist=0123456789 --psm 6')
data = pytesseract.pytesseract.image_to_data(image, config='-c tessedit_char_whitelist=0123456789 --psm 6')
print(data)
# level   page_num        block_num       par_num line_num        word_num       left                              top      width   height  conf    text

for box in data.splitlines():
    level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text = box.split(' ')

    left = int(left)
    top = int(top)

    width = int(width)
    height = int(height)
    
    cv2.rectangle(result, (left, top), (left + width, top + height), (0, 0, 255), 1)
    cv2.putText(result, text, (left, top - 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.6, (0, 0, 255))

cv2.imshow('preview ocr', result)
cv2.waitKey(0)
