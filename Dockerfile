FROM python:3.8

WORKDIR \Users\Kostik\OCR

COPY . .

RUN pip install numpy pandas streamlit opencv-contrib-python pytesseract

CMD ["python", "./ocr4.py"]