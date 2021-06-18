FROM python:3.8

COPY . .

RUN pip install numpy pandas streamlit opencv-contrib-python pytesseract
RUN brew install tesseract

CMD ["python", "./ocr4.py"]