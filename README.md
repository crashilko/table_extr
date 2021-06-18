# Table Extraction
This is a project dedicated to computer vision. The goal is to define a table on a given picture and extract information with respect to its structure. 
The project is the solution to the test task in Alpha Internship

## Table of Contents
* [Used libraries](#Used-Libraries)
* [Image upload](#Image-Upload)
* [Image preprocessing](#Image-Preprocessing)


## Used Libraries
* NumPy
* Pandas 
* Cv2
* Pytesseract
* StreamLit 

## Image Upload
```python3
st.title('Table recognition')

uploaded_file = st.file_uploader("Choose a image file", type = ['jpg','jpeg','png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    st.image(img, width=None)
``` 

## Image Preprocessing


