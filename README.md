# Table Extraction
This is a project dedicated to computer vision. The goal is to define a table on a given picture and extract information with respect to its structure. 
The project is the solution to the test task in Alpha Internship

## Table of Contents
* [Used libraries](#Used-Libraries)
* [Image upload](#Image-Upload)
* [Image preprocessing](#Image-Preprocessing)
* [Line extraction](#Line-Extraction)


## Used Libraries
* NumPy
* Pandas 
* Cv2
* Pytesseract
* StreamLit 

## Image Upload
The StreamLit file uploader is used to drag and drop an image. A moment later it is grayscaled and showed to the user.
```python3
st.title('Table recognition')

uploaded_file = st.file_uploader("Choose a image file", type = ['jpg','jpeg','png'])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    st.image(img, width=None)
``` 

## Image Preprocessing
A threshold is applied to the image making pixels go black and white. You can adjust it using the slider.
```python3
thres = st.slider('Choose a better threshold',100,255,128,2,help ='usually it is nice to set 128')  #

convert_bin, grey_scale = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)

grey_scale = 255 - grey_scale

st.header('Inverted image')
st.image(grey_scale)
``` 
## Line Extraction
