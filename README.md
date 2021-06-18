# Table Extraction
This is a project dedicated to computer vision. The goal is to define a table on a given picture and extract information with respect to its structure. 
The project is the solution to the test task in Alpha Internship

## Table of Contents
* [Used libraries](#Used-Libraries)
* [Image upload](#Image-Upload)
* [Image preprocessing](#Image-Preprocessing)
* [Line extraction](#Line-Extraction)
* [Box extraction](#Box-Extraction)


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
Extracting horizontal and vertical lines and combining it to form a table structure.
```python3
length = np.array(img).shape[1]//100

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
horizontal_detect = cv2.erode(grey_scale, horizontal_kernel, iterations = 3)
hor_line = cv2.dilate(horizontal_detect, horizontal_kernel, iterations = 3)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations = 3)
ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations = 3)

final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
combine = cv2.addWeighted(ver_lines, 0.5, hor_line, 0.5, 0.0)
combine = cv2.erode(~combine, final, iterations = 2)
thresh, combine = cv2.threshold(combine, 128, 255, cv2.THRESH_BINARY)
convert_xor = cv2.bitwise_xor(img,combine)
inverse = cv2.bitwise_not(convert_xor)

st.header('Extracting table structure')
st.image(combine)
``` 
## Box Extraction
Then we need to extract single cells to use Tesseract. You can adjust thresholding using slider for a better result.

contours, hierarchy = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x : x[1][1]))
```python3
boxes = []
img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) 
w_thres = st.slider('Choose a better threshold for a box width',100,1000,500,10,help ='') 
for contour in contours:
  x, y, w, h = cv2.boundingRect(contour)
  if (w < w_thres and h < 500 and w > 10 and h > 10):

    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    boxes.append([x, y, w, h])
```
