# Table Extraction
This is a project dedicated to computer vision. The goal is to define a table on a given picture and extract information with respect to its structure. 
The project is the solution to the test task in Alpha Internship

## How to make it work
I am currently working on creating a container for this app

Before it is done there are several necessary actions to make it work
```bash
pip3 install numpy
```

## Table of Contents
* [Used libraries](#Used-Libraries)
* [Image upload](#Image-Upload)
* [Image preprocessing](#Image-Preprocessing)
* [Line extraction](#Line-Extraction)
* [Box extraction](#Box-Extraction)
* [Creating rows](#Creating-Rows)
* [Equalizing rows](#Equalizing-Rows)
* [Applying Tesseract](#Applying-Tesseract)
* [Resultind dataframe](#Resulting-Dataframe)


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
```python3
contours, hierarchy = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x : x[1][1]))

boxes = []
img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) 
w_thres = st.slider('Choose a better threshold for a box width',100,1000,500,10,help ='') 
for contour in contours:
  x, y, w, h = cv2.boundingRect(contour)
  if (w < w_thres and h < 500 and w > 10 and h > 10):

    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    boxes.append([x, y, w, h])
```
## Creating Rows
A function needed to remove boxes in boxes
```python3
def drop_box(list_of_boxes):
	for box_1 in list_of_boxes:
		for box_2 in list_of_boxes:
			if box_1 == box_2:
				pass
			else:
				if box_1[0] == box_2[0] and box_1[1] == box_2[1]:
					if box_1[2]*box_1[3] > box_2[2]*box_2[3]:
						list_of_boxes.remove(box_1)
					else:
						list_of_boxes.remove(box_1)
	return list_of_boxes
```
The program organizes the boxes due to their height, creating rows out of boxes which have the same y value
```python3
boxes = drop_box(boxes)

boxes.sort(key = lambda x : x[1], reverse = True)
print('boxes', boxes)
row_y = []
col_x = []
for box in boxes:
	row_y.append(box[1])
	col_x.append(box[0])
print(set(row_y))

rows = []
row = []
for y in sorted(set(row_y)):
	for box in boxes:
		if abs(box[1] - y) < 5:
			row.append(box)
	rows.append(row)
	row = []
print(rows)

max_len = 0
for i in range(len(rows)):
	rows[i].sort(key=lambda x: x[0])
	if len(rows[i])>max_len:
		max_len=len(rows[i])

x_start = list(set(col_x))
x_start = sorted(x_start)
```
## Equalizing Rows
As we can see, rows are not equal in length. The folowwing can organize them with respect to their location in the original table
```python3
i=0
for row in rows:
	while i < len(row):
		#st.write(row[i][0])
		if row[i][0] !=  x_start[i]:

			row.insert(i,[x_start[i],None,None,None])
			i=i+1
		else:
			i=i+1
	i=0	
```
## Applying Tesseract
Now when we have boxes it is time to apply Tesseract and put everything in a dataframe
```python3
dataframe_final = []
df_row = []
df_col = []
for row in rows:
    for column in row:
        s = ''
        y, x, w, h = column[0], column[1], column[2], column[3]
        
        if x != None:
        
        	roi = im2[x - 1 : x + h + 1, y - 1 : y + w + 1]
        
        	
        	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        	resizing = cv2.resize(roi, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)
        	dilation = cv2.dilate(resizing, kernel,iterations = 1)
        	erosion = cv2.erode(dilation, kernel,iterations = 2)  
        	#st.image(erosion)

        	out = pytesseract.image_to_string(erosion, lang = 'rus')
        	if(len(out)==0):
        		out = pytesseract.image_to_string(erosion, lang = 'rus')
        	s = s + " " + out
        	df_row.append(s)
        	
        	
        	#st.write(s)
        else:
        	s= ' '
        	df_row.append(s)
        
        
    df_col.append(df_row)
    df_row = []
```
## Resulting Dataframe
```python3
dataframe = pd.DataFrame(df_col)
dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]

#print(dataframe)
st.header('Resulting table')
st.write(dataframe)

dataframe.to_excel("output.xlsx")
```
