import numpy as np
import pandas as pd
import cv2
import pytesseract
import streamlit as st


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

#####################   UPLOADING AN IMAGE   ###########################

st.title('Table recognition')

uploaded_file = st.file_uploader("Choose a image file", type = ['jpg','jpeg','png'])
filename = uploaded_file.name
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype = np.uint8)
    img = cv2.imdecode(file_bytes, 0)
    st.image(img, width=None)




#####################   PREPROCESSING   ###########################


 

thres = st.slider('Choose a better threshold',100,255,180,2,help ='usually it is nice to set 128')  #

convert_bin, grey_scale = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
convert_bin, im2 = cv2.threshold(img, thres, 255, cv2.THRESH_BINARY)
grey_scale = 255 - grey_scale

st.header('Inverted image')
st.image(grey_scale)

#####################   EXTRACTING LINES   ###########################


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

#####################   EXTRACTING BOXES   ###########################


contours, hierarchy = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x : x[1][1]))

boxes = []
img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB) 
w_thres = st.slider('Choose a better threshold for a box width',100,2000,500,10,help ='') 
for contour in contours:
  x, y, w, h = cv2.boundingRect(contour)
  if (w < w_thres and h < 500 and w > 10 and h > 10):

    img = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    boxes.append([x, y, w, h])


st.image(img)

#####################   CREATING ROWS OF BOXES   ###########################
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

x_start=list(set(col_x))
x_start = sorted(x_start)
#st.write(x_start)

#####################  COOL ALGO TO EQUALIZE ROWS   ###########################
#####################  WITH RESPECT TO THEIR LOCATION   ###########################
i=0
for row in rows:
	while i < len(row):
		#st.write(row[i][0])
		if row[i][0] !=  x_start[i]:

			row.insert(i,[x_start[i],None,None,None])
			i=i+1
		else:
			i=i+1
	i=0		#st.write(i,len(row))


#for i in range(len(rows)):
#	if len(rows[i])<max_len:
#		for j in range(max_len-len(rows[i])):
#			rows[i].append([None,None,None,None])



#####################  EXTRACTING TEXT VIA PYTESSERACT  ###########################


st.spinner('loading')    

dataframe_final=[]
df_row = []
df_col = []



for row in rows:
    for column in row:
        s = ''
       
        

        y, x, w, h = column[0], column[1], column[2], column[3]
        
        if x != None:
        	if x == 0 or y == 0:
        		roi = im2[x : x + h, y : y + w ]
        	else:
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

st.success('Done!')


#####################  CREATING DATAFRAME  ###########################


dataframe = pd.DataFrame(df_col)
dataframe = dataframe.loc[:, ~dataframe.columns.duplicated()]

#print(dataframe)
st.header('Resulting table')
st.write(dataframe)

dataframe.to_excel(filename + ".xlsx")
