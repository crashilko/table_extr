import numpy as np
import pandas as pd
import cv2
import pytesseract
from operator import itemgetter
import streamlit as st


st.title('Table recognition')

uploaded_file = st.file_uploader("Choose a image file", type=['jpg','jpeg','png'])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)

    # Now do something with the image! For example, let's display it:
    st.image(img, width=None)





#img = cv2.imread(uploaded_file, cv2.IMREAD_GRAYSCALE)
im2 = np.copy(img) 
thres = st.slider('Choose a better threshold',100,255,128,2,help ='usually it is nice to set 128')  #

convert_bin, grey_scale = cv2.threshold(img,thres,255,cv2.THRESH_BINARY)
grey_scale = 255-grey_scale

st.write('Your image')
st.image(grey_scale)
#cv2.imshow("image", grey_scale) #name the window as "image"
#cv2.waitKey(0)
#cv2.destroyWindow("image") #close the window

length = np.array(img).shape[1]//100
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

horizontal_detect = cv2.erode(grey_scale, horizontal_kernel, iterations=3)
hor_line = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)


#st.image(horizontal_detect)
#cv2.imshow("image", horizontal_detect) #name the window as "image"
#cv2.waitKey(0)
#cv2.destroyWindow("image") #close the window

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations=3)
ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)

#st.image(vertical_detect)
#cv2.imshow("image", vertical_detect) #name the window as "image"
#cv2.waitKey(0)
#cv2.destroyWindow("image") #close the window


final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
combine = cv2.addWeighted(ver_lines, 0.5, hor_line, 0.5, 0.0)
combine = cv2.erode(~combine, final, iterations=2)
thresh, combine = cv2.threshold(combine,128,255, cv2.THRESH_BINARY)
convert_xor = cv2.bitwise_xor(img,combine)
inverse = cv2.bitwise_not(convert_xor)

st.write('Table structure')
st.image(combine)
#cv2.imshow("image", combine) #name the window as "image"
#cv2.waitKey(0)
#cv2.destroyWindow("image") #close the window


contours, hierarchy = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
boundingBoxes = [cv2.boundingRect(contour) for contour in contours]
(contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),key=lambda x:x[1][1]))

boxes = []

for contour in contours:
  x, y, w, h = cv2.boundingRect(contour)
  if (w<500 and h<500 and w>10 and h>10):
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    boxes.append([x,y,w,h])


st.image(img)
#cv2.imshow("image", img) #name the window as "image"
#cv2.waitKey(0)
#cv2.destroyWindow("image") #close the window



boxes.sort(key=lambda x:x[1],reverse = True)
print('boxes',boxes)
row_y = []
for box in boxes:
	row_y.append(box[1])
print(set(row_y))



rows = []
row = []
for y in sorted(set(row_y)):
	for box in boxes:
		if abs(box[1] - y)<5:
			row.append(box)
	rows.append(row)
	row = []
print(rows)




max_len = 0
for i in range(len(rows)):
	rows[i].sort(key=lambda x: x[0])
	if len(rows[i])>max_len:
		max_len=len(rows[i])

for i in range(len(rows)):
	if len(rows[i])<max_len:
		for j in range(max_len-len(rows[i])):
			rows[i].append([None,None,None,None])






dataframe_final=[]
df_row = []
df_col = []

for row in rows:
    for column in row:
        s=''
       
        

        y,x,w,h = column[0],column[1], column[2],column[3]
        
        if x!=None:
        
        	roi = im2[x-1:x+h+1, y-1:y+w+1]
        

        	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        	resizing = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        	dilation = cv2.dilate(resizing, kernel,iterations=1)
        	erosion = cv2.erode(dilation, kernel,iterations=2)  
        

        	out = pytesseract.image_to_string(erosion,lang='rus')
        	if(len(out)==0):
        		out = pytesseract.image_to_string(erosion,lang='rus')
        	s = s +" "+ out
        	df_row.append(s)
        else:
        	s= ' '
        	df_row.append(s)
    df_col.append(df_row)
    df_row = []



#print(df_col)


dataframe = pd.DataFrame(df_col)
dataframe = dataframe.loc[:,~dataframe.columns.duplicated()]

#print(dataframe)
dataframe

dataframe.to_excel("output.xlsx")
