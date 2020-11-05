import pytesseract
import sys
import os
from pdf2image import convert_from_path
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import glob
import time
try:
    from PIL import Image
except ImportError:
    import Image

# approach #1 - read simple image
# print(sys.argv)
# path = sys.argv[1]
# text = pytesseract.image_to_string(Image.open(path))

#approach #2 - read a bunch of images
#Part 2, recognizing text from the images using OCR
# filelimit = 26
# output_file = 'out_text.txt'
# with open(output_file, 'a') as f:
#     # for i in range(1, filelimit+1):
#     #     filename = f'page_{str(i)}.jpg'
#     #     text = str(pytesseract.image_to_string(Image.open(filename)))
#         # f.write(text)
#     text = str(pytesseract.image_to_string(Image.open('page_1.jpg')))
#     f.write(text)

# start at pg 95, end in pg 120
path = os.path.abspath(os.path.join(os.path.dirname(__file__),'Indirect Inspection Report Final Dec 2014.pdf'))

#Part 1, convert pages from PDF into images
# pages = convert_from_path(path, 300, first_page=95, last_page=120)

# image_counter = 1

# for page in pages:
#     filename= f'page_{str(image_counter)}.png'
#     print(filename)
#     page = page.rotate(90, expand=True)
#     page.save(filename, 'PNG')
#     image_counter += 1

for i in range(1,26+1):
    # Create directory
    dirName = f'page_{i}'
    try:
        # Create target Directory
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ") 
    except FileExistsError:
        # print("Directory " , dirName ,  " already exists")
        pass



## METHOD 2
def read_table_image(file, output_folder, output_name='output'):
    img = cv2.imread(file,0)
    scale_percent = 200 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #thresholding the image to a binary image
    thresh,img_bin = cv2.threshold(img,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #inverting the image 
    img_bin = 255-img_bin
    cv2.imwrite(os.path.join(output_folder,'cv_inverted.png'),img_bin)
    #Plotting the image to see the output
    plotting = plt.imshow(img_bin,cmap='gray')
    # plt.show()

    # countcol(width) of kernel as 100th of total width
    kernel_len = np.array(img).shape[1]//100
    # Defining a vertical kernel to detect all vertical lines of image 
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    # Defining a horizontal kernel to detect all horizontal lines of image
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    # A kernel of 2x2
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    #Use vertical kernel to detect and save the vertical lines in a jpg
    image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
    vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
    cv2.imwrite(os.path.join(output_folder,"vertical.png"),vertical_lines)
    #Plot the generated image
    plotting = plt.imshow(image_1,cmap='gray')
    # plt.show()

    #Use horizontal kernel to detect and save the horizontal lines in a jpg
    image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
    horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
    cv2.imwrite(os.path.join(output_folder,"horizontal.png"),horizontal_lines)
    #Plot the generated image
    plotting = plt.imshow(image_2,cmap='gray')
    # plt.show()

    # Combine horizontal and vertical lines in a new third image, with both having same weight.
    img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
    #Eroding and thesholding the image
    img_vh = cv2.erode(~img_vh, kernel, iterations=2)
    thresh, img_vh = cv2.threshold(img_vh,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_folder,"img_vh.png"), img_vh)
    bitxor = cv2.bitwise_xor(img,img_vh)
    bitnot = cv2.bitwise_not(bitxor)
    #Plotting the generated image
    plotting = plt.imshow(bitnot,cmap='gray')
    # plt.show()

    # Detect contours for following box detection
    contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    def sort_contours(cnts, method="left-to-right"):
        # initialize the reverse flag and sort index
        reverse = False
        i = 0
        # handle if we need to sort in reverse
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        # handle if we are sorting against the y-coordinate rather than
        # the x-coordinate of the bounding box
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return (cnts, boundingBoxes)

    # Sort all the contours by top to bottom.
    contours, boundingBoxes = sort_contours(contours, method="top-to-bottom")

    #Creating a list of heights for all detected boxes
    heights = [boundingBoxes[i][3] for i in range(len(boundingBoxes))]

    #Get mean of heights
    mean = np.mean(heights)

    #Create list box to store all boxes in  
    box = []
    # Get position (x,y), width and height for every contour and show the contour on image
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if (w<1000 and h<500):
            image = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            box.append([x,y,w,h])
            
    plotting = plt.imshow(image,cmap='gray')
    # plt.show()

    #Creating two lists to define row and column in which cell is located
    row=[]
    column=[]
    j=0

    #Sorting the boxes to their respective row and column
    for i in range(len(box)):    
            
        if(i==0):
            column.append(box[i])
            previous=box[i]    
        
        else:
            if(box[i][1]<=previous[1]+mean/2):
                column.append(box[i])
                previous=box[i]            
                
                if(i==len(box)-1):
                    row.append(column)        
                
            else:
                row.append(column)
                column=[]
                previous = box[i]
                column.append(box[i])
                
    # print(column)
    # print(row)

    #calculating maximum number of cells
    countcol = 0
    for i in range(len(row)):
        countcol = len(row[i])
        if countcol > countcol:
            countcol = countcol

    #Retrieving the center of each column
    center = [int(row[i][j][0]+row[i][j][2]/2) for j in range(len(row[i])) if row[0]]

    center=np.array(center)
    center.sort()
    # print(center)
    #Regarding the distance to the columns center, the boxes are arranged in respective order

    finalboxes = []
    for i in range(len(row)):
        lis=[]
        for k in range(countcol):
            lis.append([])
        for j in range(len(row[i])):
            diff = abs(center-(row[i][j][0]+row[i][j][2]/4))
            minimum = min(diff)
            indexing = list(diff).index(minimum)
            lis[indexing].append(row[i][j])
        finalboxes.append(lis)


    #from every single image-based cell/box the strings are extracted via pytesseract and stored in a list
    outer=[]
    for i in range(len(finalboxes)):
        for j in range(len(finalboxes[i])):
            inner=''
            if(len(finalboxes[i][j])==0):
                outer.append(' ')
            else:
                for k in range(len(finalboxes[i][j])):
                    y,x,w,h = finalboxes[i][j][k][0],finalboxes[i][j][k][1], finalboxes[i][j][k][2],finalboxes[i][j][k][3]
                    finalimg = bitnot[x:x+h, y:y+w]
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
                    border = cv2.copyMakeBorder(finalimg,2,2,2,2, cv2.BORDER_CONSTANT,value=[255,255])
                    resizing = cv2.resize(border, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    dilation = cv2.dilate(resizing, kernel,iterations=1)
                    erosion = cv2.erode(dilation, kernel,iterations=2)
                    
                    out = pytesseract.image_to_string(erosion)
                    if(len(out)==0):
                        out = pytesseract.image_to_string(erosion, config='--psm 3')
                    inner = inner +" "+ out
                outer.append(inner)

    #Creating a dataframe of the generated OCR list
    arr = np.array(outer)
    dataframe = pd.DataFrame(arr.reshape(len(row), countcol))
    # print(dataframe)
    data = dataframe.style.set_properties(align="left")
    #Converting it in a excel-file
    data.to_excel(os.path.join(output_folder,f"{output_name}.xlsx"))

if __name__ == '__main__':
    os.chdir(r'C:/Users/armando_borjas/Documents/Python/projects/interpret-PDF-table-image/')
    # following code runs the image processing for each .png file in the current folder
    # file = r'page_1.png'
    # s = time.time()
    # files = glob.glob('*.png')
    # names = list(map(lambda x: os.path.splitext(x)[0], files))
    # for file, name in zip(files,names):
    #     print(f'Processing {file}...', end=' ', flush=True)
    #     sf = time.time()
    #     read_table_image(file, name, output_name=name)
    #     print(f'Finished. {time.time()-sf:.4f} seconds.')

    # print(f'Completed. {time.time()-s:.4f} seconds.')
    
    # following code copies all excel files from each individual folder to the main folder.
    excel = glob.glob('./**/*.xlsx', recursive=True)
    import shutil
    for file in excel:
        shutil.copy(file,'.')
    print(excel)