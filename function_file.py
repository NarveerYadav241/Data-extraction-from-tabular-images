import os
import cv2
import imutils
import numpy as np
from imutils import contours
from imutils.object_detection import non_max_suppression
import matplotlib.pyplot as plt
import pytesseract
import re
regex = r"\w|\d|\/|\.|\$|\,|\%|\ |\=|\:|\#"
matcher = re.compile(regex)
###############first function is border removal####################
def border_removal(original_image,top_border_threshold,left_border_threshold):
    orig_height, orig_width, _ = original_image.shape
    left_border_position = int(orig_width*left_border_threshold/100)
    top_border_position = int(orig_height*top_border_threshold/100)
    bottom_border_position = orig_height-top_border_position
    right_border_position = orig_width-left_border_position
    gray_image = cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist(gray_image, [0], None, [256], [0, 256])
    pixel_min = list(hist).index(np.min(hist))
    ret, min_pix_threshold = cv2.threshold(gray_image, pixel_min, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    image_col_sum_array = np.sum(min_pix_threshold, axis=0)
    image_row_sum_array = np.sum(min_pix_threshold,axis = 1)
    image_col_sum_array[left_border_position:right_border_position] = np.zeros((image_col_sum_array[left_border_position:right_border_position].shape))
    image_row_sum_array[top_border_position:bottom_border_position] = np.zeros((image_row_sum_array[top_border_position:bottom_border_position].shape))
    vertical_border_positions = np.where(image_col_sum_array // 255 > orig_height * 55 / 100)
    horizontal_border_positions = np.where(image_row_sum_array//255 > orig_width*50/100)
    # print(vertical_border_positions)
    # print(horizontal_border_positions)
    for white_col in vertical_border_positions[0]:
        original_image[:,white_col] = 255
    for white_row in horizontal_border_positions[0]:
        original_image[white_row,:] = 255
    # plt.plot(plot_vert,color="red")
    # plt.show()
    return original_image
####################This function is for table detection if a image have tables or is arranged as tables #################
def image_type(line_image_bwline,draw_image):
    cordinates_image = ~line_image_bwline.copy()
    cv_contours = cv2.findContours(line_image_bwline, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im_contours = imutils.grab_contours(cv_contours)
    final_contures = contours.sort_contours(im_contours, method="top-to-bottom")[0]
    boxes = []
    number_of_blocks = 0
    for (i, c) in enumerate(final_contures):
        box = cv2.boundingRect(c)
        h = box[3]
        boxes.append(box)
    for (x, y, w, h) in boxes:
        if w>40 and h>8:
            number_of_blocks = number_of_blocks + 1
            # cv2.rectangle(draw_image,(x,y),(x+w,y+h),(255,0,0),2)
    # cv2.imshow("detection_image",draw_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    indices = np.where(cordinates_image == [0])
    y_cordinates = np.unique(indices[0])
    x_cordinates = np.unique(indices[1])
    # print("cordinate",np.max(x_cordinates),np.max(y_cordinates))
    start_cordinates = [np.min(x_cordinates), np.min(y_cordinates)]
    end_cordinates = [np.max(x_cordinates), np.max(y_cordinates)]
    if number_of_blocks > 4:
        return True,start_cordinates,end_cordinates
    else:
        return False,start_cordinates,end_cordinates
def group_consecutives(vals, step=1):
    run = []
    result = [run]
    expect = np.min(vals)
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result
def image_table_regions(y_range,table_threshold):
    tables_range = list()
    for ranges in y_range:
        # print(len(ranges))
        # print("table_threshold",table_threshold)
        if (len(ranges) >= table_threshold):
            one_table_start_at = np.min(np.array(ranges))
            one_table_ends_at = np.max(np.array(ranges))
            tables_range.append([one_table_start_at,one_table_ends_at])
    return tables_range
def purify_lines(image,to_compare):
    cv_contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im_contours = imutils.grab_contours(cv_contours)
    final_contures = contours.sort_contours(im_contours, method="top-to-bottom")[0]
    boxes = []
    hori_ave = []
    for (i, c) in enumerate(final_contures):
        box = cv2.boundingRect(c)
        h = box[3]
        boxes.append(box)
        hori_ave.append(box[2])
    average_line = int(np.sum(np.array(hori_ave)) / len(hori_ave))
    for (x, y, w, h) in boxes:
        if to_compare == "w":
            if (w<average_line):
                image[y:y+h,x:x+w] = 0
        if to_compare == "h":
            if (h<average_line):
                image[y:y+h,x:x+w] = 0
    return image


#############################This function detect what is the range of tables in the image##########################
def table_range_cordinates(table_line_image):
    image_height,image_width,_ = table_line_image.shape
    gray_image = cv2.cvtColor(table_line_image, cv2.COLOR_BGR2GRAY)
    ret, binary_image_inv = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binary_image = cv2.bitwise_not(binary_image_inv)
    horizontal = binary_image_inv.copy()
    vertical = binary_image_inv.copy()
    row, cols = binary_image.shape
    horizontalsize = int(row / 40)
    horizontal_kernal = np.ones((1, int((3 * horizontalsize) / 4)), np.uint8)
    horizontal = cv2.erode(horizontal, horizontal_kernal, iterations=3)
    horizontal = cv2.dilate(horizontal, horizontal_kernal, iterations=5)
    vertical_kernal = np.ones((11, 1), np.uint8)
    vertical = cv2.erode(vertical, vertical_kernal, iterations=3)
    dilate_kernal = np.ones((11, 1), np.uint8)
    vertical = cv2.dilate(vertical, dilate_kernal, iterations=3)
    # pure_horizontal = purify_lines(horizontal.copy(),"w")
    # pure_vertical = purify_lines(vertical.copy(),"h")
    dilated_horizontal = cv2.dilate(horizontal, np.ones((3, 3), np.uint8), iterations=2)
    vertical_dilated = cv2.dilate(vertical, np.ones((3, 3), np.uint8), iterations=2)
    # cv2.imshow("dilated_hori",cv2.resize(pure_horizontal,None,fx = 0.3,fy=0.3))
    # cv2.imshow("horizontal",cv2.resize(horizontal,None,fx = 0.3,fy=0.3))
    # cv2.imshow("dilated_vert",pure_vertical)
    final_image = dilated_horizontal + vertical_dilated
    # cv2.imshow("final_image",final_image)
    indices = np.where(~final_image == [0])
    y_cordinates = np.unique(indices[0])
    y_regions = group_consecutives(y_cordinates)
    # plt.plot(y_cordinates)
    # plt.show()
    region_should_have_table = int(image_height / 10)
    table_ranges = image_table_regions(y_regions, region_should_have_table)
    return table_ranges,final_image,horizontal,vertical


########### Now we will detect all blocks cordinates which are in table################
def blocks_codinates_detection(horizontal_line,vertical_line,table_start_position,table_end_position,header_position = 0):
    #######line image is white with black lines####################
    # cv2.imshow("original_vertical",vertical_line)
    # cv2.imshow("rotated",rotated90)
    # print(table_start_position, table_end_position)
    # header_number = header_position
    table_startx = table_start_position[0]
    table_starty = table_start_position[1]
    table_endx = table_end_position[0]
    table_endy = table_end_position[1]
    column_sum = np.sum(~vertical_line,axis=0)
    column_position = np.where(column_sum != [0])
    column_position = np.unique(column_position)
    columns_group = group_consecutives(column_position)
    horizontal_positions = np.sum(~horizontal_line,axis=1)
    horizontal_positions = np.where(horizontal_positions!=[0])
    horizontal_positions = np.unique(horizontal_positions)
    horizontal_groups = group_consecutives(horizontal_positions)
    # print("horizontal_positions",horizontal_groups)
    # print("vertical_positons",columns_group)
    line_number = 0
    first_run = True
    first_col_run = True
    block_list = list()
    col_run = False
    if ((abs(horizontal_groups[0][0]-table_starty))>8):
        horizontal_groups.insert(0,[table_starty])
    elif ((abs(horizontal_groups[0][0]-table_starty))<10):
        first_line_missing = False
    if (abs(horizontal_groups[-1][0]-table_endy)>8):
        bottom_line_missing = True
        horizontal_groups.append([table_endy])
    elif (abs(horizontal_groups[-1][0]-table_endy)<10):
        bottom_line_missing = False
    if (abs(columns_group[0][0] - table_startx)>25):
        first_col_line_missing = True
        columns_group.insert(0,[table_startx])
    elif  (abs(columns_group[0][0] - table_startx)<25):
        first_col_line_missing = False
    if (abs(columns_group[-1][0] - table_endx)>25):
        right_line_missing = True
        columns_group.append([table_endx])
    elif (abs(columns_group[-1][0] - table_endx)<25):
        right_line_missing = False
    row_count = 0
    column_count = 0
    below_header = False
    for i in range(len(horizontal_groups)-1):
        block_starty = horizontal_groups[i][-1]
        block_endy = horizontal_groups[i+1][0]
        if ((i == header_position) or (below_header)):
            row_count = row_count + 1
            column_count = 0
            below_header = True
        for j in range(len(columns_group)-1):
            column_count = column_count + 1
            block_startx = columns_group[j][-1]
            block_endX = columns_group[j+1][0]
            block_list.append([block_startx,block_starty,block_endX,block_endy])
    return block_list,row_count,column_count,horizontal_groups,columns_group
def line_extractor(block_image):
    hist = cv2.calcHist(block_image, [0], None, [256], [0, 256])
    pixel_min = list(hist).index(np.min(hist))
    ret, binary_image_inv = cv2.threshold(block_image, pixel_min, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    raw_sum = np.sum(binary_image_inv, axis=1)
    raws_position = np.where(raw_sum != [0])
    raws_position = np.unique(raws_position)
    raws_ranges = group_consecutives(raws_position)
    return raws_ranges
def V1_parser(blocks_cord, rows_in_table, cols_in_table,hor_groups,cols_groups,data_image,header_num = 0):
    data_to_fill = np.zeros((rows_in_table,cols_in_table), dtype=object)
    is_header = False
    below_header = False
    row_count = -1
    for i in range(len(hor_groups)-1):
        block_starty = hor_groups[i][-1]
        block_endy = hor_groups[i+1][0]
        # row_count = 0
        if ((i == header_num) or (below_header)):
            row_count = row_count + 1
            column_count = 0
            below_header = True
            is_header = False
            if (i == header_num):
                is_header = True
        for j in range(len(cols_groups)-1):
            # column_count = column_count + 1
            if below_header:
                block_startx = cols_groups[j][-1]
                block_endX = cols_groups[j+1][0]
                # cv2.imshow("segment_image",segment_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                # segment_image = cv2.resize(segment_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                if is_header:
                    segment_image = data_image[block_starty:block_endy, block_startx:block_endX]
                    segment_image = cv2.cvtColor(segment_image, cv2.COLOR_BGR2GRAY)
                    resize_gray_image = cv2.resize(segment_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                    data = pytesseract.image_to_string(resize_gray_image, lang="eng+enm", config="--psm 7")
                    word = ''
                    # print(data)
                    matchlist = matcher.findall(data)
                    for character in matchlist:
                        word = word + character
                    if (len(word) >= 1 and word != " "):
                        data_to_fill[row_count][j] = word
                else:
                    segment_image = data_image[block_starty:block_endy, block_startx:block_endX]
                    border_removed_block = border_removal(segment_image,4,4)
                    segment_image = cv2.cvtColor(border_removed_block, cv2.COLOR_BGR2GRAY)
                    lines_cords = line_extractor(segment_image.copy())
                    word = ''
                    for one_line in lines_cords:
                        if (len(one_line)>3):
                            line_image = border_removed_block[one_line[0]-3:one_line[-1]+3,:]
                            white_border_line = cv2.copyMakeBorder(line_image,5,5,0,0,cv2.BORDER_CONSTANT,value=(255,255,255))
                            line_gray = cv2.cvtColor(white_border_line,cv2.COLOR_BGR2GRAY)
                            resize_gray_image = cv2.resize(line_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                            # cv2.imshow("line_image",line_image)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()
                            data = pytesseract.image_to_string(resize_gray_image, lang="eng+enm", config="--psm 7")
                            matchlist = matcher.findall(data)
                            for character in matchlist:
                                word = word + character
                        word = word + " "  ## previously tried \n
                    if (len(word) >= 1 and word != " "):
                        data_to_fill[row_count][j] = word
    return data_to_fill

