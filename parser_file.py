import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import Counter
import lxml.etree as ET
import pytesseract
import re
regex = r"\w|\d|\/|\.|\$|\,|\%|\ |\=|\:|\#"
matcher = re.compile(regex)
imagexml = ET.Element("image")
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


def image_reader_parser(table_data_xml,tess_image,header_position=0):
    total_lines_including_header = table_data_xml.xpath("//image//table_data//table_line")
    blocks_in_header_line = table_data_xml.xpath("//image//table_data//table_line[@type='lineHeader']/block")
    lines_without_header = table_data_xml.xpath("//image//table_data//table_line[@type!='lineHeader']")
    total_blocks_in_table = table_data_xml.xpath("//image//table_data//table_line//block")
    blocks_without_header = table_data_xml.xpath("//image//table_data//table_line[@type!='lineHeader']/block")
    cols = []
    dataframe_cols = 0
    mean_line=0
    header_mean = []
    dataframe_rows = len(total_lines_including_header)
    for i in range(len(total_lines_including_header)-1):
        line = total_lines_including_header[i]
        cols.append(int(line.attrib['total_blocks']))
    dataframe_cols = max(cols)
    max_cols_index = cols.index(max(cols))
    if max_cols_index<4:
        mean_line = max_cols_index
    else:
        mean_line=header_position
    for i in range(len(total_lines_including_header)-1):
        line = total_lines_including_header[i]
        if i == mean_line:
            mean_blocks = line.getchildren()
            for block in mean_blocks:
                header_mean.append(int(int(block.attrib['startX'])+int(abs(int(block.attrib['endX'])-int(block.attrib['startX']))/2)))
            break
    data_to_fill = np.full((dataframe_rows,dataframe_cols),np.nan, dtype=object)
    for block in total_blocks_in_table:
        block_startx = int(block.attrib['startX'])
        block_endx = int(block.attrib['endX'])
        block_startY = int(block.attrib['startY'])
        block_endy = int(block.attrib['endY'])
        parent_type = block.getparent().values()[1]
        word_mean = block_startx+int(abs(block_endx-block_startx)/2)
        info_list = list()
        dataFill = False
        line_number = int(block.getparent().values()[0])
        if (parent_type == "lineHeader"):
            segment_image = tess_image[block_startY-2:block_endy+2,block_startx-2:block_endx+2]
            bordered_image = cv2.copyMakeBorder(segment_image,5,5,5,5,cv2.BORDER_CONSTANT,value=(255,255,255))
            segment_gray = cv2.cvtColor(bordered_image,cv2.COLOR_BGR2GRAY)
            resize_gray_image = cv2.resize(segment_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            data = pytesseract.image_to_string(resize_gray_image, lang="eng+enm", config="--psm 7")
            word = ''
            matchlist = matcher.findall(data)
            for character in matchlist:
                word = word + character
            if (len(word) >= 1 and word != " "):
                dataFill = True
                info_list = [word,word_mean]
        elif (parent_type == "line"):
            segment_image = tess_image[block_startY - 3:block_endy + 3, block_startx - 3:block_endx + 3]
            bordered_image = cv2.copyMakeBorder(segment_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            segment_gray = cv2.cvtColor(bordered_image, cv2.COLOR_BGR2GRAY)
            resize_gray_image = cv2.resize(segment_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            data = pytesseract.image_to_string(resize_gray_image, lang="eng+enm", config="--psm 7")
            word = ''
            matchlist = matcher.findall(data)
            for character in matchlist:
                word = word + character
            if (len(word) >= 1 and word != " "):
                dataFill = True
                info_list = [word, word_mean]

        if (dataFill):
            x_mean_np = np.array(header_mean)
            subarray = []
            subnp = np.subtract(x_mean_np, info_list[1]).tolist()
            min_index = 0
            for subvalue in subnp:
                subarray.append(abs(subvalue))
                min_index = subarray.index(min(subarray))
            data_to_fill[line_number-1][min_index] = info_list[0]
            dataFill = False

    return data_to_fill

def image_reading_converting(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist(gray_image, [0], None, [256], [0, 256])
    pixel_min = list(hist).index(np.min(hist))
    ret, binary_image_inv = cv2.threshold(gray_image, pixel_min, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ret,binary_image = cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)
    # cv2.imshow("binary_image",binary_image)
    binary_image = cv2.bitwise_not(binary_image_inv)
    horizontal = binary_image_inv.copy()
    vertical = binary_image_inv.copy()
    row, cols = binary_image.shape
    horizontalsize = int(row / 10)
    horizontal_kernal = np.ones((1, int((3 * horizontalsize) / 4)), np.uint8)
    horizontal = cv2.erode(horizontal, horizontal_kernal, iterations=3)
    horizontal = cv2.dilate(horizontal, horizontal_kernal, iterations=5)
    vertical_kernal = np.ones((11, 1), np.uint8)
    vertical = cv2.erode(vertical, vertical_kernal, iterations=3)
    dilate_kernal = np.ones((11, 1), np.uint8)
    vertical = cv2.dilate(vertical, dilate_kernal, iterations=3)
    final_image = horizontal + vertical
    # cv2.imshow("table_image",final_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    final_image = ~final_image
    and_image = cv2.bitwise_and(final_image, binary_image_inv)
    return and_image
def block_generator(and_image,header_pos=0):
    raw_sum = np.sum(and_image, axis=1)
    raws_position = np.where(raw_sum != [0])
    raws_position = np.unique(raws_position)
    raws_ranges = group_consecutives(raws_position)
    # text_box = []
    block_startx = 0
    block_endx = 0
    header_count = -1
    below_header = False
    is_kind_header = False
    fill_data = True
    header_run = False
    imagedata = ET.Element("table_data")
    for raw in raws_ranges:
        # cv2.line(image,(0,raw[0]-2),(image_width,raw[0]-2),(0,0,255),1)
        # cv2.line(image,(0,raw[-1]+2),(image_width,raw[-1]+2),(0,0,255),1)
        if (len(raw) > 5):
            header_count = header_count + 1
            word_count = 0
            total_blocks = []
            if ((header_count == header_pos) or below_header):
                fill_data = False
                line_width = abs(raw[-1] - raw[0])
                line_image = and_image[raw[0]:raw[-1], :]
                column_sum = np.sum(line_image, axis=0)
                word_position = np.where(column_sum != [0])
                word_position = np.unique(word_position)
                below_header = True
                if (header_run):
                    paraXml = ET.Element('table_line',type='line',number=str(header_count))
                if (header_count == header_pos):
                    paraXml = ET.Element('table_line',type='lineHeader',number = str(header_count))
                    header_run = True
                count = 0
                for i in range(len(word_position) - 1):
                    if (abs(word_position[i + 1] - word_position[i]) < line_width * 65 / 100):
                        count = count + 1
                        fill_data = False
                    if (i == len(word_position) - 2):
                        block_startx = word_position[(len(word_position) - 1) - count]
                        block_endx = word_position[-1]
                        # text_box.append([block_startx, raw[0], block_endx, raw[-1]])
                        count = 0
                        fill_data = True
                        word_count = word_count + 1
                        total_blocks.append(word_count)
                        blockXml = ET.Element('block',line_number = str(header_count),number = str(word_count),startX=str(block_startx),
                                              startY=str(raw[0]),endX=str(block_endx),endY=str(raw[-1]))
                    if (abs(word_position[i + 1] - word_position[i]) > line_width * 65 / 100):  ##510 536
                        block_startx = word_position[i - count]
                        block_endx = word_position[i]
                        count = 0
                        fill_data = True
                        word_count= word_count+1
                        total_blocks.append(word_count)
                        # text_box.append([block_startx, raw[0], block_endx, raw[-1]])
                        blockXml = ET.Element('block',line_number = str(header_count),number = str(word_count),startX=str(block_startx),
                                              startY=str(raw[0]),endX=str(block_endx),endY=str(raw[-1]))
                        # if (is_kind_header):
                        #     data_to_fill_cols = data_to_fill_cols + 1
                        #     mean_block = block_startx + int(abs(block_endx - block_startx) / 2)
                        #     header_blocks_x_mean.append(mean_block)
                    if fill_data:
                        paraXml.append(blockXml)
                        paraXml.attrib['total_blocks'] = str(max(total_blocks))
                imagedata.append(paraXml)
                # fill_data=False
            imagexml.append(imagedata)
    return imagexml

def without_tables_function(orig_image,header_select=0):
    and_image = image_reading_converting(orig_image.copy())
    draw_image = orig_image.copy()
    table_xml = block_generator(and_image,header_pos=header_select)
    try:
        et = ET.ElementTree(table_xml)
        et.write("my_table_data.xml",pretty_print=True,xml_declaration=True,encoding="utf-8")
    except Exception as e:
        print(e)
    table_dataframe = image_reader_parser(table_xml,orig_image.copy(),header_position=header_select)
    return table_dataframe
