import os
import cv2
import imutils
import numpy as np
from imutils import contours
from imutils.object_detection import non_max_suppression
import time
#############Project Files imports#####################
from function_file import *
from parser_file import *
# os.environ['OMP_THREAD_LIMIT'] = '10'
def main_function(image_dir,head_pos = 0,num_threads=4):
    top_border_threshold = 8
    left_border_threshold = 8
    image = cv2.imread(image_dir)
    passing_file = image.copy()
    ######At first we will remove all borders in the image so for that we make a new function as border_removal
    removed_bordered_image = border_removal(passing_file,top_border_threshold,left_border_threshold)
    removed_bordered_image = cv2.copyMakeBorder(removed_bordered_image,10,10,10,10,cv2.BORDER_CONSTANT,value=(255,255,255))
    Error_in_V2 = False
    table_ranges = []
    try:
        table_ranges,line_image,horizontal_image,vertical_image = table_range_cordinates(removed_bordered_image.copy())
        # cv2.imshow("line_image",line_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    except Exception as e:
        Error_in_V2 = True
        print("This is error in main function in finding table range",e)
    # print(table_ranges)
    complete_table_data = []
    if (len(table_ranges)!=0):
        for table_regions in table_ranges:
            if table_regions[0] > 15:
                crop_image = line_image[table_regions[0] - 10:table_regions[1], :]
                horizontal_line_crop_img = purify_lines(horizontal_image[table_regions[0] - 10:table_regions[1], :],"w")
                horizontal_dilated_img = cv2.dilate(horizontal_line_crop_img.copy(), np.ones((3, 3), np.uint8), iterations=2)
                vertical_line_crop_img = purify_lines(vertical_image[table_regions[0] - 10:table_regions[1], :],"h")
                vertical_dilated_img = cv2.dilate(vertical_line_crop_img.copy(), np.ones((3, 3), np.uint8), iterations=2)
                data_crop_image = removed_bordered_image[table_regions[0] - 10:table_regions[1], :]
                # horizontal_line_crop = horizontal_line[table_regions[0] - 10:table_regions[1], :]
                # vertical_line_crop = vertical_line[table_regions[0] - 10:table_regions[1], :]
            else:
                crop_image = line_image[table_regions[0]:table_regions[1], :]
                horizontal_line_crop_img = purify_lines(horizontal_image[table_regions[0]:table_regions[1], :], "w")
                horizontal_dilated_img = cv2.dilate(horizontal_line_crop_img.copy(), np.ones((3, 3), np.uint8), iterations=2)
                vertical_line_crop_img = purify_lines(vertical_image[table_regions[0]:table_regions[1], :], "h")
                vertical_dilated_img = cv2.dilate(vertical_line_crop_img.copy(), np.ones((3, 3), np.uint8), iterations=2)
                data_crop_image = removed_bordered_image[table_regions[0]:table_regions[1], :]
            is_block_image,table_start_cordinate,table_end_cordinate = image_type(crop_image,data_crop_image.copy())
            # print(table_start_cordinate,table_end_cordinate)
            # cv2.imshow("crop_image",data_crop_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # header_position = int(input("please enter header position:(remember to enter relative to zero)"))
            print("The image is table image")
            Error_in_V1 = False
            if is_block_image:
                try:
                    blocks_cordinates,rows,columns,horizontal_groups,columns_groups = blocks_codinates_detection(~horizontal_dilated_img.copy(),~vertical_dilated_img.copy(),table_start_cordinate,table_end_cordinate,header_position=head_pos)
                    # for blocks  in blocks_cordinates:
                    #     cv2.rectangle(data_crop_image,(blocks[0],blocks[1]),(blocks[2],blocks[3]),(0,0,255),1)
                    data_array = V1_parser(blocks_cordinates,rows,columns,horizontal_groups,columns_groups,data_crop_image.copy(),header_num=head_pos)
                    complete_table_data.append(data_array)
                    Error_in_V1 = False
                except Exception as e:
                    Error_in_V1 = True
                    print("Error in V1:",e)
                # print("image_have_rows_as:",rows)
                # print("image_have_columns:",columns)
                # cv2.imwrite("bounding_box_image.jpg",data_crop_image)
                # cv2.imshow("data_cropV1", data_crop_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

            elif not is_block_image or Error_in_V1:
                try:
                    print("The image seems to be a non blocks table image processing it another way")
                    start_time = time.time()
                    non_table_data = without_tables_function(data_crop_image,header_select=head_pos,thread_pool_size=num_threads)
                    complete_table_data.append(non_table_data)
                    end_time = time.time()-start_time
                    print("Total execution time is:",end_time)
                except Exception as e:
                    Error_in_V2 = True
                    print("The error in V2",e)
        return complete_table_data
    elif((len(table_ranges)==0) or Error_in_V2):
        print("image seems to be non table image")
        non_table_data = without_tables_function(image,header_select=head_pos,thread_pool_size=num_threads)
        complete_table_data.append(non_table_data)
        return complete_table_data
table_data = main_function("/home/narveeryadav/Downloads/images/image2.png",head_pos=1,num_threads=10)
print(table_data)
