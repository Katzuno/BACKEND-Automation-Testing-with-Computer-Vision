import cv2
import numpy as np
import imutils
import glob
import re
import os
import pytesseract
from pytesseract import image_to_string
from PIL import Image, ImageEnhance, ImageFilter

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

'''
	notes:
		-	for get_evet, if OCR does not work propperly, continue to preprocess image. 
				link: https://medium.freecodecamp.org/getting-started-with-tesseract-part-ii-f7f9a0899b3f
'''


def get_event(frame, elements, elements_coord, key, functions, types, input_fields_path):
    text_size = 150

    if key in functions:
        function_name = functions[key][0]
        event = function_name + '('

        parameters = functions[key][1:]
        for param in parameters:
            if types[param] == 'TextField':  # get text in text area
                empty_field = elements[param]
                empty_field = process_image_for_OCR(empty_field, scale_factor=5, filter='bilateralFilter')
                empty_field_text = image_to_string(empty_field, lang='eng')
                split_empty_field_text = re.split(' |\n', empty_field_text)
                print('---')
                print(split_empty_field_text)

                (startX, startY), (endX, endY) = elements_coord[param]
                field_image = frame[startY:endY, startX:endX]
                field_image = process_image_for_OCR(field_image, scale_factor=4)
                field_string = image_to_string(field_image, lang='eng')
                split_field_text = re.split(' |\n', field_string)
                print(split_field_text)
                print('---')

                diff = list_diff(split_field_text, split_empty_field_text)
                text = ' '.join(diff)
                # processed_string = field_string.split(' ')
                # print param + ' - ' + processed_string
                event = event + text + ', '

            if types[param] == 'RadioButton':  # get text after the checked radio button
                (startX, startY), (endX, endY) = elements_coord[param]
                field_image = frame[startY:endY, startX:endX]

                radio_on = cv2.imread(os.path.join(input_fields_path, 'RadioButton_on.png'))
                print('========================== get_event =====================')
                print(radio_on)
                print('========================== end get_event =================')

                (startX, startY, endX, endY) = find_element(field_image, radio_on)

                textbox_startX, textbox_startY = endX, startY
                textbox_endX, textbox_endY = endX + text_size, endY
                text_image = field_image[textbox_startY:textbox_endY, textbox_startX:textbox_endX]
                # cv2.imwrite('image_of_text.png',text_image)
                text_image = process_image_for_OCR(text_image, scale_factor=2)
                text_string = image_to_string(text_image, lang='eng')
                # cv2.imshow('img',text_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                event = event + text_string + ', '

            if types[param] == 'CheckBox':
                options = []

                (startX, startY), (endX, endY) = elements_coord[param]
                field_image = frame[startY:endY, startX:endX]

                checkbox_on = cv2.imread(os.path.join(input_fields_path, 'CheckBox_on.png'))
                (startX, startY, endX, endY) = find_element(field_image, checkbox_on)

                while (startX, startY, endX, endY) != (0, 0, 0, 0):
                    textbox_startX, textbox_startY = endX, startY
                    textbox_endX, textbox_endY = endX + text_size, endY
                    text_image = field_image[textbox_startY:textbox_endY, textbox_startX:textbox_endX]
                    # cv2.rectangle(field_image, (textbox_startX, textbox_startY), (textbox_endX, textbox_endY), (51, 255, 153), 3)
                    # cv2.imshow('img',field_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    image = process_image_for_OCR(text_image, scale_factor=3)
                    text_string = image_to_string(image, lang='eng')
                    options.append(text_string)
                    field_image[startY:endY, startX:endX] = (0, 0, 0)
                    (startX, startY, endX, endY) = find_element(field_image, checkbox_on)

                param_value = ''
                for option in options:
                    param_value = param_value + ' ' + option

                event = event + param_value + ', '

            if types[param] == 'PopUp':
                (startX, startY), (endX, endY) = elements_coord[param]
                field_image = frame[startY:endY, startX:endX]

                popup_icon = cv2.imread(os.path.join(input_fields_path, 'PopUp_on.png'))
                (startX, startY, endX, endY) = find_element(field_image, popup_icon)

                textbox_startX, textbox_startY = startX - text_size, startY
                textbox_endX, textbox_endY = startX, endY
                text_image = field_image[textbox_startY:textbox_endY, textbox_startX:textbox_endX]
                text_image = process_image_for_OCR(text_image, scale_factor=2)
                text_string = image_to_string(text_image, lang='eng')

                event = event + text_string + ', '

        if len(parameters) > 0:
            event = event[:-2]
        event = event + ')'
        return event

    else:
        return None


def process_image_for_OCR(image, scale_factor, filter='GaussianBlur'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    if filter == 'bilateralFilter':
        image = cv2.bilateralFilter(image, 9, 75, 75)
    if filter == 'GaussianBlur':
        image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # pil_img = Image.fromarray(image)
    return image


def list_diff(l1, l2):
    for i in l2:
        if i in l1:
            l1.remove(i)
    return l1


# def process_image_for_ocr(image,factor):
#     # TODO : Implement using opencv
#     image = set_image_dpi(image,factor)
#     image = remove_noise_and_smooth(image)
#     return image


# def set_image_dpi(image, factor):
#     length_x, width_y, _ = image.shape 
#     size = factor * length_x, factor * width_y
#     # size = (1800, 1800)
#     im_resized = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
#     return im_resized


# def image_smoothening(image):
#     ret1, th1 = cv2.threshold(image, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
#     ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     blur = cv2.GaussianBlur(th2, (1, 1), 0)
#     ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     return th3


# def remove_noise_and_smooth(image):
#     filtered = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
#     kernel = np.ones((1, 1), np.uint8)
#     opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
#     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
#     image = image_smoothening(image)
#     or_image = cv2.bitwise_or(image, closing)
#     return or_image

def get_elements_type(elements):
    types = dict()

    for eid in elements.keys():
        types[eid] = eid[eid.find('_') + 1:]

    return types


def check_keyframe(frame, old_frame, threshold):
    diff = cv2.absdiff(frame, old_frame)
    non_zeros = np.count_nonzero(diff > 3)
    if non_zeros > threshold:
        return 1
    else:
        return 0


def load_pages(file_name):
    pages = dict()
    # added .lower()
    with open(file_name, "r") as input_file:
        for line in input_file:
            string_list = [s.replace('\n', '') for s in line.split(' ')]
            pages[string_list[0]] = string_list[1:]

    return pages


def get_current_page(elements_coord, pages):
    all_pages = set()

    for page in pages.values():
        all_pages = all_pages.union(page)

    for eid in elements_coord.keys():
        if elements_coord[eid] != [(0, 0), (0, 0)]:
            all_pages = all_pages.intersection(pages[eid])

    if len(all_pages) == 1:
        current_page = all_pages.pop()
        return current_page
    else:
        print("There are more than 1 page with the same elements")


def load_functions(file_name):
    functions = dict()

    # added lower
    with open(file_name, "r") as input_file:
        for line in input_file:
            string_list = [s for s in re.split(' |, |,|\(|\)', line) if s != '' and s != '\n']
            functions[string_list[0], string_list[1]] = string_list[2:]

    print(functions)
    return functions


def load_elements(path, type):
    elements = {}
    images_path = os.path.join(path, '*' + type)
    # added .lower()
    for file in glob.glob(images_path):  # ex: 'Assets/elements/*png'
        id = file[file.rfind('\\') + 1: file.find('.' + type)]  # for windows change find \ and for linux find /
        elements[id] = cv2.imread(file)

    return elements


def get_elements_color_diff(elements, elements_coord, screenshot):
    elements_color_diff = dict()

    for eid in elements.keys():
        if elements_coord[eid] != None:
            coord_image = screenshot[elements_coord[eid][0][1]:elements_coord[eid][1][1],
                          elements_coord[eid][0][0]:elements_coord[eid][1][0]]
            avg1 = cv2.mean(elements[eid])[0:3]
            avg2 = cv2.mean(coord_image)[0:3]
            elements_color_diff[eid] = color_diff(avg1, avg2)

    return elements_color_diff


def get_elements_coordinates(elements, screenshot, threshold):
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    elements_coord = dict()

    for eid in elements.keys():
        template = cv2.cvtColor(elements[eid], cv2.COLOR_BGR2GRAY)
        (startX, startY, endX, endY) = find_element(screenshot_gray, template, threshold)
        elements_coord[eid] = [(startX, startY), (endX, endY)]

    return elements_coord


def find_element(image, element, threshold=0.9, edge_detection=False, multi_scale=False, visualize=False):
    if multi_scale:
        scales = np.linspace(0.2, 1.0, 20)[::-1]
    else:
        scales = [1]

    print('========================== find_elem =====================')
    print(element)
    print('========================== end find_elem =================')
    (tH, tW) = element.shape[:2]
    found = None

    for scale in scales:

        image_resized = imutils.resize(image, width=int(image.shape[1] * scale))
        r = image.shape[1] / float(image_resized.shape[1])

        if image_resized.shape[0] < tH or image_resized.shape[1] < tW:
            break

        if edge_detection:
            image_resized = cv2.Canny(image_resized, 50, 200)
            element = cv2.Canny(element, 50, 200)

        result = cv2.matchTemplate(image_resized, element, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        if visualize:
            clone = np.dstack([result, result, result])
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                          (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 2)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    (maxVal, maxLoc, r) = found
    if maxVal > threshold:
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
    else:
        (startX, startY, endX, endY) = (0, 0, 0, 0)

    return (startX, startY, endX, endY)


def do_overlap(startX1_startY1, endX1_endY1, startX2_startY2, endX2_endY2):
    startX1, startY1 = startX1_startY1
    endX1, endY1 = endX1_endY1
    startX2, startY2 = startX2_startY2
    endX2, endY2 = endX2_endY2

    if startX1 > endX2 or startX2 > endX1 or startY1 > endY2 or startY2 > endY1:
        return False

    return True


def color_diff(r1_g1_b1, r2_g2_b2):
    r1, g1, b1 = r1_g1_b1
    r2, g2, b2 = r2_g2_b2
    return np.sqrt((r2 - r1) ** 2 + (g2 - g1) ** 2 + (b2 - b1) ** 2)
