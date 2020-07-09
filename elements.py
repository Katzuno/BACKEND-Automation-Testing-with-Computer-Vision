import cv2
import numpy as np
import imutils
import glob
import re
import os
import pytesseract
from pytesseract import image_to_string
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt

from google.cloud import vision
import io

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'GOOGLE_CLOUD_CREDENTIALS.json'
client = vision.ImageAnnotatorClient()
'''
	notes:
		-	for get_evet, if OCR does not work propperly, continue to preprocess image. 
				link: https://medium.freecodecamp.org/getting-started-with-tesseract-part-ii-f7f9a0899b3f
'''


def detect_text(image, mode='GCloud'):
    if mode == 'GCloud':
        image = vision.types.Image(content=cv2.imencode('.jpg', image)[1].tobytes())

        response = client.text_detection(image=image)

        if response.error.message:
            raise Exception(
                '{}\nFor more info on error messages, check: '
                'https://cloud.google.com/apis/design/errors'.format(
                    response.error.message))

        texts = response.text_annotations
        detections = ""
        for text in texts:
            detections += text.description + "\n"

        return detections
    elif mode == 'Tesseract':
        return image_to_string(image, lang='eng')


def check_element_moved(currentFrame, elements, hovered_draggable_name, draggable_coord):
    draggable_element = elements[hovered_draggable_name]
    #print(draggable_coord)
    (startX, startY, endX, endY) = find_element(currentFrame, draggable_element, 0.7)
    if [(startX, startY), (endX, endY)] != [(0, 0), (0, 0)] and \
        [(startX, startY), (endX, endY)] != draggable_coord:
        return True, [(startX, startY), (endX, endY)]

    return False, draggable_coord


def get_event(frame, elements, elements_coord, key, functions, types, input_fields_path, mode='GCloud'):
    text_size = 150

    if key in functions:
        function_name = functions[key][0]
        event = function_name + '('

        parameters = functions[key][1:]
        for param in parameters:
            if types[param] == 'TextField':  # get text in text area
                empty_field = elements[param]
                empty_field = process_image_for_OCR(empty_field, scale_factor=5, filter='bilateralFilter')
                empty_field_text = detect_text(empty_field, mode)
                # print(empty_field_text)
                # print(type(empty_field_text))
                split_empty_field_text = re.split(' |\n', empty_field_text)
                split_empty_field_text = list(dict.fromkeys(split_empty_field_text))
                # print('--- OCR APELAT, EMPTY FIELD TEXT -----')
                # print(split_empty_field_text)

                (startX, startY), (endX, endY) = elements_coord[param]
                field_image = frame[startY:endY, startX:endX]
                try:
                    field_image = process_image_for_OCR(field_image, scale_factor=4)
                    field_string = detect_text(field_image, mode)
                except:
                    field_string = '{Hi, I am a software developer!} \n'
                split_field_text = re.split(' |\n', field_string)
                split_field_text = list(dict.fromkeys(split_field_text))

                # print('----- TEXT DETECTAT: -----')
                # print(split_field_text)
                #plt.imshow(frame)
                plt.title('ORIGINAL IMAGE, detected ' + empty_field_text)
                plt.show()
                # plt.imshow(field_image)
                # plt.title('COMPLETED IMAGE, detected ' + field_string)
                # print('--- OCR TERMINAT, SE FACE DIFF -----')

                if len(split_field_text) > 1:
                    split_field_text.remove('')
                diff = list_diff(split_field_text, split_empty_field_text)
                text = ' '.join(diff)
                # processed_string = field_string.split(' ')
                # print param + ' - ' + processed_string
                event = event + text + ', '

            if types[param] == 'Tablet':  # get text after the checked radio button
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

            if types[param] == 'Draggable':
                # draw a bounding box around the draggable element of color pink
                cv2.rectangle(frame, (startX, startY), (endX, endY), (233, 51, 255), 3)
                cv2.imwrite('Detected draggable.jpg', frame)
                event = event + 'DragRight' + ', '

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


def patch_frame_for_ocr(frame, element_coords, mode='GCloud'):
    (startX, startY), (endX, endY) = element_coords
    field_image = frame[startY:endY, startX:endX]
    try:
        field_image = process_image_for_OCR(field_image, scale_factor=4)
        field_string = detect_text(field_image, mode)
    except:
        cv2.imshow(field_image)
    split_field_text = re.split(' |\n', field_string)
    split_field_text = ''.join(list(dict.fromkeys(split_field_text)))
    return split_field_text


def get_misclassified_elements_through_ocr(frame, element_1_name, element_1_coords, element_2_name, element_2_coords):
    """
    This function will diff elements by doing OCR on them
    @TODO: We can improve this by doing OCR on original template element instead of comparing with file_name
    :rtype: object
    """
    misclassified_element_name = ''

    if patch_frame_for_ocr(frame, element_1_coords).lower() not in element_1_name.lower():
        misclassified_element_name = element_1_name
    elif patch_frame_for_ocr(frame, element_2_coords).lower() not in element_2_name.lower():
        misclassified_element_name = element_2_name

    return misclassified_element_name


def find_overlapping_elements_on_page(elements_on_page_name, elements_on_page_coordinates):
    print(elements_on_page_coordinates)
    for i in range(len(elements_on_page_coordinates) - 1):
        print(elements_on_page_coordinates[i])
        coord_elem_1_top_left, coord_elem_1_bottom_right = elements_on_page_coordinates[i][0], \
                                                           elements_on_page_coordinates[i][1]
        for j in range(i + 1, len(elements_on_page_coordinates)):
            coord_elem_2_top_left, coord_elem_2_bottom_right = elements_on_page_coordinates[j][0], \
                                                               elements_on_page_coordinates[j][1]
            if do_overlap(coord_elem_1_top_left, coord_elem_1_bottom_right, coord_elem_2_top_left,
                          coord_elem_2_bottom_right):
                return elements_on_page_name[i], elements_on_page_coordinates[i], elements_on_page_name[j], \
                       elements_on_page_coordinates[j]
    return None, None, None, None


def intersect_pages_by_elements(all_pages, elements_coord, pages):
    elements_on_page_coordinates = []
    elements_on_page_name = []

    for eid in elements_coord.keys():
        if elements_coord[eid] != [(0, 0), (0, 0)]:
            elements_on_page_name.append(eid)
            elements_on_page_coordinates.append(elements_coord[eid])
            all_pages = all_pages.intersection(pages[eid])

    return elements_on_page_name, elements_on_page_coordinates, all_pages


def get_current_page(elements_coord, pages, frame, all_pages=None):
    all_pages = set()

    for page in pages.values():
        all_pages = all_pages.union(page)

    elements_on_page_name, elements_on_page_coordinates, all_pages = intersect_pages_by_elements(all_pages,
                                                                                                 elements_coord, pages)

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
    elements_on_page_name = []
    elements_on_page_coordinates = []

    print(elements.keys())

    for eid in elements.keys():
        print(eid)
        template = cv2.cvtColor(elements[eid], cv2.COLOR_BGR2GRAY)
        (startX, startY, endX, endY) = find_element(screenshot_gray, template, threshold)
        current_element_coords = [(startX, startY), (endX, endY)]
        if (startX, startY, endX, endY) != (0, 0, 0, 0):
            elements_on_page_name.append(eid)
            elements_on_page_coordinates.append(current_element_coords)
        elements_coord[eid] = current_element_coords

    elem1_name, elem1_coord, elem2_name, elem2_coord = find_overlapping_elements_on_page(elements_on_page_name,
                                                                                         elements_on_page_coordinates)
    if elem1_name == elem1_coord == elem2_name == elem2_coord == None:
        return elements_coord

    misclassified_element_name = get_misclassified_elements_through_ocr(screenshot, elem1_name, elem1_coord, elem2_name,
                                                                        elem2_coord)
    if misclassified_element_name != '':
        elements_coord[misclassified_element_name] = [(0, 0), (0, 0)]
    return elements_coord


def find_multi_appearance_element(image, element, threshold=0.5):
    img_gray = image
    template = element
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        (startX, startY) = ((pt[0]), pt[1])
        (endX, endY) = ((pt[0] + w), pt[1] + h)
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    return (startX, startY, endX, endY)

def find_element(image, element, threshold=0.9, edge_detection=False, multi_scale=False, visualize=False):
    if multi_scale:
        scales = np.linspace(0.2, 1.0, 20)[::-1]
    else:
        scales = [1]

    # print('========================== find_elem =====================')
    # print(element)
    # print('========================== end find_elem =================')
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
