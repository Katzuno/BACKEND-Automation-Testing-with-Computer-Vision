import cv2
import imutils
import sys
import os

import random

import elements as el
import numpy as np

if len(sys.argv) > 1:
    assets_path = sys.argv[1]
else:
    assets_path = "C:\\Users\\erikh\\Desktop\\WORK\\Personal\\Backend-AutomaticTestingCV\\scenes\\scene_6_user_2\\Assets"

print(assets_path)

output_file_name = os.path.join(assets_path, 'output_file.txt')
output_file = open(output_file_name, 'w')

input_fields_path = os.path.join(assets_path, 'input_fileds')
cursor_path = os.path.join(assets_path, 'cursor.png')
elements_path = os.path.join(assets_path, 'elements')
input_path = os.path.join(assets_path, 'app_rec.mov')
elements_img_type = 'png'
pages_path = os.path.join(assets_path, 'pages.txt')
functions_path = os.path.join(assets_path, 'functions.txt')

color_green = (51, 255, 153)
color_red = (51, 51, 255)
color_yellow = (51, 225, 255)

visualize = True
tm_threshold_cursor = 0.5
if 'scene_1' in assets_path:
    tm_threshold_elements = 0.7
elif 'scene_6' in assets_path:
    tm_threshold_elements = 0.6
else:
    tm_threshold_elements = 0.85

threshold_page = 150000
intensity_threshold = 4.0
current_page = None
click = False
animation_in_progress = False
animation_start = False
cursor_on_element = False
keyframes = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
event_history = []  # event_history list, first elements means nothing

# Elements load
elements = el.load_elements(elements_path, elements_img_type)
cursor = cv2.imread(cursor_path)
cv2.imwrite("cursor.jpg", cursor)

cursor = cv2.cvtColor(cursor, cv2.COLOR_BGR2GRAY)

# Element types
types = el.get_elements_type(elements)

# Pages load
pages = el.load_pages(pages_path)

# Function load
functions = el.load_functions(functions_path)

# print(input_path)
# test file
cap = cv2.VideoCapture(input_path)
if cap.isOpened() == False:
    print("Error opening video stream or file!")

# get first frame for initial element search
_, first_frame = cap.read()
elements_coord = el.get_elements_coordinates(elements, first_frame, tm_threshold_elements)
elements_color_diff = el.get_elements_color_diff(elements, elements_coord, first_frame)

view_frame = first_frame.copy()
for elem in elements_coord:
    (startX, startY), (endX, endY) = elements_coord[elem]
    cv2.rectangle(view_frame, (startX, startY), (endX, endY), color_green, 3)

cv2.imwrite('DEBUG_IMAGE.jpg', view_frame)
# get current page
# print()
# print()
# print('se apeleaza ====================================================')

# print('-----------------------------')
# print(types)
current_page = el.get_current_page(elements_coord, pages, view_frame)
# current_page = 'RegisterPage'
# print(current_page)
event_history.append('Starting Page - ' + current_page)

# process video frame by frame
old_frame = first_frame
framesHistory = [old_frame.copy()]
index = 0

started_moving = False
new_draggable_coords = None
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # check for new page
    if index % 2 == 0:
        new_page = False
        keyframes = keyframes[:-1]
        keyframe = el.check_keyframe(frame, old_frame, threshold_page)
        keyframes = np.append(keyframe, keyframes)
        # print(keyframes)
        if keyframe:
            if animation_in_progress is False:
                animation_start = True
            animation_in_progress = True
        elif animation_in_progress and np.sum(keyframes[:3]) == 0:
            new_page = True
            animation_in_progress = False
            elements_coord = el.get_elements_coordinates(elements, frame, tm_threshold_elements)
            elements_color_diff = el.get_elements_color_diff(elements, elements_coord, frame)

        if new_page:
            new_current_page = el.get_current_page(elements_coord, pages, frame)
            if new_current_page is not None:
                current_page = new_current_page
                # print('---------------------------')
                # print(current_page)
                # cv2.imwrite('DEBUG_IMAGE_draggabke.jpg', frame)
                # for eid in elements.keys():
                #    print(eid, elements_coord[eid])
                # print('=======================', end="\n\n")
        old_frame = frame
        framesHistory.append(old_frame.copy())

        # find cursor
        image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (startX, startY, endX, endY) = el.find_element(image_gray, cursor, tm_threshold_cursor)

        # draw a bounding box around the cursor
        view_frame = frame.copy()
        cv2.rectangle(view_frame, (startX, startY), (endX, endY), color_red, 3)

        (cursor_startX, cursor_startY, cursor_endX, cursor_endY) = (startX, startY, endX, endY)

        cursor_on_element = False
        cursor_on_draggable = False
        cursor_on_tablet = False
        element_moved = False
        draggable_element_coord = None
        hovered_draggable_name = None
        hovered_tablet_name = None
        tablet_element_coord = None
        tablet_exists = False

        for eid in elements.keys():
            color = color_green
            if elements_coord[eid] != [(0, 0), (0, 0)]:
                if types[eid] == 'Tablet':
                    cursor_on_tablet = True
                    tablet_element_coord = elements_coord[eid]
                    tablet_exists = True
                    hovered_tablet_name = eid
                    if tablet_exists:
                        print('--------------------- TABLET EXISTS -----------------')
                        action = 'pressTablet(' + str(el.calculate_error_margin(cursor_startX)) + ', ' + str(
                            el.calculate_error_margin(cursor_startY)) + ')'
                        event = current_page + ' ' + action
                        print(event)
                        event_history.append(event)
                    # el.find_multi_appearance_element(view_frame, elements[eid])

            if elements_coord[eid] != [(0, 0), (0, 0)] and el.do_overlap(elements_coord[eid][0], elements_coord[eid][1],
                                                                         (startX, startY), (endX, endY)):
                cursor_on_element = True
                print('TYPE: ', types[eid])
                if types[eid] == 'Tablet' or current_page == 'BoardPage':
                    cursor_on_tablet = True
                    tablet_element_coord = elements_coord[eid]
                    tablet_exists = True
                    hovered_tablet_name = eid
                    if tablet_exists:
                        print('--------------------- TABLET EXISTS -----------------')
                        action = 'pressTablet(' + str(el.calculate_error_margin(cursor_startX)) + ', ' + str(
                            el.calculate_error_margin(cursor_startY)) + ')'
                        event = current_page + ' ' + action
                        print(event)
                        event_history.append(event)
                    # el.find_multi_appearance_element(view_frame, elements[eid])

                if types[eid] == 'Draggable':
                    cursor_on_draggable = True
                    draggable_element_coord = elements_coord[eid]
                    hovered_draggable_name = eid

                color = color_yellow
                el_image = view_frame[elements_coord[eid][0][1]:elements_coord[eid][1][1],
                           elements_coord[eid][0][0]:elements_coord[eid][1][0]]
                avg1 = cv2.mean(elements[eid])[0:3]
                avg2 = cv2.mean(el_image)[0:3]
                intensity_diff = abs(elements_color_diff[eid] - el.color_diff(avg1, avg2))

                if intensity_diff > intensity_threshold:
                    click = True
                if click:
                    # print('---------- Clicked --------------', animation_start, animation_in_progress, intensity_diff, intensity_threshold)
                    # print('----------Click--------', cursor_on_draggable, draggable_element_coord)

                    if cursor_on_draggable:
                        element_moved, new_draggable_coords = el.check_element_moved(old_frame, elements,
                                                                                     hovered_draggable_name,
                                                                                     draggable_element_coord)
                        if element_moved:
                            started_moving = True
                            # print(element_moved)
                            # print(new_draggable_coords[0], new_draggable_coords[1])
                            # color draggable pink every frame
                            cv2.rectangle(view_frame, new_draggable_coords[0], new_draggable_coords[1], (233, 51, 255),
                                          3)

                    if tablet_exists:
                        print('--------------------- TABLET EXISTS -----------------')
                        started_moving = False
                        action = 'pressTablet(' + str(cursor_startX) + ', ' + str(cursor_endX) + ')'
                        event = current_page + ' ' + action
                        print(event)
                        event_history.append(event)

                    # print('=====ELEMENT MOVED: ', element_moved, started_moving)
                    if element_moved is False and started_moving is True:
                        started_moving = False
                        elements_coord[hovered_draggable_name] = new_draggable_coords
                        action = 'dragTo(' + str(new_draggable_coords[0]) + ', ' + str(new_draggable_coords[1])
                        event = current_page + ' ' + action
                        event_history.append(event)

                    if animation_start or (intensity_diff < intensity_threshold and animation_in_progress is False):
                        # print('======= Clicked 2 ============')
                        # Element str(eid) pressed!
                        key = eid, current_page
                        action = el.get_event(framesHistory[-2], elements, elements_coord, key, functions, types,
                                              input_fields_path)
                        event = current_page + ' ' + str(action)
                        if action is not None and event != event_history[-1]:
                            event_history.append(event)
                            print(event)

                        animation_start = False
                        click = False

            cv2.rectangle(view_frame, elements_coord[eid][0], elements_coord[eid][1], color, 3)
        if cursor_on_element is False:
            click = False

        if visualize:
            frame = view_frame
    resized = imutils.resize(frame, width=int(image_gray.shape[1] * 0.6))
    cv2.imshow("Video", resized)

    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for event in event_history:
    output_file.write(event + '\n')

output_file.close()
cap.release()
cv2.destroyAllWindows()
