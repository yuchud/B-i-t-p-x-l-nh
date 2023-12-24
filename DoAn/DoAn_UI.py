import PySimpleGUI as sg
import cv2
import numpy as np
from PIL import ImageColor
import os

MAX_TEMPLATES_IN_PAGE = 8

# SIZE
WINDOW_SIZE = (820, 800)
BTN_SIZE = (11, 1)
BTN_MOVE_SIZE = (2, 2)
BTN_TEMPLATE_SIZE = (9, 3)
IMAGE_SIZE = (224, 224)
TEMPLATE_IMAGE_SIZE = (78, 54)

# COLOR
TEXT_COLOR = 'black'
CAPTURING_BTN_COLOR = 'red'
DEFAULT_BTN_COLOR = sg.theme_button_color_background()
IMAGE_BG_COLOR = 'grey'

# DEFAULT IMAGE
DEFAULT_IMG = np.zeros((256, 256, 3), np.uint8)
DEFAULT_IMG[:] = ImageColor.getrgb(IMAGE_BG_COLOR)


import tensorflow as tf
IMAGE_SIZE = (224,224)
def decode_and_resize(path):
  image = tf.io.read_file(path)
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.convert_image_dtype(image, dtype='float32')
  image = tf.image.resize(image, IMAGE_SIZE)

  return image

def get_all_styles():
    images = []
    folder = f"{os.getcwd()}\\Styles_Template"
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is not None:
            img = cv2.resize(img, IMAGE_SIZE)
            images.append(img)
    return images


def fill_image():
    window['-IMAGE-'].update(data=cv2.imencode('.png', image)[1].tobytes())


def fill_style():
    window['-STYLE-'].update(data=cv2.imencode('.png', style)[1].tobytes())


def fill_result():
    window['-RESULT-'].update(data=cv2.imencode('.png', result)[1].tobytes())


def clear_result():
    global result
    result = None
    current_image = cv2.resize(DEFAULT_IMG, IMAGE_SIZE)
    window['-RESULT-'].update(data=cv2.imencode('.png', current_image)[1].tobytes())


def get_template_index(current_choose_index):
    return (current_page - 1) * MAX_TEMPLATES_IN_PAGE + current_choose_index


def fill_template():
    for current_choose_index in range(MAX_TEMPLATES_IN_PAGE):
        current_template_idx = get_template_index(current_choose_index)
        if current_template_idx < len(templates_list):
            current_image = cv2.resize(templates_list[current_template_idx], TEMPLATE_IMAGE_SIZE)
            window[f'-BTN TEMPLATE {current_choose_index}-'].update(
                image_data=cv2.imencode('.png', current_image)[1].tobytes(), disabled=False)
        else:
            current_image = cv2.resize(DEFAULT_IMG, TEMPLATE_IMAGE_SIZE)
            window[f'-BTN TEMPLATE {current_choose_index}-'].update(
                image_data=cv2.imencode('.png', current_image)[1].tobytes(), disabled=True)


# GLOBAL VARIABLE
image = None
style = None
result = None
recording = False

templates_list = get_all_styles()

current_page = 1
max_page = len(templates_list) // 10 + 1



# Khởi tạo các gui
image_description = sg.Text(text='Image'
                            , text_color=TEXT_COLOR)
image_gui = sg.Image(size=IMAGE_SIZE
                     , background_color=IMAGE_BG_COLOR
                     , key='-IMAGE-')
btn_choose_image_gui = sg.Button(size=BTN_SIZE
                                 , button_text='Choose Image'
                                 , key='-BTN CHOOSE IMAGE-')

style_description = sg.Text(text='Style'
                            , text_color=TEXT_COLOR)
style_gui = sg.Image(size=IMAGE_SIZE
                     , background_color=IMAGE_BG_COLOR
                     , key='-STYLE-')
btn_choose_style_gui = sg.Button(size=BTN_SIZE
                                 , button_text='Choose Style'
                                 , key='-BTN CHOOSE STYLE-')

result_description = sg.Text(text='Result'
                             , text_color=TEXT_COLOR)
result_gui = sg.Image(size=IMAGE_SIZE
                      , background_color=IMAGE_BG_COLOR
                      , key='-RESULT-')
btn_combine_gui = sg.Button(size=BTN_SIZE
                            , button_text='Combine'
                            , key='-BTN COMBINE-')
btn_save_gui = sg.Button(size=BTN_SIZE
                         , button_text='Save'
                         , key='-BTN SAVE-')
btn_capture_gui = sg.Button(size=BTN_SIZE
                            , button_text='Open Camera'
                            , key='-BTN Open Camera-')

template_description = sg.Text(text='Choose Template'
                               , text_color=TEXT_COLOR)
templates_framework = [sg.Button(size=BTN_TEMPLATE_SIZE
                                 , button_color=IMAGE_BG_COLOR
                                 , key=f'-BTN TEMPLATE {idx}-') for idx in range(MAX_TEMPLATES_IN_PAGE)]
btn_move_left = sg.Button(size=BTN_MOVE_SIZE
                          , button_text='<'
                          , key='-BTN MOVE LEFT-')
btn_move_right = sg.Button(size=BTN_MOVE_SIZE
                           , button_text='>'
                           , key='-BTN MOVE RIGHT-')
pages_description = sg.Text(text=f'Trang 1 / {max_page}'
                            , text_color=TEXT_COLOR
                            , key='-PAGES DESCRIPTION-')

# Nơi chọn Image và Style
control_column = sg.Column([
    [image_description]
    , [image_gui]
    , [btn_choose_image_gui]
    , [sg.HSeparator()]
    , [style_description]
    , [style_gui]
    , [btn_choose_style_gui]
])

# Nơi hiện kết quả
result_column = sg.Column([
    [result_description]
    , [result_gui]
    , [btn_combine_gui, btn_save_gui]
    , [btn_capture_gui]
], justification="center")

# Nơi chứa những khung ảnh
main_layout = [
    [control_column, sg.VSeparator(), result_column]
]

# Nơi chọn những Style Mẫu
template_layout = [
    [template_description]
    , [btn_move_left, *templates_framework, btn_move_right]
    , [pages_description]
]

full_layout = [
    [main_layout]
    , [sg.HSeparator()]
    , [template_layout]
]

window = sg.Window("Test", full_layout, size=WINDOW_SIZE, finalize=True)

# Khởi tạo dữ liệu trước khi mở window
window['-BTN MOVE LEFT-'].update(disabled=True)
window['-BTN SAVE-'].update(disabled=True)
if max_page == 1:
    window['-BTN MOVE RIGHT-'].update(disabled=True)

fill_template()

# Mở window
while True:
    event, values = window.read(timeout=20)
    if event == sg.WIN_CLOSED:  # There is no key call Exit so event == "Exit" is unimportant
        break

    # Chọn ảnh từ browser
    if event == '-BTN CHOOSE IMAGE-':
        file_path = sg.popup_get_file(message='', no_window=True, file_types=(('Image Files', '*.jpg; *.png'),))
        if file_path:
            try:
                image = decode_and_resize(file_path)
                print(type(image))
                fill_image()
            except Exception as e:
                sg.popup_error(e)
                print(e)

    # Chọn style từ browser
    if event == '-BTN CHOOSE STYLE-':
        file_path = sg.popup_get_file(message='', no_window=True, file_types=(('Image Files', '*.jpg; *.png'),))
        if file_path:
            try:
                style = cv2.imread(file_path)
                style = cv2.resize(style, IMAGE_SIZE)
                fill_style()
            except Exception as e:
                sg.popup_error(e)
                print(e)

    # Thực hiện combine
    if event == '-BTN COMBINE-':
        try:
            result = image
            if image is None:
                sg.popup_ok("Image is empty!", no_titlebar=True, background_color='red', text_color='white')
                continue

            if style is None:
                sg.popup_ok("Style is empty!", no_titlebar=True, background_color='red', text_color='white')
                continue

            if image is not None:
                fill_result()
        except Exception as e:
            sg.popup_error(e)
            print(e)

    # Lưu ảnh sau xử lý vào máy
    if event == '-BTN SAVE-':
        try:
            file_name = sg.popup_get_file(message='', no_window=True, save_as=True, file_types=(("PNG File", '*.png'),))
            current_capture_img = result
            if file_name:
                if cv2.imwrite(file_name, current_capture_img):
                    sg.popup_ok('Image saved successfully')

        except Exception as e:
            sg.popup_error(e)
            print(e)

    # Xử lý khi nhấn nút Open Camera
    if event == '-BTN Open Camera-':
        # Xử lý khi đang bật cam
        if recording:
            window['-BTN COMBINE-'].update(disabled=False)
            window['-BTN Open Camera-'].update(button_color=DEFAULT_BTN_COLOR)
            video.release()
        # Xử lý khi đang tắt cam
        else:
            window['-BTN COMBINE-'].update(disabled=True)
            window['-BTN Open Camera-'].update(button_color=CAPTURING_BTN_COLOR)
            video = cv2.VideoCapture(0)
        clear_result()
        recording = not recording

    # Xử lý khi chọn style
    for current_index in range(MAX_TEMPLATES_IN_PAGE):
        if event == f'-BTN TEMPLATE {current_index}-':
            style = cv2.resize(templates_list[get_template_index(current_index)], IMAGE_SIZE)
            fill_style()

    # Xử lý khi nhấn nút move right
    if event == '-BTN MOVE RIGHT-':
        current_page += 1
        window['-BTN MOVE LEFT-'].update(disabled=False)
        window['-PAGES DESCRIPTION-'].update(f'Trang {current_page} / {max_page}')
        if current_page == max_page:
            window['-BTN MOVE RIGHT-'].update(disabled=True)
        fill_template()

    # Xử lý khi nhấn nút move left
    if event == '-BTN MOVE LEFT-':
        current_page -= 1
        window['-BTN MOVE RIGHT-'].update(disabled=False)
        window['-PAGES DESCRIPTION-'].update(f'Trang {current_page} / {max_page}')
        if current_page == 1:
            window['-BTN MOVE LEFT-'].update(disabled=True)
        fill_template()

    # Xử lý việc kích hoạt các nút khi thỏa điều kiện
    window['-BTN SAVE-'].update(disabled=(result is None))

    if recording:
        _, result = video.read()
        result = cv2.resize(result, IMAGE_SIZE)
        fill_result()
