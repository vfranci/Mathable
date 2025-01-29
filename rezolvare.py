import numpy as np
import cv2 as cv
import os

def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.3, fy=0.3)
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
def extract_gameboard(image):
    original_image = image.copy()

    res = hsv_filter_gameboard(image)
    gray_image = cv.cvtColor(res, cv.COLOR_BGR2GRAY)

    image_m_blur = cv.medianBlur(gray_image, 3)
    image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 5)
    image_sharpened = cv.addWeighted(image_m_blur, 1.6, image_g_blur, -0.8, 0)

    thresh = cv.adaptiveThreshold(image_sharpened, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv.THRESH_BINARY, 25, 5)

    kernel = np.ones((5, 5), np.uint8)
    thresh = cv.erode(thresh, kernel, iterations=1)

    edges = cv.Canny(thresh, 200, 400)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    max_area = 0

    top_left, top_right, bottom_left, bottom_right = None, None, None, None

    for contour in contours:
        if len(contour) > 3:
            possible_top_left = None
            possible_bottom_right = None

            for point in contour.squeeze():
                if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                    possible_top_left = point

                if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + \
                        possible_bottom_right[1]:
                    possible_bottom_right = point

            diff = np.diff(contour.squeeze(), axis=1)
            possible_top_right = contour.squeeze()[np.argmin(diff)]
            possible_bottom_left = contour.squeeze()[np.argmax(diff)]

            if cv.contourArea(np.array([[possible_top_left], [possible_top_right], [possible_bottom_right],
                                        [possible_bottom_left]])) > max_area:
                max_area = cv.contourArea(np.array(
                    [[possible_top_left], [possible_top_right], [possible_bottom_right], [possible_bottom_left]]))
                top_left, top_right = possible_top_left, possible_top_right
                bottom_left, bottom_right = possible_bottom_left, possible_bottom_right

    if top_left is None or top_right is None or bottom_left is None or bottom_right is None:
        print("Colturile nu au fost detectate corect.")
        return None

    width, height = 1600, 1600

    original_copy = original_image.copy()
    cv.circle(original_copy, tuple(top_left), 20, (0, 0, 255), -1)
    cv.circle(original_copy, tuple(top_right), 20, (0, 0, 255), -1)
    cv.circle(original_copy, tuple(bottom_left), 20, (0, 0, 255), -1)
    cv.circle(original_copy, tuple(bottom_right), 20, (0, 0, 255), -1)

    puzzle = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    destination_of_puzzle = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(puzzle, destination_of_puzzle)

    result = cv.warpPerspective(original_image, M, (width, height))

    return result


def hsv_filter_gameboard(image):
    low_yellow = (16, 100, 100)
    high_yellow = (80, 255, 255)
    low_white = (0, 0, 160)
    high_white = (180, 90, 255)

    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    mask_yellow = cv.inRange(img_hsv, low_yellow, high_yellow)

    img_hsv[mask_yellow > 0] = [0, 0, 255]

    img_bgr_modified = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)

    img_hsv_modified = cv.cvtColor(img_bgr_modified, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(img_hsv_modified, low_white, high_white)
    result = cv.bitwise_and(img_bgr_modified, img_bgr_modified, mask=mask_white)
    return result


lines_horizontal = []
cell_size = 114
for i in range(0, 1620, cell_size):
    l = []
    l.append((0, i))
    l.append((1618, i))
    lines_horizontal.append(l)

lines_vertical = []
for i in range(0, 1620, cell_size):
    l = []
    l.append((i, 0))
    l.append((i, 1618))
    lines_vertical.append(l)

def hsv_filter_tiles(image):
    low_white = (0, 0, 130)
    high_white = (82, 87, 255)
    img_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    mask_white = cv.inRange(img_hsv, low_white, high_white)
    result = cv.bitwise_and(image, image, mask=mask_white)
    return result

def gameboard_ox_config(thresh, lines_horizontal, lines_vertical):
    matrix = np.empty((14, 14), dtype='str')

    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0] + 20
            y_max = lines_vertical[j + 1][1][0] - 20
            x_min = lines_horizontal[i][0][1] + 20
            x_max = lines_horizontal[i + 1][1][1] - 20

            patch = thresh[x_min:x_max, y_min:y_max].copy()

            m_patch = np.mean(patch)
            dev_patch = np.std(patch)

            if m_patch < 30 and dev_patch < 15:
                matrix[i][j] = 'x'
            else:
                black_pixel_count = np.sum(patch == 0)
                black_pixel_prop = black_pixel_count / patch.size

                if black_pixel_prop > 0.05:
                    matrix[i][j] = 'x'
                else:
                    matrix[i][j] = 'o'

    return matrix

def empty_board_config(lines_horizontal, lines_vertical):
    matrix = np.empty((14, 14), dtype='str')
    for i in range(len(lines_horizontal) - 1):  # Iterăm prin liniile orizontale
        for j in range(len(lines_vertical) - 1):  # Iterăm prin liniile verticale
            matrix[i][j] = 'o'
    return matrix

def view_board_config(result, matrix, lines_horizontal, lines_vertical):
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]

            # if matrix[i][j] == 'x':
            #     cv.rectangle(result, (y_min, x_min), (y_max, x_max), color=(255, 0, 0), thickness=5)


templates1 = {}
for j in range(0, 100):
    img_template = cv.imread(f'templates1/{j}.jpg')
    if img_template is not None:
        templates1[j] = img_template
    else:
        print(f"Template {j}.jpg not found.")


def preprocess_image(image):

    blurred_image = cv.GaussianBlur(image, (5, 5), 0)

    gray_image = cv.cvtColor(blurred_image, cv.COLOR_BGR2GRAY)

    # _, thresholded = cv.threshold(gray_image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # kernel = np.ones((5, 5), np.uint8)
    # thresh = cv.erode(thresholded, kernel, iterations=1)
    return gray_image


def scale_template(template, patch):
    height, width = patch.shape
    template_height, template_width = template.shape
    scale_x = width / template_width
    scale_y = height / template_height
    resized_template = cv.resize(template, None, fx=scale_x, fy=scale_y, interpolation=cv.INTER_LINEAR)
    return resized_template


def rotate_image(image, angle):
    height, width = image.shape
    center = (width // 2, height // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def shift_patch(patch, dx, dy):
    height, width = patch.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted_patch = cv.warpAffine(patch, M, (width, height), borderMode=cv.BORDER_CONSTANT, borderValue=0)
    return shifted_patch

def correlation_score(patch, template):
    if template.shape != patch.shape:
        raise ValueError("Dimensiunile template-ului si patch-ului trebuie sa fie identice.")

    template = template.astype(float)
    patch = patch.astype(float)

    mean_template = sum(sum(template)) / template.size
    patch_mean = sum(sum(patch)) / patch.size

    dev_template = template - mean_template
    patch_dev = patch - patch_mean

    numerator = sum(sum(dev_template * patch_dev))

    template_norm = sum(sum(dev_template ** 2))
    patch_norm = sum(sum(patch_dev ** 2))
    denominator = np.sqrt(template_norm * patch_norm)


    if denominator == 0:
        return 0.0


    correlation_score = numerator / denominator

    return correlation_score

def categorize_tile(patch, templates=templates1 , shift_range=40):
    maxi = -np.inf
    poz = -1
    best_template = None
    patch = preprocess_image(patch)


    for dx in range(-shift_range, shift_range + 1, 8):
        for dy in range(-shift_range, shift_range + 1, 8):
            shifted_patch = shift_patch(patch, dx, dy)

            for j, img_template in templates.items():

                img_template = preprocess_image(img_template)
                scaled_template = scale_template(img_template, shifted_patch)

                for angle in range(-9, 10, 3):
                    rotated_template = rotate_image(scaled_template, angle)
                    corr_map = correlation_score(shifted_patch, rotated_template)
                    corr = np.max(corr_map)
                    if corr > maxi:
                        maxi = corr
                        poz = j
                        best_template = rotated_template

    return poz

def run_project(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    boards_config_list = []
    matrix_board = empty_board_config(lines_horizontal, lines_vertical)
    boards_config_list.append(matrix_board)
    files = os.listdir(input_dir)

    for file in files:
        if file[-3:] == 'jpg':
            file_scores = os.path.join(output_dir, f"{file[0]}_scores.txt")
            file_turns = os.path.join(output_dir, f"{file[0]}_turns.txt")
            # Verificăm și creăm fișierele dacă nu există
            if not os.path.exists(file_scores):
                with open(file_scores, 'w') as f:
                    f.write("")
                print(f"Fisierul {file_scores} a fost creat.")

            if not os.path.exists(file_turns):
                with open(file_turns, 'w') as f:
                    f.write("")
                print(f"Fisierul {file_turns} a fost creat.")
            img = cv.imread(input_dir + file)
            result = extract_gameboard(img)
            result_copy = hsv_filter_tiles(result)
            _, thresh = cv.threshold(result_copy, 100, 255, cv.THRESH_BINARY_INV)
            matrix_board = gameboard_ox_config(thresh, lines_horizontal, lines_vertical)
            view_board_config(result, matrix_board, lines_horizontal, lines_vertical)

            print(matrix_board)
            boards_config_list.append(matrix_board)
            new_tile_pos = []

            file_base, _ = os.path.splitext(file)
            output_file = os.path.join(output_dir, f"{file_base}.txt")

            with open(output_file, 'w') as f:
                for k in range(len(boards_config_list) - 1, -1, -1):
                    if k == 0:
                        continue
                    matrix_board1 = boards_config_list[k - 1]
                    matrix_board2 = boards_config_list[k]
                    for i in range(len(matrix_board1)):
                        for j in range(len(matrix_board1[0])):
                            if matrix_board1[i][j] == 'o' and matrix_board2[i][j] != 'o':
                                new_tile_pos.append((i, j))

                    for (i, j) in new_tile_pos:
                        x_min = lines_horizontal[i][0][1]
                        x_max = lines_horizontal[i + 1][1][1]
                        y_min = lines_vertical[j][0][0]
                        y_max = lines_vertical[j + 1][1][0]

                        patch = result[x_min:x_max, y_min:y_max].copy()
                        categorized_tile = categorize_tile(patch)
                        #cv.rectangle(result, (y_min, x_min), (y_max, x_max), color=(0, 0, 255), thickness=5)

                        output_result = f"{i + 1}{chr(j + 65)} {categorized_tile}\n"
                        f.write(output_result)
                        print(file)
                    #show_image(file, result)
                    break

input_dir = 'evaluare/fake_test/'
output_dir = 'rezultate1/'
run_project(input_dir, output_dir)





