import argparse
import os
from transform import four_point_transform
from watershed import apply_watershed, classic_contours
import cv2
import json
import numpy as np


def get_image_file_one_by_one(input_dir):
    valid_images = [".jpg", "", ".png", ".jpeg"]
    files = os.listdir(input_dir)
    for file in files:
        ext = os.path.splitext(file)[1]
        if ext.lower() in valid_images:
            yield file


def add_padding(a, b, c, d):
    padding = 15
    a += padding
    b += padding
    c += padding
    d += padding
    return a, b, c, d


def substract_padding(a, b, c, d):
    padding = 15
    a -= padding
    b -= padding
    c -= padding
    d -= padding
    return a, b, c, d


def main(input_dir, output_dir):
    if (not os.path.exists(input_dir)):
        print("Input Directory Doesn't Exist")
        return
    if (not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    for image_file in get_image_file_one_by_one(input_dir):
        print(f'[INFO] Reading the image {image_file}')
        json_file_name = f'{image_file}.info.json'
        if not os.path.exists(os.path.sep.join([input_dir, json_file_name])):
            print(f'Corresponding json file did not found for {image_file}')
            continue
        else:
            json_file_name = os.path.sep.join([input_dir, json_file_name])

        image = cv2.imread(os.path.sep.join([input_dir, image_file]))
        x_axis_padding = 15
        y_axis_padding = 15
        with open(json_file_name) as f:
            json_data = json.load(f)
            data = json_data['canonical_board']['tl_tr_br_bl']
            data[0][0], data[0][1], data[1][1], data[3][0] = substract_padding(
                data[0][0], data[0][1], data[1][1], data[3][0])
            data[1][0], data[2][0], data[2][1], data[3][1] = add_padding(
                data[1][0], data[2][0], data[2][1], data[3][1])
            four_rect_pts = np.array(
                data, dtype="float32")

        print('[INFO] Perspective transformation...')
        transformed_image = four_point_transform(
            image, four_rect_pts, json_data['canonical_board']['board_width_to_board_height'])

        output_file = f'{image_file}.visual_feedback.jpg'
        image_output_path = os.path.sep.join([output_dir, output_file])

        print('[INFO]Applying watershed to detect the checkers...')
        checker_detected_image = apply_watershed(transformed_image)

        print('[INFO] Writing the image to the output directory')
        cv2.imwrite(image_output_path, checker_detected_image)
        json_data = {"top": [0]*12, "bottom": [0]*12}
        json_out_file = os.path.sep.join(
            [output_dir, f'{image_file}.checkers.json'])
        with open(json_out_file, 'w') as fout:
            json.dump(json_data, fout, indent=4)
        print()


if __name__ == "__main__":
    print('[INFO] Starting the scripts')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_dir')
    parser.add_argument('-o', '--output_dir')
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
    print('[INFO] End of script')
