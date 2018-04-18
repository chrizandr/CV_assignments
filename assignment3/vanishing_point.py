"""Vanishing point detection."""

import pdb
import matplotlib.pyplot as plt
import os
import itertools
import random

import cv2
import numpy as np


def hough_transform(img):
    """Use hough_transform and find hough lines."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
    kernel = np.ones((10, 10), np.uint8)

    opening = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    edges = cv2.Canny(opening, 50, 150, apertureSize=3)
    plt.imshow(edges, "gray")
    plt.show()
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    hough_lines = []

    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x_0 = a * rho
        y_0 = b * rho
        x_1 = int(x_0 + 1000 * (-b))
        y_1 = int(y_0 + 1000 * (a))
        x_2 = int(x_0 - 1000 * (-b))
        y_2 = int(y_0 - 1000 * (a))
        hough_lines.append(((x_1, y_1), (x_2, y_2)))

    return hough_lines


def sample_lines(lines, size):
    """Randomly sample lines."""
    if size > len(lines):
        size = len(lines)
    return random.sample(lines, size)


def det(a, b):
    """Det of matrix."""
    return a[0] * b[1] - a[1] * b[0]


# Find intersection point of two lines (not segments!)
def line_intersection(line1, line2):
    """Find intersection between two lines using two samples points from the line."""
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div

    return x, y


def find_intersections(lines):
    """Find potential points of intersection using hough lines."""
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:  # If lines cross, then add
                    intersections.append(intersection)

    return intersections


def find_vanishing_point(img, grid_size, intersections):
    """Find vanishing point in each grid cell."""
    image_height = img.shape[0]
    image_width = img.shape[1]

    grid_rows = (image_height // grid_size) + 1
    grid_columns = (image_width // grid_size) + 1

    for i, j in itertools.product(range(grid_rows), range(grid_columns)):
        cell_left = i * grid_size
        cell_right = (i + 1) * grid_size
        cell_bottom = j * grid_size
        cell_top = (j + 1) * grid_size
        cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 10)

        current_intersections = 0
        current_x = 0
        current_y = 0
        for x, y in intersections:
            if cell_left < x < cell_right and cell_bottom < y < cell_top:
                current_intersections += 1
                current_x += x
                current_y += y

        if current_intersections > 0:
            current_x = current_x/current_intersections
            current_y = current_y/current_intersections
            cv2.circle(img, (int(current_x), int(current_y)), current_intersections // 4, (255, 0, 0), -1)

    return img


if __name__ == "__main__":
    IMG_FILE = "data/img1.jpg"
    img = cv2.imread(IMG_FILE)
    hough_lines = hough_transform(img)
    if hough_lines:
        random_sample = sample_lines(hough_lines, 100)
        intersections = find_intersections(random_sample)
        if intersections:
            grid_size = min(img.shape[0], img.shape[1]) // 8
            new_img = find_vanishing_point(img, grid_size, intersections)
            plt.imshow(new_img)
            plt.show()
            pdb.set_trace()
            filename = 'output/' + os.path.splitext(IMG_FILE)[0] + '_center' + '.jpg'
            cv2.imwrite(filename, img)


#Equation y=13/267x+455401/267
