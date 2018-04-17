"""Extract SIFT descriptors from the dataset images."""

import cv2
import pdb
import os
import numpy as np


def extract_SIFT(SIFT, img):
    """Extract SIFT descriptors from an image."""
    kp, desc = SIFT.detectAndCompute(img, None)
    kp_vars = np.array([[x.pt[0], x.pt[1], x.size, x.angle] for x in kp])
    sift_desc = np.concatenate((kp_vars, desc), axis=1)
    return sift_desc


def extract_descriptors(infolder, outfolder, train=True):
    label = "_train_" if train else "_test_"
    files = os.listdir(infolder)
    SIFT = cv2.xfeatures2d.SIFT_create()
    for f in files:
        print("Processing", f)
        img = cv2.imread(infolder + f)
        desc = extract_SIFT(SIFT, img)
        write_to_file(desc, outfolder + f[0:-4] + label + 'sift.csv')


def write_to_file(desc, filename, files):
    f = open(filename, "w")
    for d in desc:
        out = ",".join([str(x) for x in d])
        f.write(out + '\n')
    f.close()


if __name__ == "__main__":
    TEST_FOLDER = "/test/data/images/"
    TRAIN_FOLDER = "/train/data/images/"

    TEST_OUT = "/output/for/test/data/features/"
    TRAIN_OUT = "/output/for/train/data/features/"

    extract_descriptors(TEST_FOLDER, TEST_OUT, train=False)
    extract_descriptors(TRAIN_FOLDER, TRAIN_OUT, train=True)
