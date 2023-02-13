# Imports
import sys
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

import utils


# Reading images
def get_bubbles(path):
    path = str(path)
    main_paper = cv2.imread(path)
    # width2, height2, channels = main_paper.shape
    # print("original image resolution: width of {}, height of {}".format(width2, height2))
    img = utils.image_resize(main_paper, height=1800)
    width, height, channels = img.shape
    # print("resized image resolution: width of {}, height of {}".format(width, height))

    plt.figure("question table")
    # plt.imshow(main_paper, cmap="gray")# plotting the image.
    # plt.show()
    k = cv2.waitKey(0)  # preventing the images to automatically just disappear.
    if k == ord("s"):  # save the image if we pressed the letter s in out keyboard.
        cv2.imwrite("OMR4.jpg", main_paper)
    if main_paper is None:  # checking if the images is valid to be read or Not.
        sys.exit("couldn't read this images")

    main_paper_copy = img.copy()
    main_paper_copy2 = img.copy()
    main_paper_copy3 = img.copy()

    # --------------------------Pre-processing----------------------
    # Converting the images to gray scale
    main_paper_rgb = cv2.cvtColor(main_paper_copy, cv2.COLOR_BGR2RGB)
    """ the previous line actually convert the BGR image to RGB image first because that is
     what is used in matplotlib and if we don't do it, it will mix things up eventually"""

    main_paper_gray = cv2.cvtColor(main_paper_rgb, cv2.COLOR_RGB2GRAY)  # Here we convert our RGB to gray image.
    """ So, if we plot the images now we will see it's converted to grayscale image.
        to do that us the code below again:"""

    norm_img = np.zeros((width, height))

    main_paper_gray_normalized = cv2.normalize(main_paper_gray, norm_img, 0, 255, cv2.NORM_MINMAX)

    main_paper_blur = cv2.GaussianBlur(main_paper_gray_normalized, (5, 5),
                                       1)  # Making the image blur (necessary for edge detection below).

    # ---------------EDGE DETECTION AND CONTOURS------------------------------

    # Edge detection
    main_paper_canny = cv2.Canny(main_paper_blur, 100, 200)

    # Contours
    main_paper_contours, hierarchy = cv2.findContours(main_paper_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(main_paper_copy2, main_paper_contours, -1, (255, 0, 0), 2)

    # sorted rectangles (biggest area first)
    rectangles = utils.find_sorted_rectangles(contours=main_paper_contours)

    question_box = utils.detect_corners(rectangles[0])
    cv2.drawContours(main_paper_copy3, rectangles, -1, (0, 0, 255), 2)

    # ----------------------------- RESHAPING THE CORNERS -----------------------------
    question_box = utils.rearrange(points=question_box)

    p1 = np.float32(question_box)
    p2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(p1, p2, solveMethod=cv2.DECOMP_LU)
    questions_image = cv2.warpPerspective(img, matrix, (width, height))

    width = int(questions_image.shape[1] * 60 / 100)
    height = int(questions_image.shape[0] * 100 / 100)
    dim = (width, height)

    # resize image
    questions_image = cv2.resize(questions_image, dim, interpolation=cv2.INTER_AREA)
    h, w, c = questions_image.shape

    questions_image = questions_image[8:-8, 8:-8]


    questions_image_copy = questions_image.copy()
    questions_image_copy2 = questions_image.copy()
    # ---------------------------------- applying circle contours  ----------------------------------------
    questions_image_gray = cv2.cvtColor(questions_image_copy, cv2.COLOR_RGB2GRAY)
    questions_image_blur = cv2.GaussianBlur(questions_image_gray, (5, 5), -1)

    questions_image_canny = cv2.Canny(questions_image_blur, 0, 255)

    # applying a threshold to the canny image using thresh otso.
    ret, questions_image_thresh = cv2.threshold(questions_image_canny, 128, 255, cv2.THRESH_OTSU)
    # cv2.imshow("question_image_thresh", questions_image_thresh)
    """This is the value that our algorithm choose
    as a threshold to push value to black when exceeding it"""

    # getting questions_image_contours using RETR_EXTERNAL to prevent having inner/outer circles
    questions_image_contours = cv2.findContours(questions_image_thresh.copy(), cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_NONE)

    questions_image_contours = imutils.grab_contours(questions_image_contours)
    # print(len(questions_image_contours))
    questions_image_contours = list(questions_image_contours)
    questions_image_contours.sort(key=lambda x: utils.get_contour_precedence(x, main_paper.shape[1]))
    questions_image_contours = tuple(questions_image_contours)
    circles_contours = []
    # loop over the contours
    all_h = 0
    all_w = 0
    h_list = []
    w_list = []
    for c in questions_image_contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        all_h += h
        all_w += w
        h_list.append(h)
        w_list.append(w)

    all_h = all_h / len(questions_image_contours)
    all_w = all_w / len(questions_image_contours)
    # print(all_h)
    # print(all_w)
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    plt.hist(h_list, bins=bins, edgecolor="red")
    plt.xlabel("contours intervals(numbers)")
    plt.ylabel("contours area")
    # plt.show()

    for c in questions_image_contours:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        if w >= all_w and h >= all_h and 0.2 <= ar <= 1.5:
            circles_contours.append(c)
    cv2.drawContours(questions_image, circles_contours, -1, (255, 0, 0), 2)

    """ 
        GETTING CIRCLE VALUES, Now we have our contours sorted for each circle
        We should apply threshold to count number of zero pixels in each circle
        having many zero pixels means that this circle is marked
        having few zero pixels means its white circle (not marked)
    """

    # NOW we have circles_contours which has all 800 sorted contours
    # apply Otsu's threshold method to binarize the warped piece of paper
    bubbled_thresh = cv2.threshold(questions_image_gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[
        1]  # inverted values (search for non-zeros)
    # we will loop on every contour and get its value
    bubbled = []

    for (j, c) in enumerate(circles_contours):
        # construct a mask that reveals only the current
        # "bubble" for the question
        mask = np.zeros(bubbled_thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # apply the mask to the threshold image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(bubbled_thresh, bubbled_thresh, mask=mask)
        total = cv2.countNonZero(mask)
        is_bubbled = 0
        if total > 500:
            is_bubbled = 1
        bubbled.append(is_bubbled)
        # print("index: {} total: {} bubbled?: {}".format(j, total, is_bubbled))
    return bubbled

# # ----------------------------------------------- PLOTTING --------------------------
# plt.figure("question table")
# plt.imshow(questions_image, cmap="gray")  # plotting the image.
# plt.show()
# k = cv2.waitKey(0)  # preventing the images to automatically just disappear.
# if k == ord("s"):  # save the image if we pressed the letter s in out keyboard.
#     cv2.imwrite("OMR4.jpg", main_paper)

## -------------------------------------------------- MAIN FUNCTION --------------------
if __name__ == '__main__':
    rename_images = True
    cwd = os.getcwd()
    samples = os.path.join(cwd, 'samples')
    ans_path = os.path.join(samples, 'ans.jpg')
    answers = get_bubbles(ans_path)

    for q in os.scandir(samples):
        if 'ans' in q.name:
            continue

        bubbles = get_bubbles(q.path)
        if 1 in bubbles:
            grade = 0
            for i in range(0, 608, 4):
                if bubbles[i] == answers[i] and bubbles[i + 1] == answers[i + 1] and bubbles[i + 2] == answers[
                    i + 2] and bubbles[i + 3] == answers[i + 3]:
                    grade += 1
            if rename_images:
                utils.rename_image_with_grades(q, grade)
            print(q.name, grade)
        else:
            print(q.name, 0)


