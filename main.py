# Imports
import sys

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

import utils

# Reading images
main_paper = cv2.imread("OMR4Main.png")
width, height, channels = main_paper.shape  # extract image shape


if main_paper is None:  # checking if the images is valid to be read or Not.
    sys.exit("couldn't read this images")

main_paper_copy = main_paper.copy()

# --------------------------Pre-processing----------------------
# Converting the images to gray scale
main_paper_rgb = cv2.cvtColor(main_paper, cv2.COLOR_BGR2RGB)
""" the previous line actually convert the BGR image to RGB image first because that is
 what is used in matplotlib and if we don't do it, it will mix things up eventually"""


main_paper_gray = cv2.cvtColor(main_paper_rgb, cv2.COLOR_RGB2GRAY)# Here we convert our RGB to gray image.
""" So, if we plot the imges now we will see it's converted to grayscale image.
    to do that us the code below again:"""

main_paper_blur = cv2.GaussianBlur(main_paper_gray, (5, 5), 1)# Making the image blur (necessary for edge detection below).

# ---------------EDGE DETECTION AND CONTOURS------------------------------

# Edge detection
main_paper_canny = cv2.Canny(main_paper_blur, 100, 200)

# Contours
main_paper_contours, hierarchy = cv2.findContours(main_paper_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(main_paper_copy, main_paper_contours, -1, (0, 0, 255), 1)

# sorted rectangles (biggest area first)
rectangles = utils.find_sorted_rectangles(contours=main_paper_contours)

question_box = utils.detect_corners(rectangles[0])
# Mark corners of the question box in the main_paper
cv2.drawContours(main_paper, question_box, -1, (0, 0, 255), 10)

# ----------------------------- RESHAPING THE CORNERS -----------------------------
question_box = utils.rearrange(points=question_box)

p1 = np.float32(question_box)
p2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(p1, p2, solveMethod=cv2.DECOMP_LU)
questions_image = cv2.warpPerspective(main_paper, matrix, (width, height))

width = int(questions_image.shape[1] * 60 / 100)
height = int(questions_image.shape[0] * 100 / 100)
dim = (width, height)

# resize image
questions_image = cv2.resize(questions_image, dim, interpolation=cv2.INTER_AREA)
h, w, c = questions_image.shape


questions_image_copy = questions_image.copy()
# ---------------------------------- REUSE ----------------------------------------
questions_image_gray = cv2.cvtColor(questions_image_copy, cv2.COLOR_RGB2GRAY)
questions_image_blur = cv2.GaussianBlur(questions_image_gray, (5, 5), -1)
questions_image_canny = cv2.Canny(questions_image_blur, 150, 225)

# getting questions_image_threshold using RETR_EXTERNAL to prevent having inner/outer circles
ret, questions_image_thresh = cv2.threshold(questions_image_canny, 128, 255, cv2.RETR_EXTERNAL)
# cv2.imshow("question_image_thresh", questions_image_thresh)

"""This is the value that our algorithm choose
as a threshold to push value to black when exceeding it"""

# the list of contours that correspond to questions
questions_image_contours = cv2.findContours(questions_image_thresh, cv2.RETR_EXTERNAL,
                       cv2.CHAIN_APPROX_SIMPLE)
questions_image_contours = imutils.grab_contours(questions_image_contours)

questions_image_contours.sort(key=lambda x: utils.get_contour_precedence(x, main_paper.shape[1]))

circles_contours = []
# loop over the contours
for c in questions_image_contours:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 12 and h >= 12 and 0.8 <= ar <= 1.5:
        circles_contours.append(c)
cv2.drawContours(questions_image_copy, circles_contours[50], -1, (0, 0, 255), 1)
print(len(circles_contours))

""" 
    GETTING CIRCLE VALUES, Now we have our contours sorted for each circle
    We should apply threshold to count number of zero pixels in each circle
    having many zero pixels means that this circle is marked
    having few zero pixels means its white circle (not marked)
"""

# NOW we have circles_contours which has all 800 sorted contours
# apply Otsu's threshold method to binarize the warped piece of paper
bubbled_thresh = cv2.threshold(questions_image_gray, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # inverted values (search for non-zeros)
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
    is_bubbled = False
    if(total > 100):
        is_bubbled = True
    bubbled.append(is_bubbled)
    print("index: ", j, "total:", total, "is bubbled: ", is_bubbled)

# ------------------------------------------ plotting ----------------------------
plt.figure("question table")
plt.imshow(questions_image_copy, cmap="gray")  # plotting the image.
plt.show()
k = cv2.waitKey(0)  # preventing the images to automatically just disappear.
if k == ord("s"):  # save the image if we pressed the letter s in out keyboard.
    cv2.imwrite("OMR4.jpg", main_paper)