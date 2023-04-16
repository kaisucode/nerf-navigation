import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

class KeyframeDetector():
    def __init__(self):
        # Initiate feature detector
        self.detector = cv.SIFT_create()
        # self.detector = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_L2)
        pass

    def computeFeature(self, img_arr, output_name):
    
        # find the keypoints and descriptor
        gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)

        # draw the feature
        img = cv.drawKeypoints(gray, kp , img_arr, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.imwrite(output_name, img)


        pass

    def frameMatch(self, img_arr, prev_img_arr):
        # match the current img with the prev image
        gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        prev_gray = cv.cvtColor(prev_img_arr, cv.COLOR_BGR2GRAY)
        prev_kp, prev_des = self.detector.detectAndCompute(prev_gray, None)

        # apply the matcher
        matches = self.matcher.knnMatch(des, prev_des, k=2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                good.append([m])
        
        # cv.drawMatchesKnn expects list of lists as matches.
        match_img = cv.drawMatchesKnn(gray, kp, 
                                 prev_gray, prev_kp, 
                                 good, None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imwrite("./test.jpg", match_img)
        pass

    def detectKeyframe(self):
        pass

    def readImage(self, file_name):
        img = cv.imread(file_name)
        return img


if __name__ == "__main__":
    print("Start...")
    kfd = KeyframeDetector()
    
    # read image
    image_name = "./date/img0.jpg"
    img0_arr = kfd.readImage(image_name)

    image_name = "./date/img1.jpg"
    img1_arr = kfd.readImage(image_name)
    print("image shape: ", img0_arr.shape)

    ## feature detection
    print("Feature detection stage...")
    img_arr = img0_arr
    output_name = "./sift_keypoints" + image_name + ".jpg"
    kfd.computeFeature(img_arr, output_name)

    ## feature matching
    print("feature matching stage...")
    kfd.frameMatch(img1_arr, img0_arr)

     