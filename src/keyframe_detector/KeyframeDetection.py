import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
# import os
# from SensorStream import * 

class KeyframeDetector():
    def __init__(self):
        # Initiate feature detector
        self.detector = cv.SIFT_create(100)
        # self.detector = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_L2)
        # reference frame
        self.ref_frame = None
        # keyframe buffer
        self.keyframe_buffer = []
        self.num_keyframe = 0
        
        # keyframe detector constant
        self.match_threshold = 60
        pass

    def computeFeature(self, img_arr, output_name):
    
        # find the keypoints and descriptor
        gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)

        # draw the feature
        img = cv.drawKeypoints(gray, kp , img_arr, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv.imwrite(output_name, img)


        pass

    def frameMatch(self, img_arr, prev_img_arr, save_name):
        # match the current img with the prev image
        gray = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        prev_gray = cv.cvtColor(prev_img_arr, cv.COLOR_BGR2GRAY)
        prev_kp, prev_des = self.detector.detectAndCompute(prev_gray, None)

        # apply the matcher
        matches = self.matcher.knnMatch(des, prev_des, k=2)
        
        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        
        # cv.drawMatchesKnn expects list of lists as matches.
        draw_params = dict(matchColor = (0,255,0), 
                           singlePointColor = None
        )
        match_img = cv.drawMatches(gray, kp, 
                                 prev_gray, prev_kp, 
                                 good, None,
                                 **draw_params)
        # cv.imwrite("./test.jpg", match_img)

        # RANSAC
        query_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        train_pts = np.float32([prev_kp[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, mask = cv.findHomography(query_pts, train_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

         
        # render
        if np.sum(matchesMask) < self.match_threshold:
            draw_params = dict(matchColor = (0,255,0), 
                            singlePointColor = None,
                            matchesMask = matchesMask # draw only inliers
            )

            match_img = cv.drawMatches(gray, kp, 
                                    prev_gray, prev_kp, 
                                    good, None,
                                    **draw_params)
            cv.imwrite(save_name, match_img)


        return np.sum(matchesMask)

    def run(self, img_arr, frame_id):
        # print(frame_id)
        # if the reference frame is not set, set it.
        if self.ref_frame is None:
            # self.keyframe_buffer.append(img_arr)
            self.ref_frame = img_arr
        # keyframe detection
        save_name = "./keyframes/frame_" + str(frame_id) + ".jpg"
        # time_begin = time()
        try:
            matched_feature = self.frameMatch(img_arr, self.ref_frame, save_name)
        except:
            return False
        # print(time() - time_begin)
        detected = matched_feature < self.match_threshold 
        if detected:
            self.num_keyframe += 1
            # self.keyframe_buffer.append(img_arr)
            self.ref_frame = img_arr

        return detected

    def readImage(self, file_name):
        img = cv.imread(file_name)
        return img


# if __name__ == "__main__":
#     # setting for the sensor listener
#     sensors_listener = SensorListener()
#     kfd = KeyframeDetector()
#     # task settings
#     loop_hz = 60
#     rate = rospy.Rate(loop_hz)
    
#     # task begins
#     frame_id = 0
#     while not rospy.is_shutdown():
#         if not sensors_listener.sensor_initialzed:
#             continue
#         # ---- keyframe detection ----
#         kfd.run(sensors_listener.rgb_image, frame_id)
#         print("length of the keyframe buffer", len(kfd.keyframe_buffer))
#         frame_id += 1
#         rate.sleep()

    # print("Start...")
    # kfd = KeyframeDetector()
    
    # # read image
    # image_name = "./date/img0.jpg"
    # img0_arr = kfd.readImage(image_name)

    # image_name = "./date/img1.jpg"
    # img1_arr = kfd.readImage(image_name)
    # print("image shape: ", img0_arr.shape)

    # ## feature detection
    # print("Feature detection stage...")
    # img_arr = img0_arr
    # output_name = "./sift_keypoints" + image_name + ".jpg"
    # kfd.computeFeature(img_arr, output_name)

    # ## feature matching
    # print("feature matching stage...")
    # kfd.frameMatch(img1_arr, img0_arr)

     