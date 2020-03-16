#
# CSCI3290 Computational Imaging and Vision *
# --- Declaration --- *
# I declare that the assignment here submitted is original except for source
# material explicitly acknowledged. I also acknowledge that I am aware of
# University policy and regulations on honesty in academic work, and of the
# disciplinary guidelines and procedures applicable to breaches of such policy
# and regulations, as contained in the website
# http://www.cuhk.edu.hk/policy/academichonesty/ *
# Assignment 2
# Name : Lee Tsz Yan
# Student ID : 1155110177
# Email Addr : 1155110177@link.cuhk.edu.hk
#

import cv2
import numpy as np
import argparse


def extract_and_match_feature(img_1, img_2, ratio_test=0.7):
    """
    1/  extract SIFT feature from image 1 and image 2,
    2/  use a bruteforce search to find pairs of matched features:
        for each feature point in img_1, find its best matched feature point in img_2
    3/ apply ratio test to select the set of robust matched points

    :param img_1: input image 1
    :param img_2: input image 2
    :param ratio_test: ratio for the robustness test
    :return list_pairs_matched_keypoints: a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]]]
    """
    list_pairs_matched_keypoints = []

    # to be completed ....
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_1, None)
    kp2, des2 = sift.detectAndCompute(img_2, None)

    bf = cv2.BFMatcher()
    good = []
    matches = bf.knnMatch(des1, des2, k = 2)
    for m, n in matches:
        if m.distance < ratio_test * n.distance:
            good.append(m)

    # reference to opencv doc
    src = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    list_pairs_matched_keypoints = np.array([[i, j] for i, j in zip(src, dst)])

    return list_pairs_matched_keypoints


def find_homography_ransac(list_pairs_matched_keypoints,
                           threshold_ratio_inliers=0.85,
                           threshold_reprojection_error=3,
                           max_num_trial=1000):
    """
    Apply RANSAC algorithm to find a homography transformation matrix that align 2 sets of feature points,
    transform the second set of feature point to the first (e.g. warp image 2 to image 1)

    :param list_pairs_matched_keypoints: a list of pairs of matched points: [[[p1x,p1y],[p2x,p2y]],...]
    :param threshold_ratio_inliers: threshold on the ratio of inliers over the total number of samples,
                                    accept the estimated homography if ratio is higher than the threshold
    :param threshold_reprojection_error: threshold of reprojection error (measured as euclidean distance, in pixels)
                                            to determine whether a sample is inlier or outlier
    :param max_num_trial: the maximum number of trials to take sample and do testing to find the best homography matrix
    :return best_H: the best found homography matrix
    """
    best_H = None

    # to be completed ...

    src, des = zip(*list_pairs_matched_keypoints)
    src = np.float32(src).reshape(-1, 1, 2)
    des = np.float32(des).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src, des, cv2.RANSAC, ransacReprojThreshold=threshold_reprojection_error, maxIters=max_num_trial, mask=threshold_ratio_inliers)
    best_H = M
    # if(mask > threshold_ratio_inliers):
    #     best_H = M
    # matchesMask = mask.ravel().tolist()

    # h, w, d = img_1.shape
    # pts = np.float32([ [0,0],[0, h - 1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv2.perspectiveTransform(pts,M)
    
    # img_2 = cv2.polylines(img_2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
    # print(pts)
    # print(dst)
    # draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    #                singlePointColor = None,
    #                matchesMask = matchesMask, # draw only inliers
    #                flags = 2)

    # img3 = cv2.drawMatches(img_1,kp1,img_2,kp2,good,None,**draw_params)

    # cv2.imwrite("img3.png", img3)

    return best_H


def warp_blend_image(img_1, H, img_2):
    """
    1/  warp image img_2 using the homography H to align it with image img_1
        (using inverse warping and bilinear resampling)
    2/  stitch image img_2 to image img_1 and apply average blending to blend the 2 images into a single panorama image

    :param img_1:  the original first image
    :param H: estimated homography
    :param img_2:the original second image
    :return img_panorama: resulting panorama image
    """
    img_panorama = None

    # to be completed ...
    h, w, d = img_2.shape
    h *= 2
    w *= 2
    size = (w, h)
    H = np.linalg.inv(H)
    #np.resize(img_1, size)
    new_img2 = cv2.warpPerspective(img_2, H, size)

    img_panorama = np.zeros([h, w, 3])
    # refernce to Zhansijing
    for i in range(h):
        for j in range(w):
            if i <img_1.shape[0] and j < img_1.shape[1]:
                if new_img2[i, j].any():
                    img_panorama[i, j] = np.clip(0.5 * img_1[i, j] + 0.5 * new_img2[i, j], 0, 255)
                else:
                    img_panorama[i, j] = img_1[i, j]
            else:
                img_panorama[i, j] = new_img2[i, j]



    # new_img = np.zeros([h, w, 3]).astype(np.uint8)
    # new_img[0:img_1.shape[0], 0:img_1.shape[1]] = img_1
    # print(new_img.shape)
    # print(type(new_img))
    # print(type(new_img2))
    # print(type(img_1[0][0][0]))
    # print(type(new_img[0][0][0]))
    # print(type(new_img2[0][0][0]))
    # print(new_img2.shape)
    # img_panorama = cv2.addWeighted(new_img, 0.5, new_img2, 0.5, 0.0)



    # img_panorama[0:img_1.shape[0], 0:img_1.shape[1]] = img_1


    # img_1 = cv2.resize(img_1, new_img.shape[1::-1])
    # cv2.imwrite("new.png", img_1)
    # img_panorama = cv2.addWeighted(img_1, 0.5, new_img, 0.5, 0)

    # foreground, background = new_img.copy(), img_1.copy()

    # foreground_height = foreground.shape[0]
    # foreground_width = foreground.shape[1]
    # alpha =0.5

    # # do composite on the upper-left corner of background image.
    # blended_portion = cv2.addWeighted(foreground,
    #             alpha,
    #             background[:foreground_height,:foreground_width,:],
    #             1 - alpha,
    #             0,
    #             background)
    # background[:foreground_height,:foreground_width,:] = blended_portion
    # cv2.imshow('composited image', background)
    
    return img_panorama


def stitch_images(img_1, img_2):
    """
    :param img_1: input image 1 is the reference image. We will not warp this image
    :param img_2: We warp this image to align and stich it to the image 1
    :return img_panorama: the resulting stiched image
    """
    print('==================================================================================')
    print('===== stitch two images to generate one panorama image =====')
    print('==================================================================================')

    # ===== extract and match features from image 1 and image 2
    list_pairs_matched_keypoints = extract_and_match_feature(img_1=img_1, img_2=img_2, ratio_test=0.7)

    # ===== use RANSAC algorithm to find homography to warp image 2 to align it to image 1
    H = find_homography_ransac(list_pairs_matched_keypoints, threshold_ratio_inliers=0.85,
                               threshold_reprojection_error=3, max_num_trial=1000)

    # ===== warp image 2, blend it with image 1 using average blending to produce the resulting panorama image
    img_panorama = warp_blend_image(img_1=img_1, H=H, img_2=img_2)

    return img_panorama


if __name__ == "__main__":
    print('==================================================================================')
    print('CSCI3290, Spring 2020, Assignment 2: image stitching')
    print('==================================================================================')

    parser = argparse.ArgumentParser(description='Image Stitching')
    parser.add_argument('--im1', type=str, default='test_images/MelakwaLake1.png',
                        help='path of the first input image')
    parser.add_argument('--im2', type=str, default='test_images/MelakwaLake2.png',
                        help='path of the second input image')
    parser.add_argument('--output', type=str, default='MelakwaLake.png',
                        help='the path of the output image')
    args = parser.parse_args()

    # ===== read 2 input images
    img_1 = cv2.imread(args.im1)
    img_2 = cv2.imread(args.im2)

    # ===== create a panorama image
    img_panorama = stitch_images(img_1=img_1, img_2=img_2)

    # ===== save panorama image
    cv2.imwrite(filename=args.output, img=img_panorama.clip(0.0, 255.0).astype(np.uint8))
