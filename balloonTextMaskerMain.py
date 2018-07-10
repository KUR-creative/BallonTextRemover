#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      JeongHyeon
#
# Created:     28-06-2018
# Copyright:   (c) JeongHyeon 2018
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import sys
import os
import cv2
import numpy as np
import bubbleFinder  # github.com/sKHJ/speechBubbleFinder
import ballTextMasker
from pathlib import Path
import utils
from tqdm import tqdm

def make_and_save(origin_dir, cleaned_dir, mask_dir,
                  origin_path, textFinder, save_mask=False):
    old_parent = Path(origin_dir).parts[-1]
    cleaned_path = utils.make_dstpath(origin_path, old_parent, cleaned_dir) 
    cleaned_path = os.path.splitext(cleaned_path)[0] + '.png'
    ## mask_path = utils.make_dstpath(origin_path, old_parent, mask_dir) 
    ## mask_path = os.path.splitext(mask_path)[0] + '.png'

    if os.path.exists(cleaned_path): 
        return

    img = cv2.imread(origin_path)
    if img is None: 
        return
    #print(origin_path,'|',cleaned_path,'|',mask_path)
    #cv2.imshow('img',img); cv2.waitKey(0)

    mask = np.zeros(img.shape,np.uint8)
    data = bubbleFinder.bubbleFinder(img)
    if not data:
        return
    for [x, y, w, h] in data:
        mask[y:y + h, x:x + w], img[y:y + h, x:x + w] = textFinder.cleanBalloon(img[y:y+h, x:x+w])
        #cv2.rectangle(img, (x, y), (x + w, y + h), (30, 0, 255), 3)
        #shrink = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
        #cv2.imshow('process', shrink)
        #cv2.waitKey(0)
    cv2.imwrite(cleaned_path, img)
    ## cv2.imwrite(mask_path, mask)

def main(origin_dir, cleaned_dir='cleaned', mask_dir='mask'):
    ignores = ['*.db','*.gif','*.jpg', '*.jpeg', '*.png']
    utils.safe_copytree(origin_dir, cleaned_dir, ignores)
    utils.safe_copytree(origin_dir, mask_dir, ignores)
    textFinder = ballTextMasker.BalloonCleaner()

    origin_paths = list(utils.file_paths(origin_dir))
    expected_num_imgs = len(origin_paths)
    for origin_path in tqdm(origin_paths):
        make_and_save(origin_dir, cleaned_dir, mask_dir,
                      origin_path, textFinder)

if __name__ == '__main__':
    main(*sys.argv[1:])
'''
        old_parent = Path(origin_dir).parts[-1]

        cleaned_path = utils.make_dstpath(origin_path, old_parent, cleaned_dir) 
        ## mask_path = utils.make_dstpath(origin_path, old_parent, mask_dir) 
        cleaned_path = os.path.splitext(cleaned_path)[0] + '.png'
        ## mask_path = os.path.splitext(mask_path)[0] + '.png'

        if os.path.exists(cleaned_path):
            continue

        #print(origin_path,'|',cleaned_path,'|',mask_path)
        img = cv2.imread(origin_path)
        if img is None:
            continue
        #cv2.imshow('img',img); cv2.waitKey(0)

        mask = np.zeros(img.shape,np.uint8)
        data = bubbleFinder.bubbleFinder(img)
        for [x, y, w, h] in data:
            mask[y:y + h, x:x + w], img[y:y + h, x:x + w] = textFinder.cleanBalloon(img[y:y+h, x:x+w])
            #cv2.rectangle(img, (x, y), (x + w, y + h), (30, 0, 255), 3)
            #shrink = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
            #cv2.imshow('process', shrink)
            #cv2.waitKey(0)
        cv2.imwrite(cleaned_path, img)
        ## cv2.imwrite(mask_path, mask)
'''
