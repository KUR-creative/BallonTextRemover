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


def main(origin_dir, cleaned_dir, mask_dir):
    utils.safe_copytree(origin_dir, cleaned_dir, ['*.jpg', '*.jpeg', '*.png'])
    utils.safe_copytree(origin_dir, mask_dir, ['*.jpg', '*.jpeg', '*.png'])
    textFinder = ballTextMasker.BalloonCleaner()

    for origin_path in utils.file_paths(origin_dir):
        old_parent = Path(origin_dir).parts[-1]
        cleaned_path = utils.make_dstpath(origin_path, old_parent, cleaned_dir) 
        mask_path  = utils.make_dstpath(origin_path, old_parent, mask_dir) 
        print(origin_path,cleaned_path,mask_path)
        img = cv2.imread(origin_path)
        cv2.imshow('img',img); cv2.waitKey(0)
        '''
        mask = np.zeros(img.shape,np.uint8)
        data = bubbleFinder.bubbleFinder(img)
        for [x, y, w, h] in data:
            mask[y:y + h, x:x + w], img[y:y + h, x:x + w] = textFinder.cleanBalloon(img[y:y+h, x:x+w])
            cv2.rectangle(img, (x, y), (x + w, y + h), (30, 0, 255), 3)
            #shrink = cv2.resize(img, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_AREA)
            #cv2.imshow('process', shrink)
            #cv2.waitKey(0)
        cv2.imwrite(cleaned_path,img)
        cv2.imwrite(mask_path, mask)
        '''

if __name__ == '__main__':
    main(sys.argv[1], 'cleaned', 'mask')
