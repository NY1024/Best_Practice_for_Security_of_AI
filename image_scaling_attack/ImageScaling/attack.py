import sys
import os
import numpy as np
import cv2
from PIL import Image
import utils
import argparse
import time
from l2 import *

if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser(description=
                                      "Launch the scaling attack.")
    parser.add_argument('-s', '--sourceImg',
                        type=str,
                        help="The path of the soruce image.",
                        required=True)
    parser.add_argument('-t', '--targetImg',
                        type=str,
                        help="The path of the target image.",
                        required=True)
    parser.add_argument('-a', '--attackImg',
                        type=str,
                        help="The path of the attack image.",
                        required=True)
    parser.add_argument('-o', '--outputImg',
                        type=str,
                        help="The path of the ouput image.",
                        default="")
    parser.add_argument('-f', '--resizeFunc',
                        type=str,
                        help="The resizing function: cv2.resize(default), \
                                                     Image.Image.resize",
                        default="cv2.reisze")
    parser.add_argument('-i', '--interpolation',
                        type=str,
                        help="The interpolation method. \
                              (Default:cv2.INTER_LINEAR)",
                        default="cv2.INTER_LINEAR")
    parser.add_argument('-p', '--penalty',
                        type=float,
                        help="The penalty set in the attack. \
                              (Default:1)",
                        default=1)
    parser.add_argument('-m', '--imageFactor',
                        type=float,
                        help="The factor used to \
                              scale image pixel value to [0,1]. \
                              (Default:255)",
                        default=255.)

    args = parser.parse_args()



    sourceImg = utils.imgLoader(args.sourceImg,1)
    targetImg = utils.imgLoader(args.targetImg,1)

    func_lookup = {'cv2.resize': cv2.resize,
                   'Image.Image.resize': Image.Image.resize}

    interpolation_lookup = {'cv2.INTER_NEAREST': cv2.INTER_NEAREST,
                            'cv2.INTER_LINEAR': cv2.INTER_LINEAR,
                            'cv2.INTER_CUBIC': cv2.INTER_CUBIC,
                            'cv2.INTER_AREA': cv2.INTER_AREA,
                            'cv2.INTER_LANCZOS4': cv2.INTER_LANCZOS4,
                            'Image.NEAREST': Image.NEAREST,
                            'Image.LANCZOS': Image.LANCZOS,
                            'Image.BILINEAR': Image.BILINEAR,
                            'Image.BICUBIC': Image.BICUBIC,}

    sc = ScalingAttack(sourceImg,
                              targetImg,
                              func=func_lookup[args.resizeFunc],
                              interpolation=interpolation_lookup[args.interpolation],
                              penalty=args.penalty,
                              img_factor=args.imageFactor)

    attackImg = sc.attack()
    utils.imgSaver(args.attackImg, attackImg)

    if len(args.outputImg) > 0:
        attackImg = utils.imgLoader(args.attackImg,1)
        outputImg = sc._resize(attackImg, (targetImg.shape[0], targetImg.shape[1]))
        utils.imgSaver(args.outputImg, outputImg)

    time_cost = time.time() - start_time
    print("Time cost:", time_cost)
