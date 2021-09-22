import argparse
import os
import cv2
import glob
import numpy as np

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--frame_dir", type=str)
parser.add_argument("--output_dir", type=str)
args = parser.parse_args()

def resize_scale(frame, myshape = (512, 1024, 3)):
    curshape = frame.shape
    if curshape == myshape:
        scale = 1
        translate = (0.0, 0.0)
        return scale, translate

    x_mult = myshape[0] / float(curshape[0])
    y_mult = myshape[1] / float(curshape[1])

    if x_mult == y_mult:
        scale = x_mult
        translate = (0.0, 0.0)
    elif y_mult > x_mult:
        y_new = x_mult * float(curshape[1])
        translate_y = (myshape[1] - y_new) / 2.0
        scale = x_mult
        translate = (translate_y, 0.0)
    elif x_mult > y_mult:
        x_new = y_mult * float(curshape[0])
        translate_x = (myshape[0] - x_new) / 2.0
        scale = y_mult
        translate = (0.0, translate_x)
        
    # M = np.float32([[scale,0,translate[0]],[0,scale,translate[1]]])
    # output_image = cv2.warpAffine(frame,M,(myshape[1],myshape[0]))
    return scale, translate

def fix_image(scale, translate, frame, myshape = (512, 1024, 3)):
    M = np.float32([[scale,0,translate[0]],[0,scale,translate[1]]])
    output_image = cv2.warpAffine(frame,M,(myshape[1],myshape[0]))
    return output_image

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    
img_path = os.path.join(args.frame_dir, "*.png")
frames = glob.glob(img_path)
frames.sort()

for i in range(len(frames)):
    _frame = cv2.imread(frames[i])
    cv2.waitKey(0)
    scale_n, translate_n = resize_scale(_frame, (1024, 1024, 3))
    out_frame = fix_image(scale_n, translate_n, _frame, myshape = (1024, 1024, 3))
    _filename = "frame_"+'{:0>12}'.format(i)+".png"
    filename = os.path.join(args.output_dir, _filename)
    cv2.imwrite(filename, out_frame)
