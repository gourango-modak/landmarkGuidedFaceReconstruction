import os
from skimage import io
# import face_alignment
import argparse
import cv2
import sys
sys.path.insert(1, '/content/drive/MyDrive/lafin/src/')
from awingloss.eval import test_model

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the celeba img_align_celeba folder')
parser.add_argument('--output', type=str, help='path to the output folder')
args = parser.parse_args()

input_path = args.path
output_path = args.output

if not os.path.exists(output_path):
    os.mkdir(output_path)

# fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,face_detector='sfd')
filenames = os.listdir(input_path)

for i, filename in enumerate(filenames):
    if filename[-3:] != 'png' and filename[-3:] != 'jpg' and filename[-3:] != 'jpeg':
        continue
    print("[+]\t {} completed...... \t{}".format(i+1, os.path.join(input_path,filename)))
    img = cv2.imread(os.path.join(input_path,filename))
    # l_pos = fa.get_landmarks(img)
    
    img, l_pos = test_model(img, pretrained_weights = "/content/drive/MyDrive/lafin/src/awingloss/ckpt/WFLW_4HG.pth",\
                                gray_scale = False, \
                                hg_blocks = 4, end_relu = False, num_landmarks = 98)
    if  len(l_pos) < 98:
        os.remove(os.path.join(input_path,filename))
    else:
        with open(os.path.join(output_path,filename[:-4]+'.txt'), 'w') as f:
            for i in range(98):
                f.write(str(l_pos[i][0])+' '+str(l_pos[i][1])+' ')
            f.write('\n')



