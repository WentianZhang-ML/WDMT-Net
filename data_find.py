import os
import json
import random
import sys
import glob
import xlrd
import math
from tqdm import tqdm
from random import shuffle
import numpy as np
from PIL import Image, ImageChops
from PIL import ImageOps
from skimage.feature import hog
from skimage import io, transform, color
from skimage.filters import gaussian

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--dataset", type=str, default='IDRiD', help='[IDRiD/ADAM]')
parser.add_argument("--path", type=str, default='./dataset/', help='data dir')


def IDRiD_hog_generate(data_dir):
    path_list = glob.glob(data_dir+'*/*/*.jpg')
    path_list.sort()
    for i, file in enumerate(path_list):
        name = file.split('/IDRiD/')
        file_name = name[0] + '/IDRiD_hog/' + name[1][:-4]

        image = Image.open(file).convert('L')
        image = image.resize((768,768))
        fd4, hog_image4 = hog(image, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2',
                            visualize=True, feature_vector=False)
        fd8, hog_image8 = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2',
                            visualize=True, feature_vector=False)
        fd16, hog_image16 = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2',
                            visualize=True, feature_vector=False)

        io.imsave(file_name+'_4.png', hog_image4)
        io.imsave(file_name+'_8.png', hog_image8)
        io.imsave(file_name+'_16.png', hog_image16)
def IDRiD_hog_process(data_dir, label_save_dir):
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    train_json = []
    test_json = []
    file_train = open(label_save_dir+'train_label.json', 'w')
    file_test = open(label_save_dir+'test_label.json', 'w')
    path_list = glob.glob(data_dir+'*/*/*.png')
    path_list.sort()
    print (len(path_list))
    for i, file in enumerate(path_list):
        name = file.split('/IDRiD/')
        patch = name[1][:-4].split('_')

        file_name = name[0] + '/IDRiD_hog_patch/'+patch[0]+'_'+patch[1]
        num = patch[-1]
        if '/train/normal' in file:
            dict = {}
            dict['image_path'] = file
            dict['image_hog_path'] = file_name +'_8_'+num+'.png'
            dict['image_normal_or_abormal_label'] = 0
            dict['image_ID'] = str(i)
            train_json.append(dict)

            dict = {}
            dict['image_path'] = file
            dict['image_hog_path'] = file_name +'_4_'+num+'.png'
            dict['image_normal_or_abormal_label'] = 0
            dict['image_ID'] = str(i)
            train_json.append(dict)

            dict = {}
            dict['image_path'] = file
            dict['image_hog_path'] = file_name +'_16_'+num+'.png'
            dict['image_normal_or_abormal_label'] = 0
            dict['image_ID'] = str(i)
            train_json.append(dict)

        if '/test/abnormal/' in file:
            file_name_ = name[1].split('test/abnormal/')
            file_name_roi = name[0] + '/IDRiD/label/' + file_name_[1][:-4]

            dict = {}
            dict['image_path'] = file
            dict['image_ID'] = str(i)
            dict['image_hog_path'] = file_name +'_4_'+num+'.png'
            dict['image_ROI'] = file_name_roi + '.png'
            test_json.append(dict)


    print ('train: ', len(train_json))
    print ('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()

def ADAM_hog_generate(data_dir):

    path_list = os.listdir(data_dir+'/all/')
    list_ = ['drusen/','exudate/','hemorrhage/','others/','scar/']
    abnormal_list = []
    for name in list_:
        path_list2 = os.listdir(data_dir+'/Training400-Lesion/Lesion_Masks/'+name)
        for p in path_list2:
            if p not in abnormal_list:
                abnormal_list.append(p[:-4]+'.jpg')

    for file in path_list:
        if file not in abnormal_list:
            file_name = data_dir+'/ADAM_hog/train/' + file[:-4]

        if file in abnormal_list:
            file_name = data_dir+'/ADAM_hog/test/' + file[:-4]

        image = Image.open(data_dir+'/all/'+file).convert('L')
        image = image.resize((768,768))
        fd4, hog_image4 = hog(image, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2',
                            visualize=True, feature_vector=False)
        fd8, hog_image8 = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2',
                            visualize=True, feature_vector=False)
        fd16, hog_image16 = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2',
                            visualize=True, feature_vector=False)

        io.imsave(file_name+'_4.png', hog_image4)
        io.imsave(file_name+'_8.png', hog_image8)
        io.imsave(file_name+'_16.png', hog_image16)
def hog_split(dir, dataset='IDRiD'):
    path_list = glob.glob(dir+'*/*/*.png')
    for file in path_list:
        name = file.split('/'+str(dataset)+'/')
        file_ = name[0] + '/'+str(dataset)+'_hog/' + name[1]
        image = Image.open(file_).convert('L')
        size = image.size
        w = int(size[0] // 3)
        h = int(size[1] // 3)
        file_name = name[0]+'/'+str(dataset)+'_hog_patch/'+name[1][:-4]
        flag = 1
        for j in range(3):
            for i in range (3):
                box = (w*i, h*j, w*(i+1), h*(j+1))
                region = image.crop(box)
                region.save(file_name+'_'+str(flag)+'.png')
                flag = flag+1
def ADAM_hog_process(data_dir, label_save_dir):
    if not os.path.exists(label_save_dir):
        os.makedirs(label_save_dir)
    train_json = []
    test_json = []
    file_train = open(label_save_dir+'train_label.json', 'w')
    file_test = open(label_save_dir+'test_label.json', 'w')
    path_list = glob.glob(data_dir+'*/*.png')
    path_list.sort()
    print (len(path_list))
    for i, file in enumerate(path_list):
        name = file.split('/ADAM/')
        patch = name[1][:-4].split('_')
        file_name = name[0] + '/ADAM_hog/'+patch[0]
        num = patch[-1]
        if '/train/' in file:
            dict = {}
            dict['image_path'] = file
            dict['image_hog_path'] = file_name +'_8_'+num+'.png'
            dict['image_normal_or_abormal_label'] = 0
            dict['image_ID'] = str(i)
            train_json.append(dict)

            dict = {}
            dict['image_path'] = file
            dict['image_hog_path'] = file_name +'_4_'+num+'.png'
            dict['image_normal_or_abormal_label'] = 0
            dict['image_ID'] = str(i)
            train_json.append(dict)

            dict = {}
            dict['image_path'] = file
            dict['image_hog_path'] = file_name +'_16_'+num+'.png'
            dict['image_normal_or_abormal_label'] = 0
            dict['image_ID'] = str(i)
            train_json.append(dict)

        if '/test/' in file:
            file_name_ = name[1].split('/')
            file_name_roi = name[0] + '/ADAM/label/' + file_name_[1][:-4]

            dict = {}
            dict['image_path'] = file
            dict['image_ID'] = str(i)
            dict['image_hog_path'] = file_name +'_4_'+num+'.png'
            dict['image_ROI'] = file_name_roi + '.png'
            test_json.append(dict)


    print ('train: ', len(train_json))
    print ('test: ', len(test_json))
    json.dump(train_json, file_train, indent=4)
    file_train.close()
    json.dump(test_json, file_test, indent=4)
    file_test.close()

if __name__=="__main__":
    args = parser.parse_args()
    data_dir = os.path.join(args.path,args.dataset)

    if 'IDRiD' in data_dir:
        IDRiD_hog_generate(data_dir)
        hog_split(data_dir, 'IDRiD')
        IDRiD_hog_process(data_dir, './labels/IDRiD/')

    if 'ADAM' in data_dir:
        ADAM_hog_generate(data_dir)
        hog_split(data_dir,'ADAM')
        ADAM_hog_process_3(data_dir, './labels/ADAM/')


