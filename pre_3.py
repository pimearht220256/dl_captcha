#coding=utf-8  
import cv2
import os
import shutil
import numpy as np
import pandas as pd
import lmdb
import caffe


def binary_img(img):
    """
    二值化
    :return: 二值化之后的图像
    """
    retval, t = cv2.threshold(img, 125, 1, cv2.THRESH_BINARY)
    h_sum = t.sum(axis=0)
    v_sum = t.sum(axis=1)
    x1, x2 = (v_sum > 1).nonzero()[0][0], (v_sum > 1).nonzero()[0][-1]
    y1, y2 = (h_sum > 5).nonzero()[0][0], (h_sum > 1).nonzero()[0][-1]
    im = img[x1:x2, y1:y2]
    return im	


def ding_ge(binary_im):
    """
    对图像进行顶格
    :param binary_im: 二值化的图像
    :return:
    """
    for x in xrange(0, binary_im.shape[0]):
        line_val = binary_im[x]
        # 不全部是白色(1）
        if not (line_val == 1).all():
            line_start = x
            break
    for x1 in xrange(binary_im.shape[0]-1, -1, -1):
        line_val = binary_im[x1]
        # 不全部是白色(1）
        if not (line_val == 1).all():
            line_end = x1
            break
    for y in xrange(0, binary_im.shape[1]):
        col_val = binary_im[:, y]
        # 不全部是白色(1）
        if not (col_val == 1).all():
            col_start = y
            break
    for y1 in xrange(binary_im.shape[1]-1, -1, -1):
        col_val = binary_im[:, y1]
        # 不全部是白色(1）
        if not (col_val == 1).all():
            col_end = y1
            break
    ding_ge_im = binary_im[line_start:line_end, col_start:col_end]
    # ding_ge_im = binary_im[:, col_start:col_end]
    return ding_ge_im


def make_train_char_db(ori_img_dir):
    """
    均分5份，制作训练集
    :param ori_img_dir: 原始的图像的目录
    :return:
    """
    even_split_train_path = os.path.join(os.getcwd(), 'train_data_nosp')
    if not os.path.exists(even_split_train_path):
        os.makedirs(even_split_train_path)

    train_imgs = os.listdir(ori_img_dir)
    letters = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    file = open('label.txt')
    answer_data = file.readlines()

    # 保存数据
    img = np.zeros((len(train_imgs), 1, 32, 150), dtype=np.uint8)
    label = np.zeros((len(train_imgs), 4), dtype='int')
    index = 0
    for train_img in train_imgs:
	if '._' in train_img:
	    train_img = train_img[2:]
	print train_img
        ori_train_img = os.path.join(ori_img_dir, train_img)
	s_img = cv2.imread(ori_train_img,cv2.IMREAD_GRAYSCALE)
	if(s_img is None):
	    break
	#close op
	I0 = cv2.morphologyEx(255-s_img, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
	#blackhat op
    	I1 = cv2.morphologyEx(s_img, cv2.MORPH_BLACKHAT, np.ones((5, 5), dtype=np.uint8))
	img_closed = cv2.add(I0, I1)

        binary_train_img = binary_img(img_closed)  # 二值化之后的图像
        binary_train_img = ding_ge(binary_train_img)  # 顶格之后的图像
        
        train_resize_img = cv2.resize(binary_train_img, (150, 32))
        img[index, 0] = train_resize_img
	line = answer_data[int(train_img[:-4]) - 1].replace("\r\n","")
	index2 = 0
	for i in line:
	   label[index][index2] = letters.index(i)
	   index2 += 1
	print label[index]
        index += 1
        #cv2.imwrite(os.path.join(even_split_train_path,
                                     #train_img.split('.')[0] + '.png'), train_resize_img*255)
    
    file.close()
    caffe_train_image_path = os.path.join(os.getcwd(), 'captcha_train_image_all_lmdb')
    env = lmdb.open(caffe_train_image_path,map_size=500000000)
    txn=env.begin(write=True)
    count=0
    for i in range(img.shape[0]):
        datum=caffe.io.array_to_datum(img[i])
        str_id='{:08}'.format(count)
        txn.put(str_id,datum.SerializeToString())

        count+=1
        if count%1000==0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn=env.begin(write=True)   
    caffe_train_label_path = os.path.join(os.getcwd(), 'captcha_train_label_all_lmdb') 
    env = lmdb.open(caffe_train_label_path,map_size=500000000)
    txn=env.begin(write=True)
    count = 0
    for la in label:
        datum=caffe.proto.caffe_pb2.Datum()
	datum.channels = la.shape[0]
	datum.width = 1
	datum.height = 1
	datum.data = la.tostring()
	datum.label = 0
        str_id='{:08}'.format(count)
        txn.put(str_id,datum.SerializeToString())
	
        count+=1
        if count%1000==0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn=env.begin(write=True)

    txn.commit()
    env.close()

def make_test_char_db(ori_img_dir):
    """
    均分5份，制作测试集
    :param ori_img_dir: 原始的图像的目录
    :return:
    """
    even_split_test_path = os.path.join(os.getcwd(), 'test_data_nosp')
    if not os.path.exists(even_split_test_path):
        os.makedirs(even_split_test_path)
    test_imgs = os.listdir(ori_img_dir)
    letters = list('0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    # 保存数据
    img = np.zeros((len(test_imgs), 1, 32, 150), dtype=np.uint8)
    label = np.zeros((len(test_imgs),4), dtype='int')
    file = open('5986.txt')
    answer_data = file.readlines()
    index = 0
    for test_img in test_imgs:
	if '._' in test_img:
	    test_img = test_img[2:]
	print test_img
        ori_test_img = os.path.join(ori_img_dir, test_img)	
	s_img = cv2.imread(ori_test_img,cv2.IMREAD_GRAYSCALE)
	if(s_img is None):
	    break
	#close op
	I0 = cv2.morphologyEx(255 - s_img, cv2.MORPH_CLOSE, np.ones((5, 5), dtype='uint8'))
	#blackhat op
    	I1 = cv2.morphologyEx(s_img, cv2.MORPH_BLACKHAT, np.ones((5, 5), dtype='uint8'))
	img_closed = cv2.add(I0, I1)

        binary_test_img = binary_img(img_closed)  # 二值化之后的图像
        binary_test_img = ding_ge(binary_test_img)  # 顶格之后的图像

	line = answer_data[int(test_img[:-4]) - 1].replace("\r\n","")
            
        test_resize_img = cv2.resize(binary_test_img, (150, 32))
        img[index, 0] = test_resize_img
	index2 = 0
	for i in line:
	   label[index][index2] = letters.index(i)
	   index2 += 1
	print label[index]
	index += 1
        #cv2.imwrite(os.path.join(even_split_test_path,test_img.split('.')[0] + '.png'), test_resize_img*255)

    file.close()

    caffe_test_image_path = os.path.join(os.getcwd(), 'captcha_test_image_all_lmdb')
    env = lmdb.open(caffe_test_image_path,map_size=500000000)
    txn=env.begin(write=True)
    count=0
    for i in range(img.shape[0]):
        datum=caffe.io.array_to_datum(img[i])
        str_id='{:08}'.format(count)
        txn.put(str_id,datum.SerializeToString())

        count+=1
        if count%1000==0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn=env.begin(write=True)   
    caffe_test_label_path = os.path.join(os.getcwd(), 'captcha_test_label_all_lmdb') 
    env = lmdb.open(caffe_test_label_path,map_size=500000000)
    txn=env.begin(write=True)
    count = 0
    for la in label:
        datum=caffe.proto.caffe_pb2.Datum()
	datum.channels = la.shape[0]
	datum.width = 1
	datum.height = 1
	datum.data = la.tostring()
	datum.label = 0
        str_id='{:08}'.format(count)
        txn.put(str_id,datum.SerializeToString())
	
        count+=1
        if count%1000==0:
            print('already handled with {} pictures'.format(count))
            txn.commit()
            txn=env.begin(write=True)

    txn.commit()
    env.close()		

if __name__ == '__main__':
    make_train_char_db(os.path.join(os.getcwd(), 'Joker1'))
    make_test_char_db(os.path.join(os.getcwd(), 'Joker'))
    #make_test_char_db(os.path.join(os.getcwd(), 'test'))
