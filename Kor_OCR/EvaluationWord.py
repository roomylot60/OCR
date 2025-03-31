
import cv2
import os
import numpy as np
import time
import tensorflow as tf
import csv
import json
import argparse
import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
import matplotlib.pyplot as plt
from PIL import Image

from unicode import split_syllable_char, join_jamos_char, is_hangul

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

kr_first_label = {0: 'HandWrited_NUM',
                   1: 'HandWrited_EN',
                   2: 'HandWrited_KR',
                   3: 'HandWrited_SP',
                   4: 'Printed_NUM',
                   5: 'Printed_EN',
                   6: 'Printed_KR',
                   7: 'Printed_SP',}                    ## replace with your model labels and its index value




labels_to_names = {0: 'HandWrited_NUM',
                   1: 'HandWrited_EN',
                   2: 'HandWrited_KR',
                   3: 'HandWrited_SP',
                   4: 'Printed_NUM',
                   5: 'Printed_EN',
                   6: 'Printed_KR',
                   7: 'Printed_SP',}                    ## replace with your model labels and its index value


label2ja = {0: 'ㄱ', 1: 'ㄲ', 2: 'ㄴ', 3: 'ㄷ', 4: 'ㄸ', 5: 'ㄹ',
            6: 'ㅁ', 7: 'ㅂ', 8: 'ㅃ', 9: 'ㅅ', 10: 'ㅆ', 11: 'ㅇ',
            12: 'ㅈ', 13: 'ㅉ', 14: 'ㅊ', 15: 'ㅋ', 16: 'ㅌ', 17: 'ㅍ', 18: 'ㅎ'}

ja2label = {'ㄱ':0,
'ㄲ':1,
'ㄴ':2,
'ㄷ':3,
'ㄸ':4,
'ㄹ':5,
'ㅁ':6,
'ㅂ':7,
'ㅃ':8,
'ㅅ':9,
'ㅆ':10,
'ㅇ':11,
'ㅈ':12,
'ㅉ':13,
'ㅊ':14,
'ㅋ':15,
'ㅌ':16,
'ㅍ':17,
'ㅎ':18}

label2mo = {0: 'ㅏ', 1: 'ㅐ', 2: 'ㅑ', 3: 'ㅒ', 4: 'ㅓ', 5: 'ㅔ',
            6: 'ㅕ', 7: 'ㅖ', 8: 'ㅗ', 9: 'ㅘ', 10: 'ㅙ', 11: 'ㅚ',
            12: 'ㅛ', 13: 'ㅜ', 14: 'ㅝ', 15: 'ㅞ', 16: 'ㅟ', 17: 'ㅠ',
            18: 'ㅡ', 19: 'ㅢ', 20: 'ㅣ'}
mo2label = {'ㅏ':0,
'ㅐ':1,
'ㅑ':2,
'ㅒ':3,
'ㅓ':4,
'ㅔ':5,
'ㅕ':6,
'ㅖ':7,
'ㅗ':8,
'ㅘ':9,
'ㅙ':10,
'ㅚ':11,
'ㅛ':12,
'ㅜ':13,
'ㅝ':14,
'ㅞ':15,
'ㅟ':16,
'ㅠ':17,
'ㅡ':18,
'ㅢ':19,
'ㅣ':20}


label2ba = {0: None, 1: 'ㄱ', 2: 'ㄲ', 3: 'ㄳ', 4: 'ㄴ', 5: 'ㄵ',
            6: 'ㄶ', 7: 'ㄷ', 8: 'ㄹ', 9: 'ㄺ', 10: 'ㄻ', 11: 'ㄼ',
            12: 'ㄽ', 13: 'ㄾ', 14: 'ㄿ', 15: 'ㅀ', 16: 'ㅁ', 17: 'ㅂ',
            18: 'ㅄ', 19: 'ㅅ', 20: 'ㅆ', 21: 'ㅇ', 22: 'ㅈ', 23: 'ㅊ',
            24: 'ㅋ', 25: 'ㅌ', 26: 'ㅍ', 27: 'ㅎ'}

ba2label = {None:0,
'ㄱ':1,
'ㄲ':2,
'ㄳ':3,
'ㄴ':4,
'ㄵ':5,
'ㄶ':6,
'ㄷ':7,
'ㄹ':8,
'ㄺ':9,
'ㄻ':10,
'ㄼ':11,
'ㄽ':12,
'ㄾ':13,
'ㄿ':14,
'ㅀ':15,
'ㅁ':16,
'ㅂ':17,
'ㅄ':18,
'ㅅ':19,
'ㅆ':20,
'ㅇ':21,
'ㅈ':22,
'ㅊ':23,
'ㅋ':24,
'ㅌ':25,
'ㅍ':26,
'ㅎ':27}



def segment_char(model, image_path, threshold=0.7):
    image = cv2.imread(image_path)
    draw = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    segresult = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        if score > threshold:
            b = box.astype(int)
            segresult.append(b)

    segresult.sort(key=lambda x:x[0])

    return segresult

def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni

    return iou


def get_max_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax

def recog_chars(model, imgfname, charboxs):
    input_width  = 64  # width
    input_height = 64  # height

    out_str = ''
    x_data = []
    vals = []
    img = Image.open(imgfname)
    for i in range(len(charboxs)):
        x1 = charboxs[i][0]
        y1 = charboxs[i][1]
        x2 = charboxs[i][2]
        y2 = charboxs[i][3]
        patch=img.crop((x1,y1,x2,y2))
        resized = patch.resize((int(input_width), int(input_height)),Image.ANTIALIAS)
        x_data.append(np.asarray(resized)/255.0)

    x_data=np.asarray(x_data)
    pred_ja, pred_mo, pred_ba = model.predict(x_data,verbose=0)
    for i in range(len(pred_ja)):
        j_idx = pred_ja[i].argmax()
        m_idx = pred_mo[i].argmax()
        b_idx = pred_ba[i].argmax()
        ja = label2ja[j_idx]
        mo = label2mo[m_idx]
        ba = label2ba[b_idx]
        ch = join_jamos_char(ja,mo,ba)
        out_str += ch
        vals.append(ch)

    return out_str, vals

def load_WordAnnotation(jsonfname):

    boxs = []
    vals = []

    with open(jsonfname,encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        width = json_data["image"]["width"]
        height = json_data["image"]["height"]
        word = json_data["text"]["word"]
        c_cnt = len(word)
        for i in range(c_cnt):
            box = word[i]["charbox"]
            val = word[i]["value"]
            boxs.append(box)
            vals.append(val)

    return boxs, vals

def get_filelist_from_csv(csvfname):
    imagelist = []
    jsonlist = []

    f = open(csvfname, 'r', encoding='utf-8')
    rdr = csv.reader(f)
    for line in rdr:
        imagelist.append(line[0])
        jsonlist.append(line[1])

    f.close()
    return imagelist, jsonlist


def EvaluationWordRecog(seg,rec,args):
    #load list
    imglist, jsonlist = get_filelist_from_csv(args.input)

    total_char_cnt = 0
    total_word_cnt = 0
    total_seg_cnt = 0
    correct_char_cnt = 0
    correct_seg_cnt = 0
    corrent_word_cnt = 0
    comp = 0

    f = open(args.output,"w")

    data = 'GT(left),GT(top),GT(right),GT(bottom),Pred(left),Pred(top),Pred(right),Pred(bottom),IoU,GT(Value),Pred(Value)\n'
    f.write(data)

    total_word_cnt = len(imglist)

    for i in range(total_word_cnt):
        boxs, vals = load_WordAnnotation(jsonlist[i])
        charboxs = segment_char(seg,imglist[i])
        result_word,chs = recog_chars(rec,imglist[i],charboxs)
        data = ''

        comp = 0

        boxs = np.asarray(boxs)
        charboxs = np.asarray(charboxs)

        total_seg_cnt += len(charboxs)
        total_char_cnt += len(boxs)
        #calculate iou
        for j in range(len(boxs)):
            data += (str(boxs[j][0]) + ',' + str(boxs[j][1]) + ',' + str(boxs[j][2]) + ',' + str(boxs[j][3]))
            data += ','
            iou, iou_max, nmax = get_max_iou(charboxs,boxs[j])
            data += (str(charboxs[nmax][0]) + ',' + str(charboxs[nmax][1]) + ',' + str(charboxs[nmax][2]) + ',' + str(charboxs[nmax][3]))
            data += ','
            data += str(iou_max)
            data += ','
            data += vals[j]
            data += ','
            data += chs[nmax]
            data += '\n'
            if vals[j] != chs[nmax]:
                comp = 1
            else:
                correct_char_cnt += 1
            if iou_max >= 0.8:
                correct_seg_cnt += 1

        f.write(data)

        if comp == 0:
            corrent_word_cnt +=1
        print('total: {tcnt}, current : {cur} , correct_word : {cwc} \r'.format(tcnt = total_word_cnt, cur = i, cwc = corrent_word_cnt))

    data = '\n'
    f.write(data)
    print(data)

    data = 'Total word count : ' + str(total_word_cnt) + ',' + 'correct word count : ' + str(corrent_word_cnt) + '\n'
    f.write(data)
    print(data)

    data = 'Word EM : ' + str(corrent_word_cnt/total_word_cnt) + '\n'
    f.write(data)
    print(data)

    data = 'Total char count : ' + str(total_char_cnt) + ',' + 'correct char count : ' + str(correct_char_cnt) + '\n'
    f.write(data)
    print(data)

    data = 'Char EM : ' + str(correct_char_cnt/total_char_cnt) + '\n'
    f.write(data)
    print(data)

    data = 'Total segmented char count : ' + str(total_seg_cnt) + ',' + 'correct seg count : ' + str(correct_seg_cnt) + '\n'
    f.write(data)
    print(data)

    P = correct_seg_cnt/total_seg_cnt
    R = correct_char_cnt/total_char_cnt 
    f1 = 2*(P*R)/(P+R)

    data = 'P : 세그먼트 정확도' + str(P) + '\n'
    f.write(data)
    print(data)
    data = 'R : 글자인식 정확도' + str(R) + '\n'
    f.write(data)
    print(data)
    data = 'f1-score : 조화평균' + str(f1) + '\n'
    f.write(data)
    print(data)


    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment_model", type=str,default="./saved_model/HWKR_Seg/hwkrseg_infer_10.h5",
                       help="filename of segment model")
    parser.add_argument("--recog_model", type=str,default="./saved_model/HWKR_Char/NTCharRecog_20.h5",
                       help="filename of char recog model")
    parser.add_argument("--input", type=str, default="./evallist/evalimglist_0.csv",
                        help="filename of evaluation data list")
    parser.add_argument("--output", type=str, default="./evallist/eval_result.csv",
                        help="filename of evaluation result")
    args = parser.parse_args()

    if args.segment_model is not None:
        segment_model = models.load_model(args.segment_model, backbone_name='resnet50')

        if args.recog_model is not None:
            recog_model = keras.models.load_model(args.recog_model)
            EvaluationWordRecog(segment_model, recog_model, args)

if __name__ == "__main__":
    main()
