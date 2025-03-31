import os
import os.path
import numpy as np
import json
import argparse
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

def is_worddata(jsonfname):
    with open(jsonfname,encoding='utf-8') as json_file:
        json_data = json.load(json_file)
        disc = json_data["info"]["description"]
        if 'word handwrite' in disc:
            return True
    return False

def write_listfile(imgfname, jsonfname, file):
    data = imgfname + ',' + jsonfname + "\n"
    file.write(data)


def write_csv_datafile(imgfname, jsonfname, file):

    #annot_list = []
    if is_worddata(jsonfname) == True:
        with open(jsonfname,encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            width = json_data["image"]["width"]
            height = json_data["image"]["height"]
            word = json_data["text"]["word"]
            c_cnt = len(word)
            for i in range(c_cnt):
                box = word[i]["charbox"]
                val = word[i]["value"]
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                data = imgfname + ',' + str(x1) + ',' + str(y1) + ',' + str(x2) + ',' + str(y2) + ",HandWrited_KR" + "\n"
                #annot_list.append(data)
                file.write(data)
    #return annot_list

def get_filelist(dirname):

    imagelist = []
    jsonlist = []

    try:
        for(path,dir,files) in os.walk(dirname):
            for filename in files:
                split = os.path.splitext(filename)
                if split[1] == '.jpg':
                    full_filename = os.path.join(path, filename)
                    full_jsonname = os.path.join(path,split[0]) + '.json'
                    imagelist.append(full_filename)
                    jsonlist.append(full_jsonname)
        return imagelist, jsonlist
    except PermissionError:
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./traindata",
                        help="File path to data file")
    parser.add_argument("--csv_dir", type=str, default="./traindata",
                        help="File path to save CSV file")
    parser.add_argument("--val",type=float, default=0.2,help="validation list ratio. default = 0.2")
    parser.add_argument("--kfold",type=int, default=1,help="make validation list kfold. default = 1")

    args = parser.parse_args()

    imglist, jsonlist = get_filelist(args.data_dir)
    if args.kfold > 1:
        indics = np.arange(len(imglist))
        np.random.seed(1234)
        np.random.shuffle(indics)
        valcnt = int(len(imglist)/args.kfold)
        for k in range(args.kfold):
            ftrimg = open(args.csv_dir + '/trainimglist_'+str(k)+'.csv',"w")
            feval = open(args.csv_dir + '/evalimglist_'+str(k)+'.csv',"w")
            fval = open(args.csv_dir + '/validdata_'+str(k)+'.csv',"w")
            ftr =  open(args.csv_dir + '/traindata_'+str(k)+'.csv',"w")
            st = valcnt*k
            en = valcnt*(k+1)
            for i in range(len(indics)):
                if i>=st and i< en:
                    write_csv_datafile(imglist[indics[i]],jsonlist[indics[i]],fval)
                    write_listfile(imglist[indics[i]],jsonlist[indics[i]],feval)
                else:
                    write_csv_datafile(imglist[indics[i]],jsonlist[indics[i]],ftr)
                    write_listfile(imglist[indics[i]],jsonlist[indics[i]],ftrimg)
            feval.close()
            fval.close()
            ftr.close()
            ftrimg.close()
    else:
        if args.val == 0.0:
            f = open(args.csv_dir + '/traindata.csv',"w")
            ftrimg = open(args.csv_dir + '/trainimglist.csv',"w")
            for i in range(len(imglist)):
                write_csv_datafile(imglist[i],jsonlist[i],f)
                write_listfile(imglist[i],jsonlist[i],ftrimg)
            f.close()
            ftrimg.close()
        else:
            train_img,val_img,train_json,val_json = train_test_split(imglist,jsonlist,test_size=args.val,random_state=1234)
            f = open(args.csv_dir + '/traindata.csv',"w")
            ftrimg = open(args.csv_dir + '/trainimglist.csv',"w")
            for i in range(len(train_img)):
                write_csv_datafile(train_img[i],train_json[i],f)
                write_listfile(train_img[i],train_json[i],ftrimg)
            f.close()
            ftrimg.close()
            f = open(args.csv_dir + '/validdata.csv',"w")
            feval = open(args.csv_dir + '/evalimglist.csv',"w")
            for i in range(len(val_img)):
                write_csv_datafile(val_img[i],val_json[i],f)
                write_listfile(val_img[i],val_json[i],feval)
            f.close()
            feval.close()



if __name__ == "__main__":
    main()
