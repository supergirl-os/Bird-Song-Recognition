import csv
import librosa
import numpy as np
from skimage.transform import resize
from PIL import Image
import sys
import os
import pandas as pd
from tqdm import tqdm
# import crnn
import h5py
from pydub.utils import make_chunks
sr = 32000
class data_preprocess(object):
    def __init__(self):
        self.init_path = "/media/brl/Seagate Expansion Drive/bird_100_data"
        self.save_path = "/media/brl/Data1/gly/birds_mel/train_data_3"
    def load_path(self,init_path):

        datasets = os.listdir(init_path)
        tr_paths = []
        tr_label_file = []
        d_paths = []
        d_label_file = []
        te_paths = []

        for dataset in datasets:
            if dataset == "train_data":
                dataset_t = "train_data"
                cates = os.listdir(init_path + os.sep + dataset+os.sep+dataset_t)
                for cate in cates:
                    files = os.listdir(init_path + os.sep + dataset + os.sep+dataset_t + os.sep + cate)
                    for file in files:
                        path = init_path + os.sep + dataset + os.sep+dataset_t + os.sep + cate + os.sep+file
                        tr_paths.append(path)
                        tr_label_file.append(cate)
            elif dataset == "test_data":
                dataset_t = "test_data_blind_name"
                files = os.listdir(init_path+os.sep+dataset+os.sep+dataset_t)
                for file in files:
                    path = init_path + os.sep + dataset + os.sep+dataset_t + os.sep +file
                    te_paths.append(path)
            elif dataset =="dev_data":
                cates = os.listdir(init_path+os.sep+dataset)
                for cate in cates:
                    files = os.listdir(init_path+os.sep+dataset+os.sep+cate)
                    for file in files:
                        path= init_path+os.sep+dataset+os.sep+cate+os.sep+file
                        d_paths.append(path)
                        d_label_file.append(cate)
        return tr_paths,tr_label_file,d_paths,d_label_file,te_paths

    def class_dict(self,tr_labels):
        bird_dict={}
        bird_num_dict = {}
        for label in tr_labels:
            if label not in bird_num_dict:
                bird_num_dict[label] = 1
            else:
                bird_num_dict[label]+=1
        for label in tr_labels:
            if label not in bird_dict:
                bird_dict[label] = len(bird_dict)
            else:
                continue
        return bird_num_dict,bird_dict

    def audio_cut(self):
        tr_paths,tr_label_files,d_paths,d_label_file,te_paths = self.load_path(self.init_path)
        # print(len(tr_paths),len(tr_label_file),len(d_paths),len(d_label_file),len(te_paths))
        bird_num,bird_class = self.class_dict(tr_label_files)
        for i in bird_num:
            bird_num[i] =0
        # print("brid_class",bird_class,"\n",len(bird_class.keys()))#检查类是否正确
        result = []
        for path,tr_label_file in zip(tr_paths,tr_label_files):
            audio,fs = librosa.load(path,sr= sr)
            # print(len(audio)/fs,path)#检查audio是否与原始音频时长相近
            time = int(len(audio)/fs)
            num = int(time/10)
            if time>10:
                start_time =np.random.randint(time-10,size=1)
                duration = 10
                end_time = start_time + duration
                audio_dst = audio[start_time * fs:end_time* fs]
            else:
                audio_dst = np.hstack((audio, audio))
                print('x=', audio_dst.shape)
            save_path=self.save_path+os.sep+"train_"+path.split('/')[-1]

            librosa.output.write_wav(save_path, audio_dst, fs)
            print("successfully to move!")

            tmp = save_path+","+ tr_label_file
            print("tmp=", tmp)
            result.append(tmp)

            # 写入csv文件
        df = pd.DataFrame(result)
        df.to_csv('train.csv', mode='a', index=None, header=False)
        print("successfully!!!!")


a=data_preprocess()
a.audio_cut()