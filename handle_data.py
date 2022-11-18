import shutil
import os
import random
import numpy as np
import csv
import librosa
save_to_disk = 0
import numpy as np
from skimage.transform import resize
from PIL import Image
import sys
import os
from tqdm import tqdm
# import crnn
import h5py
import math
fft = 2048
hop = 512
# Less rounding errors this way
sr = 32000
length = 8 * sr


np.random.seed(1234)

fmin = 300


# Get some safety margin
fmin = int(fmin * 0.9)

print('Minimum frequency: ' + str(fmin) + ', maximum frequency: ' + str(0))


class data_preprocess(object):
    def __init__(self):
        self.init_path = "/media/brl/Seagate Expansion Drive/bird_100_data"
        self.save_path="/media/brl/Data1/gly/birds_mel"
        #用的话可以传initpath等参数,后面用arg或者json

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

    def mel_to_save(self,audio,path,bird_class,bird_num,tr_label_file,fs,i,status = "train"):
        if status !="other":
            mel = librosa.feature.melspectrogram(y=audio, sr=fs, n_mels=256, n_fft=2048, fmin=fmin, power=2.0,
                                                 hop_length=512)
            mel = 20.0 / 2.0 * np.log10(mel + sys.float_info.epsilon)
            maxx = mel - np.mean(mel, axis=0)
            mel = (mel - np.array(np.max(maxx))) / np.mean(mel)
            data = np.array(mel, dtype="float64")
            f = h5py.File("/media/brl/Data1/gly/birds_mel/" + status + "_data/" + status + "_" + path.split('.')[0].split('/')[
                -1] + "_" + str(i) + ".h5", 'w')
            f.create_dataset('audio_data', data=data)
            f.create_dataset('audio_label', data=[bird_class[tr_label_file]])
            f.close()
            print(path, [bird_class[tr_label_file]])
        if status=="train":
            bird_num[tr_label_file]+=1

    def handle_data(self,paths,label_files,bird_num):
        for path,tr_label_file in zip(paths,label_files):
            audio,fs = librosa.load(path,sr= sr)
            # print(len(audio)/fs,path)#检查audio是否与原始音频时长相近
            time = int(len(audio)/fs)
            num=int(math.log(time,5))
            for i in range(num):
                #复制过去
                src = path
                dst_path = self.save_path+os.sep+"traindata1"+os.sep+tr_label_file
                isexists = os.path.exists(dst_path)
                if not isexists:
                    os.mkdir(dst_path)
                dst=dst_path+os.sep+path.split('.')[0].split('/')[-1]+"_"+str(i)+".wav"
                shutil.move(src, dst)
                bird_num[tr_label_file] += 1
        return bird_num

    def handle_data_helper(self):
        tr_paths,tr_label_files,d_paths,d_label_file,te_paths = self.load_path(self.init_path)
        bird_num,bird_class = self.class_dict(tr_label_files)
        for i in bird_num:
            bird_num[i] =0
        # print("brid_class",bird_class,"\n",len(bird_class.keys()))#检查类是否正确
        bird_num = self.handle_data(tr_paths,tr_label_files,bird_num)
        print("bird_num=",bird_num)

        #self.handle_data(d_paths,d_label_file)

        with open("train_clip_num.txt","w") as fp:
            for i in bird_num:
                fp.writelines(i + "   " + str(bird_num[i]) + "\n")
            fp.close()

        # for path in te_paths:
        #     audio, fs = librosa.load(path, sr=sr)
        #     time = int(len(audio) / fs)
        #     num = math.log(time, 5)
        #     for i in range(num):
        #         # 复制过去
        #         src = path
        #         dst= self.save_path + os.sep + "testdata1/" +path.split('.')[0].split('/')[-1] + "_" + str(i) + ".wav"
        #         shutil.move(src, dst)

    def select_file(self,labels,bird_num):
        for label in labels:
            path = self.save_path +os.sep+"traindata1/"+label
            save_path = self.save_path+os.sep+"train_data_1/"+label
            isexists = os.path.exists(save_path)
            if not isexists:
                os.mkdir(save_path)
            num=bird_num[label]
            print("num=",num)
            count=math.ceil(math.log(num,2)/math.log(11,2))
            print("count=",count)
            for i in range(count):
                files=os.listdir(path)
                for file in files:
                    src=path+os.sep+file
                    dst=save_path+os.sep+file.split('.')[0]+str(i)+".wav"
                    shutil.copy(src,dst)

    def data_processing_dev(self,paths,labels,bird_num):
        for path ,label in zip(paths,labels):
            audio, fs = librosa.load(path, sr=sr)
            time = int(len(audio) / fs)
            num = int(time / 5)
            print("time=", time)
            if time > 5:
                start_time = np.random.randint(time - 5, size=1)
                # print("开始秒数", i,int(start_time*32000)/32000)
            else:
                start_time = 0
            end_time = start_time + 5
            print("start_time=", start_time)
            print("end_time=", end_time)
            t_range = [int(start_time * 32000), int(end_time * 32000)]
            audio_clip = audio[t_range[0]:t_range[1]]
            if len(audio_clip) < 160000:  # 检查切片是否出现问题。
                print(start_time, end_time, len(audio) / fs, i)
            self.mel_to_save(audio_clip,path,bird_class,bird_num,label,fs,i)





def remove_file(old_path, new_path):
    print(old_path)
    print(new_path)
    filelist = os.listdir(old_path)  # 列出该目录下的所有文件,listdir返回的文件列表是不包含路径的。
    filelist.sort(key=lambda x: int(x.split('_')[2]))
    filelist.sort(key=lambda x: int(x.split('_')[1][1:]))
    print("filelist=",filelist)



if __name__ == '__main__':
    # path= r"/media/brl/Data1/gly/birds_mel/train_data_dec"
    # filelist = os.listdir(path)
    # print(len(filelist))
    #remove_file(r"/media/brl/Data1/gly/birds_mel/train_data", r"/media/brl/Data1/gly/birds_mel/train_data_dec")
    data = data_preprocess()
    data.handle_data_helper()
