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


    def trans_to_mel(self):
        tr_paths,tr_label_files,d_paths,d_label_file,te_paths = self.load_path(self.init_path)
        # print(len(tr_paths),len(tr_label_file),len(d_paths),len(d_label_file),len(te_paths))
        bird_num,bird_class = self.class_dict(tr_label_files)
        # with open("class_label.txt","w") as fp:
        #     for i in bird_class:
        #         fp.writelines(i+"   "+str(bird_class[i])+"\n")
        #     fp.close()
        # with open("train_init_num.txt","w") as fp:
        #     for i in bird_num:
        #         fp.writelines(i + "   " + str(bird_num[i]) + "\n")
        #     fp.close()
        for i in bird_num:
            bird_num[i] =0
        # print("brid_class",bird_class,"\n",len(bird_class.keys()))#检查类是否正确

        for path,tr_label_file in zip(tr_paths,tr_label_files):
            audio,fs = librosa.load(path,sr= sr)
            # print(len(audio)/fs,path)#检查audio是否与原始音频时长相近
            time = int(len(audio)/fs)
            num = int(time/5)
            for i in range(num):
                if not i%2 and time>5*(i+2):

                    start_time = 5*i+5*np.random.random(1)
                    # print("开始秒数", i,int(start_time*32000)/32000)
                    end_time = start_time+5
                    t_range =[int(start_time*32000),int(end_time*32000)]
                    audio_clip = audio[t_range[0]:t_range[1]]
                    if len(audio_clip)<160000: #检查切片是否出现问题。
                        print(start_time,end_time,len(audio)/fs,i)
                    self.mel_to_save(audio_clip,path,bird_class,bird_num,tr_label_file,fs,i,status= "train")
                elif not i%2 and time<5*(i+2):
                    start_time=5*i+(time-5*i-5)*np.random.random(1)
                    end_time = start_time+5
                    t_range = [int(start_time*32000),int(end_time*32000)]
                    audio_clip = audio[t_range[0]:t_range[1]]
                    if len(audio_clip)<160000:#检查切片是否出现问题。
                        print(start_time,end_time,len(audio)/fs,i)
                    self.mel_to_save(audio_clip,path,bird_class,bird_num,tr_label_file,fs,i,status="train")
        with open("train_clip_num.txt","w") as fp:
            for i in bird_num:
                fp.writelines(i + "   " + str(bird_num[i]) + "\n")
            fp.close()

        for path ,label in zip(d_paths,d_label_file):
            audio, fs = librosa.load(path, sr=sr)
            time = int(len(audio) / fs)
            num = int(time / 5)
            for i in range(num):
                if i%2 and time>5*(i+2):
                    start_time = 5*i+5*np.random.random(1)
                    end_time = start_time+5
                    t_range =[int(start_time*32000),int(end_time*32000)]
                    audio_clip = audio[t_range[0]:t_range[1]]
                    if len(audio_clip)<160000: #检查切片是否出现问题。
                        print(start_time,end_time,len(audio)/fs,i)
                    self.mel_to_save(audio_clip,path,bird_class,bird_num,label,fs,i,status="dev")
                elif i%2 and time<5*(i+2):
                    start_time=5*i+(time-10*i-5)*np.random.random(1)
                    end_time = start_time+5
                    t_range = [int(start_time*32000),int(end_time*32000)]
                    audio_clip = audio[t_range[0]:t_range[1]]
                    if len(audio_clip)<160000:#检查切片是否出现问题。
                        print(start_time,end_time,len(audio)/fs,i)
                    self.mel_to_save(audio_clip,path,bird_class,bird_num,label,fs,i)

        # for path in te_paths:
        #     audio, fs = librosa.load(path, sr=sr)
        #     time = int(len(audio) / fs)
        #     num = int(time / 5)
        #     for i in range(num):
        #         if i % 2 and time > 5 * (i + 2):
        #             start_time = 5 * i + 5 * np.random.random(1)
        #             end_time = start_time + 5
        #             t_range = [int(start_time * 32000), int(end_time * 32000)]
        #             audio_clip = audio[t_range[0]:t_range[1]]
        #             if len(audio_clip) < 160000:  # 检查切片是否出现问题。
        #                 print(start_time, end_time, len(audio) / fs, i)
        #             mel = librosa.feature.melspectrogram(y=audio_clip, sr=fs, n_mels=256, n_fft=2048, fmin=fmin,
        #                                                  power=2.0, hop_length=512)
        #             mel = 20.0 / 2.0 * np.log10(mel + sys.float_info.epsilon)
        #             maxx = mel - np.mean(mel, axis=0)
        #             mel = (mel - np.array(np.max(maxx))) / np.mean(mel)
        #             data = np.array(mel, dtype="float64")
        #             f = h5py.File("/media/brl/Data1/gly/birds_mel/test_data/test_" + path.split('.')[0].split('/')[-1] + "_" + str(i) + ".h5",'w')
        #             f.create_dataset('audio_data', data=data)
        #             f.close()
        #             print(path,data.shape)
        #         elif i % 2 and time < 5 * (i + 2):
        #             start_time = 5 * i + (time - 10 * i - 5) * np.random.random(1)
        #             end_time = start_time + 5
        #             t_range = [int(start_time * 32000), int(end_time * 32000)]
        #             audio_clip = audio[t_range[0]:t_range[1]]
        #             if len(audio_clip) < 160000:  # 检查切片是否出现问题。
        #                 print(start_time, end_time, len(audio) / fs, i)
        #             mel = librosa.feature.melspectrogram(y=audio_clip, sr=fs, n_mels=256, n_fft=2048, fmin=fmin,
        #                                                  power=2.0, hop_length=512)
        #             mel = 20.0 / 2.0 * np.log10(mel + sys.float_info.epsilon)
        #             maxx = mel - np.mean(mel, axis=0)
        #             mel = (mel - np.array(np.max(maxx))) / np.mean(mel)
        #             data = np.array(mel, dtype="float64")
        #             f = h5py.File("/media/brl/Data1/gly/birds_mel/test_data/test_" + path.split('.')[0].split('/')[-1] + "_" + str(i) + ".h5",
        #                           'w')
        #             f.create_dataset('audio_data', data=data)
        #             f.close()
        #             print(path,data.shape)

            # print(data.shape)



if __name__ == '__main__':
    data = data_preprocess()
    data.trans_to_mel()