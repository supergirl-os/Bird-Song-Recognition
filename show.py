import os
import matplotlib
matplotlib.use('Agg') # No pictures displayed
import pylab
import librosa
import librosa.display
import numpy as np
import h5py
#创建文件夹,还需要判断文件夹是否存在。os.path.exit啥啥啥来着，
t_path="/media/brl/Data1/gly/display/"
path="/media/brl/Data1/gly/birds_mel/train_data_new"
files=os.listdir(path)
for i in range(100):
    p="B"+str(i).rjust(3,'0')
    dst_path=t_path+p
    isexists=os.path.exists(dst_path)
    if not isexists:
        os.mkdir(dst_path)

for file in files:
    tmp = h5py.File(path+os.sep+file, 'r')
    train_set_data = tmp['audio_data'][:]
    train_set_label = tmp['audio_label'][:]
    # make pictures nam
    save_path=t_path+file.split('_')[1]+os.sep+file.split('.')[0]+".jpg"
    print("save_path=",save_path)

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge

    librosa.display.specshow(librosa.power_to_db(train_set_data, ref=np.max))
    pylab.savefig(save_path, bbox_inches=None, pad_inches=0)
    pylab.close()