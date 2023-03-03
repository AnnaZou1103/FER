import glob
import os
import shutil

idx=0
file_dir='../data'
dir_list = glob.glob('KDEF_VAL/*/')
emotion = ['AN', 'DI', 'AF', 'HA', 'SA', 'SU', 'NE']

# if os.path.exists(file_dir):
#     print('The directory exists')
#     shutil.rmtree(file_dir)
#     os.makedirs(file_dir)
# else:
#     os.makedirs(file_dir)

for fir in dir_list:
    image_list = glob.glob(fir+'*.JPG')

    for img in image_list:
        idx+=1
        if idx%100==0:
            print(idx)

        file_class = img.split('/')[-1][4:6]
        file_class = os.path.join(file_dir + '/val/', file_class)

        if not os.path.isdir(file_class):
            os.makedirs(file_class)

        shutil.copy(img, file_class + '/' + img.split('/')[2])
