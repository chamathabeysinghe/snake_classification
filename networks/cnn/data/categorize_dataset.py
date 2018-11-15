import os, os.path
from shutil import copyfile

dataset_dir = '../../../dataset/images/{}/'
train_dir = 'train/{}/'
validation_dir = 'validation/{}/'

sample_count = 0

for i in range(1,239):
    try:
        current_directory = dataset_dir.format(i)
        files = [name for name in os.listdir(current_directory) if os.path.isfile(os.path.join(current_directory, name))]
        count = len(files)
        if count >= 2:
            sample_count += 1
            if not os.path.exists(train_dir.format(i)):
                os.makedirs(train_dir.format(i))
            if not os.path.exists(validation_dir.format(i)):
                os.makedirs(validation_dir.format(i))
            for j in range(count):
                if j%2==0:
                    copyfile(current_directory+files[j],train_dir.format(i)+files[j])
                else:
                    copyfile(current_directory + files[j], validation_dir.format(i) + files[j])

    except FileNotFoundError:
        print('Directory not found for {}'.format(i))

print(sample_count)
