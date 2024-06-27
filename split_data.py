import os
import sklearn
import sklearn.model_selection
import shutil

row_data_dir= "data/bloodcells_dataset"

# create data and split

data_set_dir = "data_set_test"
train_dir_path = os.path.join(data_set_dir,"train")
valid_dir_path = os.path.join(data_set_dir,"valid")
test_dir_path = os.path.join(data_set_dir,"test")

def copy_file(img_file_list,source_dir,tagets_dir):
    for img_file in img_file_list:
        shutil.copy(os.path.join(source_dir,img_file),os.path.join(tagets_dir))

for class_name in os.listdir(row_data_dir):
    class_dir_path = os.path.join(row_data_dir,class_name)

    if os.path.isdir(class_dir_path):
        all_data_in_class = os.listdir(class_dir_path)

        train_set,val_test_set = sklearn.model_selection.train_test_split(all_data_in_class,test_size=0.2,shuffle=True)
        valid_set,test_set = sklearn.model_selection.train_test_split(val_test_set,test_size=0.5,shuffle=True)

        os.makedirs(os.path.join(train_dir_path,class_name),exist_ok=True)
        os.makedirs(os.path.join(valid_dir_path,class_name),exist_ok=True)
        os.makedirs(os.path.join(test_dir_path,class_name),exist_ok=True)
        
        copy_file(train_set,class_dir_path,os.path.join(train_dir_path,class_name))      
        copy_file(valid_set,class_dir_path,os.path.join(valid_dir_path,class_name))
        copy_file(test_set,class_dir_path,os.path.join(test_dir_path,class_name))

