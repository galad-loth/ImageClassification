'''
Simple util functions. Mianly file/subfolder searching, etc.
'''
import os
import random
import numpy as npy

def search_subfolder(search_path):
    subfolder_list=[]
    for name in os.listdir(search_path):
        joined_path=os.path.join(search_path,name)
        if os.path.isdir(joined_path):
            subfolder_list.append(name)
    return subfolder_list


def search_files(search_path,file_ext,with_path):
    if type(file_ext)!=set:
        file_ext=set(file_ext)    
    filename_list=[]
    for name in os.listdir(search_path):
        temp_path,temp_ext=os.path.splitext(name)
        if temp_ext in file_ext:
            if with_path:
                filename_list.append(os.path.join(search_path,name))
            else:
                filename_list.append(name)
    return filename_list    
      
    
def split_data(list_file_names, num_train_per_class):
    num_class=len(list_file_names)
    list_files_train=[]
    list_files_test=[]
    for kk in range(num_class):
        list_file_names_kk=list_file_names[kk]
        if len(list_file_names_kk)<=num_train_per_class:
            list_files_train.append(list_file_names_kk)
            list_files_test.append([])
        else:
            random.shuffle(list_file_names_kk)
            list_files_train.append(list_file_names_kk[:num_train_per_class])
            list_files_test.append(list_file_names_kk[num_train_per_class:])
    return (list_files_train,list_files_test)
    
    
    
def split_feat(list_img_names, num_train):
    pass
    
def eval_accuracy(eval_label,gt_label,label_set):
    num_class=len(label_set)
    conf_mat=npy.zeros((num_class,num_class),dtype=npy.int32)
    vec_ones=npy.ones(len(gt_label))
    for ii in label_set:
        for jj in label_set:
            conf_mat[ii,jj]=npy.sum(vec_ones[npy.logical_and(eval_label==ii, gt_label==jj)])
    
    oa=npy.float32(npy.sum(npy.diagonal(conf_mat)))/ npy.sum(conf_mat) 
    kappa=1
    return (conf_mat,oa,kappa)
    


if __name__=="__main__":
    pass
