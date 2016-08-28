'''
Miscellaneous un-classified functions
'''
import os
import numpy as npy
import time
import cv2
import dsift
from random import sample as rndsample


def search_folder(parent_folder,input_ext_set):
    '''
    Search subfolders and files in the given folder
    If file_ext is empty string, then returns a list of subfolders,
    otherwise returns a list of files with given extension
    '''
    files_list=[]
    files_path_list=[]
    for file_name in os.listdir(parent_folder):
        file_path=os.path.join(parent_folder,file_name)
        file_ext=os.path.splitext(file_name)[1]
        if 0==len(input_ext_set) and os.path.isdir(file_path):
            files_path_list.append(file_path)
            files_list.append(file_name)
        elif (file_ext in input_ext_set) and os.path.isfile(file_path):
            files_path_list.append(file_path)
            files_list.append(file_name)
            
    return files_list,files_path_list
    
    
def sampling_rand_patch(img_list,patch_size,num_sampling_patches,num_patches_per_img):
    num_patches=0
    npy.random.seed(npy.int64(time.time()))
    num_imgs=len(img_list)
    visit_flag=npy.zeros(num_imgs,dtype=npy.int8)
    feature_extractor=dsift.SingleSiftExtractor(patch_size)
    sampling_feat=npy.zeros((num_sampling_patches,128),dtype=npy.float32)
    while num_patches<num_sampling_patches:
        idx_img=npy.random.randint(num_imgs)
        if 1==visit_flag[idx_img]:
            continue
        visit_flag[idx_img]=1
        img=cv2.imread(img_list[idx_img])
        if img.ndim==3:
            img=npy.mean(img,axis=2)
        nr,nc=img.shape
        num_patches_this_img=0
        while num_patches_this_img<num_patches_per_img:        
            pr0=npy.random.randint(nr-patch_size)
            pc0=npy.random.randint(nc-patch_size)
            patch=img[pr0:pr0+patch_size,pc0:pc0+patch_size]
            sampling_feat[num_patches,:]=feature_extractor.process_image(patch)
            num_patches_this_img=num_patches_this_img+1
            num_patches=num_patches+1
            if num_patches>=num_sampling_patches:
                break
            
    return sampling_feat
        

def split_train_test(sample_list,class_size_list,num_train_per_class,is_random): 
    
    train_sample_list=[]
    test_sample_list=[]
    train_label=npy.zeros(0,dtype=npy.int32)
    test_label=npy.zeros(0,dtype=npy.int32)
    num_class=len(class_size_list)
    if is_random:
        npy.random.seed(npy.int64(time.time()))
        idx_class_start=0
        for kk in range(num_class):
            if class_size_list[kk]<=num_train_per_class:
                train_sample_list.extend(sample_list[idx_class_start:\
                idx_class_start+class_size_list[kk]])
                train_label=npy.hstack((train_label,
                                        kk*npy.ones(class_size_list[kk],dtype=npy.int32)))
            else:
                flag_select=npy.zeros(class_size_list[kk],dtype=npy.int8)
                idx_select=rndsample(xrange(0,class_size_list[kk]),num_train_per_class)
                flag_select[idx_select]=1
                for ll in range(class_size_list[kk]):
                    if flag_select[ll]==0:
                        test_sample_list.append(sample_list[idx_class_start+ll])
                    else:
                        train_sample_list.append(sample_list[idx_class_start+ll])
                train_label=npy.hstack((train_label,
                                        kk*npy.ones(num_train_per_class,dtype=npy.int32)))
                test_label=npy.hstack((test_label,
                                       kk*npy.ones(class_size_list[kk]-num_train_per_class,
                                                dtype=npy.int32)))
            idx_class_start=idx_class_start+class_size_list[kk]              
    else:        
        idx_class_start=0
        for kk in range(num_class):
            if class_size_list[kk]<=num_train_per_class:
                train_sample_list.extend(sample_list[idx_class_start:\
                idx_class_start+class_size_list[kk]])
                train_label=npy.hstack((train_label,
                                        kk*npy.ones(class_size_list[kk],dtype=npy.int32)))
            else:
                train_sample_list.extend(sample_list[idx_class_start:\
                idx_class_start+num_train_per_class])
                train_label=npy.hstack((train_label,
                                        kk*npy.ones(num_train_per_class,dtype=npy.int32)))
                test_sample_list.extend(sample_list[idx_class_start+num_train_per_class:\
                idx_class_start+class_size_list[kk]])
                test_label=npy.hstack((test_label,
                                       kk*npy.ones(class_size_list[kk]-num_train_per_class,
                                                dtype=npy.int32)))
            idx_class_start=idx_class_start+class_size_list[kk]
    
    return train_sample_list,train_label,test_sample_list,test_label
        
    
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
        
        
