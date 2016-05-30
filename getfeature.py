'''
Feature extraction module.
'''
import numpy as npy
import dsift
from sklearn.cluster import KMeans
import cv2


def train_bow(list_img_names,patch_size,num_patches,num_words,raw_feat_type):
    '''
    This function trains the bag-of-words model with a given list of image names.
    Input: 
        list_img_names: list of list, in which each sub-list corresponds to a class.
        num_patches:number of sampled patches
        num_words:vocabulary size of the trained BoW model
        raw_feat_type: raw feature type, 'sift','dsfit',...(other to be added)
    Return:
        bow_model: a numpy array with each row is a visual word
    '''
    #sampling patches
    num_class=len(list_img_names)
    num_img_each_class=npy.zeros(num_class)
    for cc in range(num_class):
        num_img_each_class[cc]=len(list_img_names[cc])
    
    feat_extrcator = dsift.SingleSiftExtractor(patch_size)
    num_patches_per_img=15    
    half_patch_size=(patch_size+1)/2
    if raw_feat_type=="dsift":
        raw_feat=npy.zeros((num_patches,128))
        idx_patch=0
        while (idx_patch<num_patches):
            #firstly, sampling the class and image
            idx_class_select=npy.int(npy.random.rand()*num_class)
            idx_img_select=npy.int(npy.random.rand()*num_img_each_class[idx_class_select])
            img=cv2.imread(list_img_names[idx_class_select][idx_img_select])
            if img.ndim == 3:
                img=npy.mean(img,axis=2)
            nrow,ncol=img.shape
            for ii in range(num_patches_per_img):
                pr=npy.int(npy.random.rand()*(nrow-patch_size-2))+half_patch_size
                pc=npy.int(npy.random.rand()*(ncol-patch_size-2))+half_patch_size
                img_patch=img[pr-half_patch_size:pr+half_patch_size,
                              pc-half_patch_size:pc+half_patch_size]
                raw_feat[idx_patch,:]=feat_extrcator.process_image(img_patch)
                idx_patch=idx_patch+1
                if (idx_patch==num_patches):
                    break
    
    obj_kmeans=KMeans(num_words,'k-means++',3,500,0.001)
    obj_kmeans.fit(raw_feat)
    bow_model=obj_kmeans.cluster_centers_
    return bow_model        
    
def get_hist(arr_src,data_range,num_bin,is_normalize=False):
    '''
    Compute the histogram of the input data array
    Input:
        arr_src:the array of source data
        data_range: the min and max value of the data
    Output:        
    '''
    if arr_src.size==0:
        hist=0
        return
        
    if data_range[0]>=data_range[1]:
        hist=arr_src.size
        if is_normalize==True:
            hist=1
        return hist
   
    hist=npy.zeros(num_bin)
    data=arr_src.flatten()
    idx=data<data_range[0]
    data[idx]=data_range[0]
    idx=data>data_range[1]
    data[idx]=data_range[1]
    data=npy.int16((num_bin-1)*(data-data_range[0])/(data_range[1]-data_range[0]))
    for item in data:
        hist[item]=hist[item]+1  
        
    if is_normalize==True:
        hist=hist/arr_src.size
        
    return hist
    


def get_bow_feat(img_names,bow_model,patch_size,raw_feat_type):
    '''
    This function extract BoW feature for each images in the list of image_names.
    Input:
        img_names:list-of-list in which each sub_list is the file name of images
                  that belong to the same class
        bow_model:the BoW model array, each row is a word
        raw_feat_type: raw feature type, 'sift','dsfit',...(other to be added)
    Output:
        bow_feat: the extracted BoW feature
    '''
    num_class=len(img_names)
    num_img_each_class=npy.zeros(num_class,dtype=npy.int16)
    for cc in range(num_class):
        num_img_each_class[cc]=len(img_names[cc])

    num_all_img=sum(num_img_each_class)
    num_words=bow_model.shape[0]
    feat_bow=npy.zeros((num_all_img,num_words+1))
    idx_feat=-1
    obj_kmeans=KMeans(num_words,'k-means++',3,500,0.001)
    obj_kmeans.cluster_centers_=bow_model
    if raw_feat_type=="dsift":
        raw_extractor = dsift.DsiftExtractor(8,patch_size,1)
        for cc in range(num_class):
            for idx_img in range(num_img_each_class[cc]):
                print(img_names[cc][idx_img])
                idx_feat=idx_feat+1
                img=cv2.imread(img_names[cc][idx_img])
                if img.ndim==3:
                    img=npy.mean(img,axis=2)
                feat_raw,feat_pos=raw_extractor.process_image(img,False,False)
                print "w:%d,h:%d,num_feat:%d" %(img.shape[0],img.shape[1],feat_raw.shape[0])
                label_feat=obj_kmeans.predict(feat_raw)
                hist=get_hist(label_feat,npy.array([0,num_words-1]),num_words,True)
                feat_bow[idx_feat,0]=cc
                feat_bow[idx_feat,1:]=hist
            
    return feat_bow    
    


if __name__=="__main__":
    pass
