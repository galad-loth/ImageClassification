'''
Functions for low-level feature coding
'''

import numpy as npy
from sklearn.cluster import KMeans
import dsift
from cv2 import imread
#from sklearn.mixture import GMM

def train_bow_model(raw_feat,num_words):
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
      

def get_bow_feat(img_list,grid_spacing,patch_size,bow_model):
    raw_feat_extractor=dsift.DsiftExtractor(grid_spacing,patch_size,1)
    num_words=bow_model.shape[0]
    obj_kmeans=KMeans(num_words,'k-means++',3,500,0.001)
    obj_kmeans.cluster_centers_=bow_model
    bow_feat=npy.zeros((len(img_list),num_words),dtype=npy.float32)
    for kk in range(len(img_list)):
        img=imread(img_list[kk])
        if img.ndim==3:
            img=npy.mean(img,axis=2)
        raw_feat=raw_feat_extractor.process_image(img,False,False)[0]
        label_feat=obj_kmeans.predict(raw_feat)
        bow_feat[kk,:]=get_hist(label_feat,npy.array([0,num_words-1]),num_words,True)
    return bow_feat
  
def get_spm_hist(label,pos,num_words,pyramid_level,img_shape):
    '''
    Compute the spatial pyramid matching feature
    '''
    dim_spm=num_words*(4**(pyramid_level+1)-1)/3;
    hist=npy.zeros(dim_spm,dtype=npy.float)
    label.astype(npy.int32) 
    #Get the spatial index in each pyramid level
    num_label=label.shape[0]
    cell_start=npy.zeros([1,num_label],dtype=npy.int32)
    idx_level_start=num_words
    for ii in range(1,pyramid_level+1):
        cell_size1=img_shape[0]/(2**ii)
        cell_size2=img_shape[1]/(2**ii)
        idxr=pos[0,:]/cell_size1
        idxc=pos[1,:]/cell_size2
        idx_cell=idxr*(2**ii)+idxc
        idx_cell=idx_level_start+num_words*idx_cell
        cell_start=npy.vstack((cell_start,idx_cell))
        idx_level_start=idx_level_start+num_words*(4**ii)
    
    for kk in range(label.shape[0]):
        label_now=label[kk]
        hist[label_now]=hist[label_now]+0.5
        for ii in range(1,pyramid_level+1):
            idx_hist=cell_start[ii][kk]+label_now
            hist[idx_hist]=hist[idx_hist]+1.0/(2**(pyramid_level+1-ii));    
    #normalize histogram
    hist=hist/npy.sum(hist)
    
#    idx_nonzero=npy.nonzero(hist)
#    data=hist[idx_nonzero]
#    return (data,idx_nonzero[0])
    return hist

  
def get_spm_feat(img_list,grid_spacing,patch_size,bow_model,pyramid_level):
    raw_feat_extractor=dsift.DsiftExtractor(grid_spacing,patch_size,1)
    num_words=bow_model.shape[0]
    dim_spm=num_words*(4**(pyramid_level+1)-1)/3
    obj_kmeans=KMeans(num_words,'k-means++',3,500,0.001)
    obj_kmeans.cluster_centers_=bow_model
    spm_feat=npy.zeros((len(img_list),dim_spm),dtype=npy.float32)
    for kk in range(len(img_list)):
        img=imread(img_list[kk])
        if img.ndim==3:
            img=npy.mean(img,axis=2)
        raw_feat,pos_feat=raw_feat_extractor.process_image(img,False,False)
        label_feat=obj_kmeans.predict(raw_feat)
        spm_feat[kk,:]=get_spm_hist(label_feat,pos_feat,num_words,pyramid_level,img.shape)
    return spm_feat   

def fv_coding(raw_feat,obj_gmm,dim_fv):
    post_prob=obj_gmm.predict_proba(raw_feat)    
    num_feat,dim_feat=raw_feat.shape
    num_gmm_comp=obj_gmm.n_components
    fv_u_kk=npy.zeros((num_gmm_comp,dim_feat),dtype=npy.float32)
    fv_v_kk=npy.zeros((num_gmm_comp,dim_feat),dtype=npy.float32)
    eps_float32=npy.finfo(npy.float32).eps
    for kk in range(num_feat):
        feat_kk=raw_feat[kk,:]
        post_prob_kk=post_prob[kk,:]
        raw_feat_rep=npy.repeat(feat_kk[npy.newaxis,:],num_gmm_comp,axis=0)
        raw_feat_rep=raw_feat_rep-obj_gmm.means_
        raw_feat_rep=raw_feat_rep/(obj_gmm.covars_+eps_float32)
        post_prob_rep=npy.repeat(post_prob_kk[:,npy.newaxis],dim_feat,axis=1)
        fv_u_kk=fv_u_kk+post_prob_rep*raw_feat_rep
        fv_v_kk=fv_v_kk+post_prob_rep*(raw_feat_rep*raw_feat_rep-1)
    gmm_weights=obj_gmm.weights_
    gmm_weights=npy.repeat(gmm_weights[:,npy.newaxis],dim_feat,axis=1)
    gmm_weights=1/npy.sqrt(gmm_weights+eps_float32)
    fv_u_kk=fv_u_kk*gmm_weights/num_feat
    fv_v_kk=fv_v_kk*gmm_weights/num_feat/npy.sqrt(2)
    fv_un=npy.hstack((fv_u_kk.flatten(),fv_v_kk.flatten()))
    fv_ssr=npy.sqrt(npy.abs(fv_un))
    fv=fv_ssr
    fv[fv_un<0]=-fv_ssr[fv_un<0]
    fv=fv/(npy.sqrt(npy.sum(fv*fv)+eps_float32))
    return fv
        
    

def get_fv_feat(img_list,grid_spacing,patch_size,obj_pca,obj_gmm):
    raw_feat_extractor=dsift.DsiftExtractor(grid_spacing,patch_size,1)
    dim_pca=obj_pca.n_components
    num_gmm_comp=obj_gmm.n_components
    dim_fv=2*dim_pca*num_gmm_comp
    fv_feat=npy.zeros((len(img_list),dim_fv),dtype=npy.float32)
    for kk in range(len(img_list)):
        print("Extracting FV feature,"+str(kk)+"/"+str(len(img_list)))
        img=imread(img_list[kk])
        if img.ndim==3:
            img=npy.mean(img,axis=2)
        raw_feat,pos_feat=raw_feat_extractor.process_image(img,False,False)
        pca_feat=obj_pca.transform(raw_feat)        
        fv_feat[kk,:]=fv_coding(pca_feat,obj_gmm,dim_fv)
    return fv_feat
    
    
def get_vlad_feat(img_list,grid_spacing,patch_size,bow_model):
    raw_feat_extractor=dsift.DsiftExtractor(grid_spacing,patch_size,1)    
    num_words,dim_feat=bow_model.shape
    dim_vlad=num_words*dim_feat
    vlad_feat=npy.zeros((len(img_list),dim_vlad),dtype=npy.float32)
    obj_kmeans=KMeans(num_words,'k-means++',3,500,0.001)
    obj_kmeans.cluster_centers_=bow_model
    eps_float32=npy.finfo(npy.float32).eps
    for kk in range(len(img_list)):
        print("Extracting VLAD feature,"+str(kk)+"/"+str(len(img_list)))
        img=imread(img_list[kk])
        if img.ndim==3:
            img=npy.mean(img,axis=2)
        raw_feat,pos_feat=raw_feat_extractor.process_image(img,False,False)
        label_feat=obj_kmeans.predict(raw_feat)
        vlad_feat_kk=npy.zeros(dim_vlad,dtype=npy.float32)
        for ii in range(label_feat.shape[0]):
            label_ii=label_feat[ii]
            res_ii=raw_feat[ii,:]-bow_model[label_ii,:]
            res_ii_norm=npy.sqrt(npy.sum(res_ii*res_ii))
            res_ii=res_ii/(res_ii_norm+eps_float32)
            res_ii=res_ii+vlad_feat_kk[label_ii*dim_feat:(label_ii+1)*dim_feat]
            vlad_feat_kk[label_ii*dim_feat:(label_ii+1)*dim_feat]=res_ii
        vlad_feat_kk_ssr=npy.sqrt(npy.abs(vlad_feat_kk))
        idx_temp=vlad_feat_kk>0
        vlad_feat_kk[idx_temp]=vlad_feat_kk_ssr[idx_temp]
        idx_temp=npy.logical_not(idx_temp)
        vlad_feat_kk[idx_temp]=-vlad_feat_kk_ssr[idx_temp]
        vlad_feat[kk,:]=vlad_feat_kk/(npy.sqrt(npy.sum(vlad_feat_kk*vlad_feat_kk)+eps_float32))
    return vlad_feat
            
        
        
        
        
        
        
        
        
        
    
    
