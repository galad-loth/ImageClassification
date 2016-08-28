'''
Various kernel functions for SVM
'''

import numpy as npy

def ker_intersect(feat1, feat2):
    '''
    Intersection kernel.
    Input:
        feat1, feat2: two normalized hitogram feature
    Output:
        gram_mat:The gram matrix
    '''
    num_feat1,dim1=feat1.shape
    num_feat2,dim2=feat2.shape
    if dim1!=dim2:
        return npy.empty()
    
    feat1.astype(npy.double)
    feat2.astype(npy.double)
    gram_mat=npy.zeros((num_feat1,num_feat2),dtype=npy.double)
    for ii in range(num_feat1):
        for jj in range(num_feat2):
            idx=feat1[ii,:]>feat2[jj,:]
            temp=npy.sum(feat1[ii,~idx])+npy.sum(feat2[jj,idx])
            gram_mat[ii,jj]=temp
            
    return gram_mat


def ker_intersect_sparse(feat1, feat2):
    '''
    Intersection kernel for sparsely store feature.
    Input:
        feat1, feat2: two normalized hitogram feature
    Output:
        gram_mat:The gram matrix
    '''
    num_feat1,dim1=feat1.shape
    num_feat2,dim2=feat2.shape
    if dim1!=dim2:
        return npy.empty()
    
    gram_mat=npy.zeros((num_feat1,num_feat2),dtype=npy.double)
    for ii in range(num_feat1):
        dense_row1=feat1.getrow(ii).toarray()
        for jj in range(num_feat2):
            dense_row2=feat2.getrow(jj).toarray()
            idx=dense_row1>dense_row2
            temp=npy.sum(dense_row1[~idx])+npy.sum(dense_row2[idx])
            gram_mat[ii,jj]=temp
            
    return gram_mat	
	
def ker_rbf(feat1,feat2):
    pass
    
    
if __name__=="__main__":
    feat1=npy.random.randn(5,100)
    feat2=npy.random.randn(8,100)
    g=ker_intersect(feat1,feat1)
            
    
