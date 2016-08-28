'''
This is a demo of image classification with different mid-level feature coding methods
The data set is FifteenScenes
The low-level feature is dsift
'''
import numpy as npy
import misc
import feat_coding
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.mixture import GMM


src_data_path="E:\DevProj\Datasets\FifteenScenes"
rec_path=""
class_labels,class_folders=misc.search_folder(src_data_path,set())
num_class=len(class_labels)

imgs_list=[]
class_size_list=[]
num_imgs=0
img_type_set=set(['.jpg','.bmp'])
for kk in range(num_class):
    imgs_list.extend(misc.search_folder(class_folders[kk],img_type_set)[1])
    class_size_list.append(len(imgs_list)-num_imgs)
    num_imgs=len(imgs_list)


#Split training and testing data
num_train_per_class=100 
train_img_list,train_label,test_img_list,\
test_label=misc.split_train_test(imgs_list,class_size_list,num_train_per_class,True)    

#Sampling random patches for training mid_level coding model
print("Sampling random patches")
patch_size=16
num_sampling_patches=60000
num_patch_per_img=80
sampling_feat=misc.sampling_rand_patch(train_img_list,patch_size,
                                       num_sampling_patches,num_patch_per_img)


#PCA transformation
print("PCA transfomation")
dim_pca=64
obj_pca=PCA(n_components=dim_pca)
obj_pca.fit(sampling_feat)
pca_feat=obj_pca.transform(sampling_feat)
del sampling_feat
#feature coding model
print("Training GMM model")
obj_gmm=GMM(n_components=180)
obj_gmm.fit(pca_feat)

#get test features
grid_spacing=8
print("Extracting FV feature for training images")
feat_train=feat_coding.get_fv_feat(train_img_list,grid_spacing,patch_size,obj_pca,obj_gmm)

#classification
print("Training SVM classifier")
svm_classifier=SVC(10,"linear")
svm_classifier.fit(feat_train,train_label)
del feat_train

predict_label=npy.zeros(0,dtype=npy.int32)
batch_size=100
num_batch=len(test_img_list)/batch_size
for kk in range(num_batch):
    img_batch=test_img_list[kk*batch_size:(kk+1)*batch_size]
    feat_test=feat_coding.get_fv_feat(img_batch,grid_spacing,patch_size,obj_pca,obj_gmm)
    predict_batch_label=svm_classifier.predict(feat_test)
    predict_label=npy.hstack((predict_label,predict_batch_label))
    conf_mat,oa,kappa=misc.eval_accuracy(test_label[0:(kk+1)*batch_size],
                                                    predict_label,range(num_class))
    print("".join(("Processing the",str(kk),"-th batch, overall accuracy=",str(oa))))

    

