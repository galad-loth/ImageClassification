'''
This is a simple implementation of the spatial pyramid matching (SPM) model for 
image/scene classification. 
Reference: S.Lazebnik1 et al., Beyond Bags of Features- Spatial Pyramid Matching
 for Recognizing Natural Scene Categories, in Proc CVPR 2006.   
 
'''
import os
import utils
from getfeature import train_bow,get_bow_feat
from sklearn.svm import SVC
from kernel_fun import ker_intersect

data_path="E:\DevProj\Datasets\FifteenScenes"
img_ext=[".jpg"]
raw_feat_type="dsift"

#get all file names
list_class_names=utils.search_subfolder(data_path)
list_img_names=[]
for class_name in list_class_names:
    img_names=utils.search_files(os.path.join(data_path,class_name),img_ext,True)
    list_img_names.append(img_names)  

#split training data
num_train_perclass=100
list_train_img,list_test_img=utils.split_data(list_img_names,num_train_perclass)
    
#train bag-of-words model
patch_size=16
num_sample_patches=6000
num_feat_words=200
bow_model=train_bow(list_train_img,patch_size,num_sample_patches,
                    num_feat_words,raw_feat_type)

# get training and testing features

feat_train=get_bow_feat(list_train_img,bow_model,patch_size,raw_feat_type)
feat_test=get_bow_feat(list_test_img,bow_model,patch_size,raw_feat_type)

gram_mat_train=ker_intersect(feat_train[:,1:],feat_train[:,1:])
gram_mat_test=ker_intersect(feat_test[:,1:],feat_train[:,1:])

#image classification with SVM
svm_classifier=SVC(10,"precomputed")
svm_classifier.fit(gram_mat_train,feat_train[:,0])
predict_label=svm_classifier.predict(gram_mat_test)

conf_mat,oa,kappa=utils.eval_accuracy(feat_test[:,0],predict_label,range(len(list_train_img)))

print conf_mat
print oa
