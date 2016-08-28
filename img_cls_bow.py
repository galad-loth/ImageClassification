'''
This is a demo of image classification with different mid-level feature coding methods
The data set is FifteenScenes
The low-level feature is dsift
'''
import numpy as npy
import misc
import feat_coding
from sklearn.svm import SVC
from kernel_fun import ker_intersect


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
num_sampling_patches=6000
num_patch_per_img=15
sampling_feat=misc.sampling_rand_patch(train_img_list,patch_size,
                                       num_sampling_patches,num_patch_per_img)

#feature coding model
print("Training BoW model")
num_words=256
bow_model=feat_coding.train_bow_model(sampling_feat,num_words)

#get test features
grid_spacing=8
print("Extracting BoW feature for training images")
feat_train=feat_coding.get_bow_feat(train_img_list,grid_spacing,patch_size,bow_model)

print("Compute Kernel for SVM")
gram_mat_train=ker_intersect(feat_train,feat_train)

#classification
print("Training SVM classifier")
svm_classifier=SVC(10,"precomputed")
svm_classifier.fit(gram_mat_train,train_label)

predict_label=npy.zeros(0,dtype=npy.int32)
batch_size=100
num_batch=len(test_img_list)/batch_size
for kk in range(num_batch):
    img_batch=test_img_list[kk*batch_size:(kk+1)*batch_size]
    feat_test=feat_coding.get_bow_feat(img_batch,grid_spacing,patch_size,bow_model)
    gram_mat_test=ker_intersect(feat_test,feat_train)
    predict_batch_label=svm_classifier.predict(gram_mat_test)
    predict_label=npy.hstack((predict_label,predict_batch_label))
    conf_mat,oa,kappa=misc.eval_accuracy(test_label[0:(kk+1)*batch_size],
                                                    predict_label,range(num_class))
    print("".join(("Processing the",str(kk),"-th batch, overall accuracy=",str(oa))))

    

