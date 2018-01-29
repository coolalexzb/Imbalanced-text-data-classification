# import os
#
# def RenameFiles(dir):
#     #将目录下所有的文件命名为数字开头的名称
#     folder_list = os.listdir(dir)
#
#     for folder in folder_list:
#         new_folder_path = os.path.join(dir,folder)
#         files = os.listdir(new_folder_path)
#         i = 0
#         for file in files :
#             dstname = folder + "_" +str(i)+".txt"
#             dstname = os.path.join(new_folder_path,dstname)
#             file = os.path.join(new_folder_path,file)
#             os.rename(file, dstname)
#             i += 1
# #RenameFiles("F:/STUDY_DATA/大创/dataset/dataset/")
# dir = "F:/STUDY_DATA/大创/dataset/dataset/"
# folder_list = os.listdir(dir)
# for folder in folder_list:
#     new_folder_path = os.path.join(dir,folder)
#     files = os.listdir(new_folder_path)
#     print(folder +"     " +str(len(files)))
#
#
#
# print(__doc__)

from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt

# # Load the digits dataset
# digits = load_digits()
# X = digits.images.reshape((len(digits.images), -1))
# y = digits.target
#
# # Create the RFE object and rank each pixel
# svc = SVC(kernel="linear", C=1)
# rfe = RFE(estimator=svc, n_features_to_select=1, step=1)
# rfe.fit(X, y)
# ranking = rfe.ranking_.reshape(digits.images[0].shape)
#
# # Plot pixel ranking
# plt.matshow(ranking, cmap=plt.cm.Blues)
# plt.colorbar()
# plt.title("Ranking of pixels with RFE")
# plt.show()

# from gensim.models import word2vec
# model = word2vec.Word2Vec.load("./word2vec_wx.model")