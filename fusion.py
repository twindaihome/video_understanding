from sklearn.model_selection import KFold, cross_val_score
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.datasets import load_iris
from mlxtend.feature_selection import ColumnSelector
from sklearn.svm import SVR,SVC, LinearSVC
from sklearn.model_selection import cross_val_predict
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from mlxtend.classifier import EnsembleVoteClassifier
'''
iris = load_iris()
X = iris.data
y = iris.target

col_audio  = ColumnSelector(cols = range(2))
col_video = ColumnSelector(cols = (2,3))
col_face = ColumnSelector(cols = (2,2))
col_early = ColumnSelector(cols = range(4))

'''
def fusion(X,y):
    print('lets start fusion')
    col_audio = ColumnSelector(cols= (269, -1))
    col_video = ColumnSelector(cols= (141,269))
    col_face = ColumnSelector(cols = (13,141))
    col_meta = ColumnSelector(cols = (0,13))
    col_early = ColumnSelector(cols = (0,-1))

    clf = LinearSVC(tol=1e-200,max_iter=100000)
    pipe_audio = Pipeline([('col',col_audio),('scale',StandardScaler()),('clf',clf)])
    pipe_video = Pipeline([('col',col_video),('scale',StandardScaler()),('clf',clf)])
    pipe_face = Pipeline([('col',col_face),('scale',StandardScaler()),('clf',clf)])
    pipe_meta = Pipeline([('col',col_meta),('scale',StandardScaler()),('clf',clf)])
    pipe_early = Pipeline([('col',col_early),('scale',StandardScaler()),('clf',clf)])

    #### Computing baseline accuracies using single feature type each time and then using their early fusion. ####
    cross_val_accuracy_audio = cross_val_score(pipe_audio, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_video  = cross_val_score(pipe_video, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_face  = cross_val_score(pipe_face, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_meta = cross_val_score(pipe_meta, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_early  = cross_val_score(pipe_early, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print('audio_accuracy:{}, video_accuracy:{},\n face_accuracy:{}, meta_accuracy:{}, \n early_fusion_accuracy:{}'.format(np.mean(cross_val_accuracy_audio),np.mean(cross_val_accuracy_video),
          np.mean(cross_val_accuracy_face),np.mean(cross_val_accuracy_meta),np.mean(cross_val_accuracy_early)))

    #### Creating Weighted Majority Voting Classifiers without using the early fusion classfier. ####
    weights = [cross_val_accuracy_audio.mean(), cross_val_accuracy_video.mean(), cross_val_accuracy_face.mean(),cross_val_accuracy_meta.mean()]
    eclf = EnsembleVoteClassifier(clfs=[pipe_audio, pipe_video,pipe_face,pipe_meta],weights=weights, voting='hard')
    cross_val_accuracy_late  = cross_val_score(eclf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print('late_ensemble_accuracy:{}'.format(np.mean(cross_val_accuracy_late)))
    eclf.fit(X, y)
    return eclf


'''
#### Extracting the predicted values from each classifier for the late fusion ####
predict_audio = cross_val_predict(pipe_audio, X,y, cv=5, method="predict", n_jobs=-1)
predict_video = cross_val_predict(pipe_video, X,y, cv=5, method="predict", n_jobs=-1)
predict_face = cross_val_predict(pipe_face, X,y, cv=5, method="predict", n_jobs=-1)
# print(predict_audio,predict_video)

# fuse the predicted values
all_predicted_labels = np.array([predict_audio,predict_video,predict_face])
print(all_predicted_labels)

#### Computing final predictions with late fusion without training. ####
#### You can use different combinations of the classifiers to be fused. ####
# sum on axis = o, that is, sum the predicted labels for each user, and return the index of the max predicted value
results11 = all_predicted_labels.sum(0).argmax(0)
# do the production on axis = o, that is ,product the predicted labels for each user, and return the maximum index
results12 = all_predicted_labels.prod(0).argmax(0)
# take the median of them, along axis = 0, return the index of the maximum value
results13 = np.median(all_predicted_labels, 0).argmax(0)
# take te maximum value, and then return the index of it
results14 = np.max(all_predicted_labels, 0).argmax(0)

'''

