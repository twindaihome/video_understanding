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
from sklearn.linear_model import LogisticRegression,SGDClassifier,BayesianRidge,LogisticRegressionCV,ElasticNetCV
from sklearn.svm import SVR,SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer

'''
iris = load_iris()
X = iris.data
y = iris.target

col_audio  = ColumnSelector(cols = range(2))
col_video = ColumnSelector(cols = (2,3))
col_face = ColumnSelector(cols = (2,2))
col_early = ColumnSelector(cols = range(4))

'''
def fusion(X,y,x_valid):
    print('lets start fusion')
    col_title = ColumnSelector(cols= (397, -1))
    col_audio = ColumnSelector(cols= (269, 397))
    col_video = ColumnSelector(cols= (141,269))
    col_face = ColumnSelector(cols = (13,141))
    col_meta = ColumnSelector(cols = (0,13))
    col_early = ColumnSelector(cols = (0,-1))

    clf = LinearSVC(tol=1e-200,max_iter=100000)
    clf_sgd = SGDClassifier(loss='log', max_iter=100000000, tol=1e-200)
    clf_rf = RandomForestClassifier(n_estimators=1000, max_depth=200, random_state=0)
    clf_lg = LogisticRegression(random_state=0, multi_class='multinomial',solver='saga')
    rbm = BernoulliRBM(random_state=0, verbose=True)
    pipe_title = Pipeline([('col', col_title), ('scale', StandardScaler()),('clf', clf_sgd)])
    pipe_audio = Pipeline([('col',col_audio),('scale',StandardScaler()),('clf',clf_sgd)])
    pipe_video = Pipeline([('col',col_video),('scale',StandardScaler()),('clf',clf_sgd)])
    pipe_face = Pipeline([('col',col_face),('scale',StandardScaler()),('clf',clf_sgd)])
    pipe_meta = Pipeline([('col',col_meta),('scale',StandardScaler()),('clf',clf_sgd)])
    pipe_early = Pipeline([('col',col_early),('scale',StandardScaler()),('clf',clf_sgd)])

    #### Computing baseline accuracies using single feature type each time and then using their early fusion. ####
    cross_val_accuracy_title = cross_val_score(pipe_title, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_audio = cross_val_score(pipe_audio, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_video  = cross_val_score(pipe_video, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_face  = cross_val_score(pipe_face, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_meta = cross_val_score(pipe_meta, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_early  = cross_val_score(pipe_early, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print('title_accuracy:{}, \naudio_accuracy:{}, video_accuracy:{},\n face_accuracy:{}, meta_accuracy:{}, \n early_fusion_accuracy:{}'.format(np.mean(cross_val_accuracy_title), np.mean(cross_val_accuracy_audio),np.mean(cross_val_accuracy_video),
          np.mean(cross_val_accuracy_face),np.mean(cross_val_accuracy_meta),np.mean(cross_val_accuracy_early)))

    #### Creating Weighted Majority Voting multiple kernel learning Classifiers without using the early fusion classfier. ####
    weights = [cross_val_accuracy_title.mean(), cross_val_accuracy_audio.mean(), cross_val_accuracy_video.mean(),cross_val_accuracy_meta.mean(), cross_val_accuracy_face.mean()]
    eclf1 = EnsembleVoteClassifier(clfs=[pipe_title, pipe_audio, pipe_video, pipe_meta, pipe_face],weights=weights, voting='hard')
    eclf2 = EnsembleVoteClassifier(clfs=[pipe_audio, pipe_video, pipe_meta, pipe_face], weights=weights[1:],
                                   voting='hard')
    eclf3 = EnsembleVoteClassifier(clfs=[pipe_audio, pipe_video, pipe_meta], weights=weights[1:4],
                                   voting='hard')
    cross_val_accuracy_late1  = cross_val_score(eclf1, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_late2 = cross_val_score(eclf2, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    cross_val_accuracy_late3 = cross_val_score(eclf3, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print('MKL_accuracy_1:{}, \n MKL_accuracy_2:{}, \nMKL_accuracy_3:{}'.format(np.mean(cross_val_accuracy_late1),np.mean(cross_val_accuracy_late2),np.mean(cross_val_accuracy_late3)))
    eclf2.fit(X, y)

    #### Extracting the predicted values from each classifier for the late fusion ####
    predict_title = cross_val_predict(pipe_title, X, y, cv=5, method="predict", n_jobs=-1)
    predict_audio = cross_val_predict(pipe_audio, X,y, cv=5, method="predict", n_jobs=-1)
    predict_video = cross_val_predict(pipe_video, X,y, cv=5, method="predict", n_jobs=-1)
    predict_face = cross_val_predict(pipe_face, X,y, cv=5, method="predict", n_jobs=-1)
    predict_meta = cross_val_predict(pipe_meta, X, y, cv=5, method="predict", n_jobs=-1)
    # print(predict_audio,predict_video)
    
    # fuse the predicted values
    all_predicted_labels = np.array([predict_title,predict_audio,predict_video,predict_face,predict_meta])
    # print(all_predicted_labels.T)
    late_fusion_data = all_predicted_labels.T * weights
    pipe_late = Pipeline([('scale', StandardScaler()), ('clf', clf_sgd)])
    cross_val_accuracy_late = cross_val_score(pipe_late, late_fusion_data, y, cv=5, scoring='accuracy', n_jobs=-1)
    print('late fusion', np.mean(cross_val_accuracy_late))

    #pipe_late.fit(late_fusion_data, y)
    #y_pred_late = pipe_late.predict(x_valid)

    return eclf2
