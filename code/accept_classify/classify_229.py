"""
train linear classifier using CV and find the best model on dev set
"""
import numpy as np
import sys, pickle,random
from collections import Counter
from sklearn import datasets,preprocessing,model_selection
from sklearn import linear_model,svm,neural_network,ensemble

def get_data(features_if, scale=False, n_features = None):
  data = datasets.load_svmlight_file(features_if, n_features=n_features)
  if scale:
    new_x = preprocessing.scale(data[0].toarray())
    return new_x, data[1]
  else:
    return data[0], data[1]


def main(args, scale=False):
    if len(args) < 5:
        print(("Usage:",args[0],"<train if> <dev if> <test if> <of>"))
        return -1

    ###########################
    # data loading
    ###########################
    n_features = sum(1 for line in open(args[5])) #None #train_features.shape[1]
    train_features, train_labels = get_data(args[1], scale=scale,n_features=n_features)
    dev_features, dev_labels = get_data(args[2], scale=scale, n_features=n_features)
    test_features, test_labels = get_data(args[3], scale=scale, n_features=n_features)


    ###########################
    # majority
    ###########################
    train_counter = Counter(train_labels)
    dev_counter = Counter(dev_labels)
    test_counter = Counter(test_labels)
    print("Model Train Test")
    print(("Majority {} {}".format(
      round(100.0*train_counter[0]/(train_counter[0]+train_counter[1]),3),
      round(100.0*test_counter[0]/(test_counter[0]+test_counter[1]),3))))


    ###########################
    #classifiers
    ###########################
    l2_logistic_clfs = []
    l1_logistic_clfs = []
    svm_rbf_clfs = []
    NN_clfs = []
    
    ensemble_clfs = []

    names = ["logistic_l2", "logistic_l1", "SVM_rbf", "random_forest", "AdaBoost", "neural_network" ]
    
    for c in [x * 0.1 for x in range(1, 10)]:
      l2_logistic_clfs.append(linear_model.LogisticRegression(C=c, dual=True))
      l1_logistic_clfs.append(linear_model.LogisticRegression(C=c, penalty='l1'))
      svm_rbf_clfs.append(svm.SVC(kernel='rbf', C=c, gamma = "scale"))
      for c1 in [x for x in range(1, 3)]:
      	NN_clfs.append(neural_network.MLPClassifier(hidden_layer_sizes= (int(100*c), c1), alpha=1))
    
    # NN_clfs.append(neural_network.MLPClassifier(hidden_layer_sizes= (100, 1), alpha=1))
    # NN_clfs.append(neural_network.MLPClassifier(hidden_layer_sizes= (100, 2), alpha=1))

    ensemble_clfs += [ensemble.RandomForestClassifier(), ensemble.AdaBoostClassifier()]


    ###########################
    # training (CV) and testing
    ###########################
    
    # Logistic L2 model
    for cidx, clf in enumerate(l2_logistic_clfs):
      best_v, best_classifier = select_best_model(clf, train_features, train_labels)
    record_result(best_v, best_classifier, train_features, train_labels, 
        dev_features, dev_labels, test_features, test_labels, names[0])

    # Logistic L1 model
    for cidx, clf in enumerate(l1_logistic_clfs):
      best_v, best_classifier = select_best_model(clf, train_features, train_labels)
    record_result(best_v, best_classifier, train_features, train_labels, 
        dev_features, dev_labels, test_features, test_labels, names[1])

    # SVM RBF model
    for cidx, clf in enumerate(svm_rbf_clfs):
      best_v, best_classifier = select_best_model(clf, train_features, train_labels)
    record_result(best_v, best_classifier, train_features, train_labels, 
        dev_features, dev_labels, test_features, test_labels, names[2])

    # Ensemble model
    i = 3
    for cidx, clf in enumerate(ensemble_clfs):
      scores = model_selection.cross_val_score(clf, train_features, train_labels, cv=5, n_jobs=8)
      v = sum(scores)*1.0/len(scores)
      record_result(v, clf, train_features, train_labels, dev_features, dev_labels, test_features, test_labels, names[i])
      i += 1

	# Neural network model
    for cidx, clf in enumerate(NN_clfs):
    	best_v, best_classifier = select_best_model(clf, train_features, train_labels)
    record_result(best_v, best_classifier, train_features, train_labels, 
    	dev_features, dev_labels, test_features, test_labels, names[5])
    
def select_best_model(clf, train_features, train_labels):
    best_classifier = None
    best_v = 0
    scores = model_selection.cross_val_score(clf, train_features, train_labels, cv=5, n_jobs=8)
    v = sum(scores)*1.0/len(scores)
    if v > best_v:
      #print("New best v!",v*100.0,clf)
      best_classifier = clf
      best_v = v
    return best_v, best_classifier


def record_result(best_v, best_classifier, train_features, train_labels, dev_features, 
                dev_labels, test_features, test_labels, model_name):

    best_classifier.fit(train_features, train_labels)

    # train
    train_y_hat = best_classifier.predict(train_features)
    train_score = 100.0 * sum(train_labels == train_y_hat) / len(train_y_hat)

    # test
    test_y_hat = best_classifier.predict(test_features)
    test_score = 100.0 * sum(test_labels == test_y_hat) / len(test_y_hat)
    
    print(model_name, "%.2f" %round(train_score,3), "%.2f" %round(test_score,3))
    

if __name__ == "__main__": sys.exit(main(sys.argv))
