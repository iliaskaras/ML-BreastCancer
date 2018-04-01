from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier


class RandomForestClf(object):

    seed = 42

    # The class "constructor" - It's actually an initializer
    def __init__(self, seed=seed):
        self.seed = seed
        self.clf = None
        self.randomForestDs()
        self.name = "RandomForest_seed_"+str(self.seed)

    # import breast cancer from sklearn - 569 samples / 30 Dimensionality
    def randomForestDs(self):
        self.randomForest = load_breast_cancer()
        self.x = self.randomForest.data
        self.y = self.randomForest.target

    def randomForestClassifier(self,n_estimators,split_function):
        self.clf = RandomForestClassifier(n_estimators=n_estimators,
                                          criterion=split_function,
                                          random_state=self.seed)
        self.n_estimators = n_estimators
        self.split_function = split_function
        print("Number of trees :"+str(n_estimators)+", criterion: "+split_function)

    def getCrossValidation(self):
        if self.clf is not None:
            crossValResults = cross_val_score(self.clf, self.x, self.y, cv=5)
            average = self.getAverage(crossValResults)
            self.getScores()
            print(" =======================================================")
            print(crossValResults)
            print(" Final Average of the 5 training sets : " + str(average))
            print("\n")

        else:
            print("                          (Warning)                                    \n"
                  "You must first create the Classifier and then get the cross vailidation!\n"
                  "You can do that by simple call randomForestClassifier method.\n")

    def getAverage(self,list):
        sum = 0
        for num in range(0, 5):
            sum += list[num]
        return sum/5

    def getScores(self):

        X_train, X_test, y_train, y_test = \
            train_test_split(self.x,  self.y, test_size=0.20, random_state=1)

        pipe_svc = Pipeline([('scl', StandardScaler()),
                             ('clf', SVC(random_state=1))])
        pipe_svc.fit(X_train, y_train)

        fittedClf = self.clf.fit(self.x,
                                self.y)
        y_pred = fittedClf.predict(X_test)

        print("\nScores : ")
        print("Precision: %.3f" % precision_score(y_true=y_test, y_pred=y_pred))
        print("Recall: %.3f" % recall_score(y_true=y_test, y_pred=y_pred))
        print("F1: %.3f" % f1_score(y_true=y_test, y_pred=y_pred))
        return None

    def get_featureNames(self):
        return self.randomForest.feature_names

    def get_targetNames(self):
        return self.randomForest.target_names

    def get_clf(self):
        return self.clf

    def get_name(self):
        return self.name

    def get_maxDepth(self):
        return self.n_estimators

    def get_splitFunction(self):
        return self.split_function
