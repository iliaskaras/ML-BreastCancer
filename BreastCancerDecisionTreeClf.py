from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

# always inherit from object in 2.x. it's called new-style classes.
class BreastCancerDecisionTreeClf(object):

    seed = 42

    # The class "constructor" - It's actually an initializer
    def __init__(self,seed=seed):
        self.seed = seed
        self.clf = None
        self.loadBreastCancerDs()
        self.name = "BreastCancer_seed_"+str(self.seed)

    # import breast cancer from sklearn - 569 samples / 30 Dimensionality
    def loadBreastCancerDs(self):
        self.breastCancer = load_breast_cancer()
        self.x = self.breastCancer.data
        self.y = self.breastCancer.target

    def createDecisionTreeClassifier(self,split_function,maxdepth):
        self.clf = DecisionTreeClassifier(criterion = split_function,
                                          max_depth=maxdepth,
                                          random_state=self.seed)
        self.split_function = split_function
        self.max_depth = maxdepth
        print("criterion: "+split_function+", max depth: "+str(maxdepth))

    # import breast cancer from sklearn - 569 samples / 30 Dimensionality
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
                  "You can do that by simple call createDecisionTreeClassifier method.\n")

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
        return self.breastCancer.feature_names

    def get_targetNames(self):
        return self.breastCancer.target_names

    def get_clf(self):
        return self.clf

    def get_name(self):
        return self.name

    def get_maxDepth(self):
        return self.max_depth

    def get_splitFunction(self):
        return self.split_function
