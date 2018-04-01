from RandomForestClf import RandomForestClf

#call maxDepth = 20 ================================================

randomForest = RandomForestClf()
randomForest.randomForestClassifier(20,"gini")
randomForest.getCrossValidation()

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(20,"entropy")
randomForest.getCrossValidation()


#call maxDepth = 12 ================================================

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(15,"gini")
randomForest.getCrossValidation()

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(15,"entropy")
randomForest.getCrossValidation()


#call maxDepth = 8 ================================================

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(10,"gini")
randomForest.getCrossValidation()

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(10,"entropy")
randomForest.getCrossValidation()


#call maxDepth = 5 ================================================

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(5,"gini")
randomForest.getCrossValidation()

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(5,"entropy")
randomForest.getCrossValidation()

#call maxDepth = 2 ================================================

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(2,"gini")
randomForest.getCrossValidation()

randomForest = None

randomForest = RandomForestClf()
randomForest.randomForestClassifier(2,"entropy")
randomForest.getCrossValidation()


