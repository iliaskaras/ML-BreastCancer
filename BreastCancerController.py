from BreastCancerDecisionTreeClf import BreastCancerDecisionTreeClf
from GraphvizCreator import GraphvizCreator

#call maxDepth = 20 ================================================

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("gini",20)
graphvizCreator = GraphvizCreator(breastCancerDecisionTree)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.exportGraph()

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("entropy",20)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.changeDecisionTreeClfObject(breastCancerDecisionTree)
graphvizCreator.exportGraph()


#call maxDepth = 12 ================================================

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("gini",12)
graphvizCreator = GraphvizCreator(breastCancerDecisionTree)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.exportGraph()

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("entropy",12)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.changeDecisionTreeClfObject(breastCancerDecisionTree)
graphvizCreator.exportGraph()


#call maxDepth = 8 ================================================

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("gini",8)
graphvizCreator = GraphvizCreator(breastCancerDecisionTree)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.exportGraph()

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("entropy",8)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.changeDecisionTreeClfObject(breastCancerDecisionTree)
graphvizCreator.exportGraph()


#call maxDepth = 5 ================================================

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("gini",5)
graphvizCreator = GraphvizCreator(breastCancerDecisionTree)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.exportGraph()

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("entropy",5)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.changeDecisionTreeClfObject(breastCancerDecisionTree)
graphvizCreator.exportGraph()

#call maxDepth = 2 ================================================

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("gini",2)
graphvizCreator = GraphvizCreator(breastCancerDecisionTree)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.exportGraph()

breastCancerDecisionTree = None

breastCancerDecisionTree = BreastCancerDecisionTreeClf()
breastCancerDecisionTree.createDecisionTreeClassifier("entropy",2)
breastCancerDecisionTree.getCrossValidation()
graphvizCreator.changeDecisionTreeClfObject(breastCancerDecisionTree)
graphvizCreator.exportGraph()


