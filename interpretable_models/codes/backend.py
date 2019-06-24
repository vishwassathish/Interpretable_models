import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn import linear_model
from lime import lime_tabular
from sklearn.metrics import confusion_matrix, \
    average_precision_score, precision_recall_curve, \
    accuracy_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')


class LogRegression:

    def __init__(self, x_train, y_train, x_validate, y_validate):
        
        self.x_train = x_train 
        self.y_train = y_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.model = linear_model.LogisticRegression(max_iter=1500)

    def train_and_test(self):
        
        print(self.x_train.shape, self.y_train.shape, self.x_validate.shape, self.y_validate.shape)
        self.model.fit(self.x_train, self.y_train)
        print("Weights/coefficients : ", self.model.coef_, "\nBias/intercept : ", self.model.intercept_)
        self.prediction = self.model.predict(self.x_validate)
        print("Accuracy is : ",accuracy_score(self.y_validate, self.prediction))
        return self

    def print_confusion_matrix(self):
        
        print("Confusion Matrix :-")
        confusionMatrix = confusion_matrix(self.y_validate, self.prediction)
        for row in confusionMatrix:
            for ele in row:
                print(ele,"\t", end="")
            print()
        return self

    def plot_precision_recall_curve(self):
        
        y_score = self.model.decision_function(self.x_validate)
        average_precision = average_precision_score(self.y_validate, y_score)
        print("Average_precision score is : ",average_precision,"\n")

        precision, recall, _ = precision_recall_curve(self.y_validate, y_score)
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
        print("Saving precision recall curve ....")
        plt.savefig("../graphs/precision_recall.jpg")
        plt.clf()
        print("Graph saved in 'graphs' folder!")
        return self

    def plot_roc_curve(self):
        
        probs = self.model.predict_proba(self.x_validate)
        preds = probs[:,1]
        fpr, tpr, threshold = roc_curve(self.y_validate, preds)
        roc_auc = auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        print("Saving ROC curve ....")
        plt.savefig("../graphs/roc.jpg")
        plt.clf()
        print("Graph saved in 'graphs' folder!")
        return self

    def save_model(self):

        print("Saving the model ...")
        with open("../data/model.pickle", "wb") as f:
            pickle.dump(self.model, f)
        print("Model Saved as 'model.pickle' !")
        return self


class ExplModel:

    def __init__(self, model, x_train, x_validate, y_validate, featureNames):
        self.model = model
        self.x_train = x_train
        self.x_validate = x_validate
        self.y_validate = y_validate
        self.featureNames = featureNames

    def weight_analysis(self):
        print("Performing feature weight analysis ...")
        attr_weight_list = []
        for i in range(len(self.model.coef_[0])):
            attr_weight_list.append([self.model.coef_[0][i], self.  featureNames[i]])
        attr_weight_list.sort()
        #print(attr_weight_list[:50])
        _y = []
        _x = []
        for pair in attr_weight_list[:25]:
            _y.append(pair[1])
            _x.append(pair[0])
        plt.figure(figsize=(10, 15))
        plt.title("Simple weight Analysis")
        plt.barh(_y, _x)
        plt.ylabel("Attribute name")
        plt.xlabel("Attribute weight")
        plt.title("Top negative weights")
        plt.savefig("../graphs/Top_Negative_Attributes.jpeg")
        plt.clf()

        # Top 50 positive weights
        # print(attr_weight_list[-50:])
        _y = []
        _x = []
        for pair in attr_weight_list[-25:]:
            _y.append(pair[1])
            _x.append(pair[0])
        plt.figure(figsize=(10, 15))
        plt.title("Simple weight Analysis")
        plt.barh(_y, _x)
        plt.ylabel("Attribute name")
        plt.xlabel("Attribute weight")
        plt.title("Top positive weights")
        plt.savefig("../graphs/Top_Positive_Attributes.jpeg")
        plt.clf()
        print("Analysis done! Generated graphs in corresponding folder")

    def split_predictions(self):
        self.true_pos = []    # predicted 1, actal 1
        self.true_neg = []    # predicted 0, actual 0
        self.false_pos = []   # predicted 1, actual 0
        self.false_neg = []   # predicted 0, actual 1
        x_validate = self.x_validate.astype(float)
        predictions = self.model.predict(x_validate)

        # Split predictions into true+, true-, false+, false-
        for i in range(len(predictions)):
            if(predictions[i] == 1 and self.y_validate[i] == 1):
                self.true_pos.append(np.array(x_validate[i]))
            if(predictions[i] == 1 and self.y_validate[i] == 0):
                self.false_pos.append(np.array(x_validate[i]))
            if(predictions[i] == 0 and self.y_validate[i] == 1):
                self.false_neg.append(np.array(x_validate[i]))
            if(predictions[i] == 0 and self.y_validate[i] == 0):
                self.true_neg.append(np.array(x_validate[i]))

        return self

    def plot_instance_graphs(self, _class, _cl):

        temp = []
        for i in range(len(self.model.coef_[0])):
            temp.append([self.model.coef_[0][i]*_class[0][i], self.featureNames[i]])    # First true_pos row
        temp.sort()
        print(temp[:25])
        print(temp[-25:])

        _y = []
        _x = []

        for pair in temp[:25]:
            _y.append(pair[1])
            _x.append(pair[0])
        plt.figure(figsize=(5, 10))
        plt.barh(_y, _x)
        plt.ylabel("Attribute name")
        plt.xlabel("Attribute weight")
        plt.title("Weight * Attribute_Value Analysis : "+_cl+" Top negative weights")
        plt.savefig("../graphs/Top_Negative_Attributes_"+_cl+"0.jpeg")
        plt.clf()

        _y = []
        _x = []
        for pair in temp[-25:]:
            _y.append(pair[1])
            _x.append(pair[0])
        plt.figure(figsize=(5, 10))
        plt.barh(_y, _x)
        plt.ylabel("Attribute name")
        plt.xlabel("Attribute weight")
        plt.title("Weight * Attribute_Value Analysis : "+_cl+" Top positive weights")
        plt.savefig("../graphs/Top_Positive_Attributes_"+_cl+"0.jpeg")
        plt.clf()

    def plot_final_graphs(self, _class, _cl):

        temp_pos = {}
        temp_neg = {}
        for name in self.featureNames[:len(self.featureNames)-1]:
            temp_pos[name] = 0
            temp_neg[name] = 0
        for j in range(len(_class)):
            temp2 = []
            for i in range(len(self.model.coef_[0])):
                temp2.append([self.model.coef_[0][i]*_class[j][i], self.featureNames[i]])    # First true_pos row
            temp2.sort()
            for pair in temp2[:50]:
                temp_neg[pair[1]] += 1
            for pair in temp2[-50:]:
                temp_pos[pair[1]] += 1
        
        temp_pos = sorted(temp_pos.items(), key=lambda x:x[1], reverse=True)
        temp_neg = sorted(temp_neg.items(), key=lambda x:x[1], reverse=True)
        # print(temp_pos[:50])
        # print(temp_neg[:50])
        
        f = open("attribute_contribution_table.txt", "w")
        sump=0
        sumn=0
        
        for pair in temp_pos[:50]:
            sump += pair[1]

        print("Top Positive attributes and contributions "+_cl, file=f)
        print("Attr_Name\t\t\t\tPercentage occurrence in top contribution", file=f)
        for pair in temp_pos[:50]:
            print(pair[0],"\t\t\t\t", pair[1]*100/sump, file=f)

        for pair in temp_neg[:50]:
            sumn += pair[1]

        print("Top Negative attributes and contributions", file=f)
        print("Attr_Name\t\t\t\tPercentage occurrence in top contribution", file=f)
        for pair in temp_neg[:50]:
            print(pair[0],"\t\t\t\t", pair[1]*100/sumn, file=f)

        f.close()
        _y = []
        _x = []

        for pair in temp_neg[:50]:
            _y.append(pair[0])
            _x.append(pair[1])
        plt.figure(figsize=(10, 15))
        plt.barh(_y, _x)
        plt.ylabel("Attribute name")
        plt.xlabel("Attribute contribution count")
        plt.title("Weight * Attribute_Value Analysis : "+_cl+" Top negative weights")
        plt.savefig("../graphs/weight_attr_analysis_"+"Top_Neg_Attr_"+_cl)
        plt.clf()

        _y = []
        _x = []
        for pair in temp_pos[:50]:
            _y.append(pair[0])
            _x.append(pair[1])
        plt.figure(figsize=(10, 15))
        plt.barh(_y, _x)
        plt.ylabel("Attribute name")
        plt.xlabel("Attribute contribution count")
        plt.title("Weight * Attribute_Value Analysis : "+_cl+" Top positive weights")
        plt.savefig("../graphs/weight_attr_analysis_"+"Top_Pos_Attr_"+_cl)
        plt.clf()

    def weight_value_analysis(self):

        print("Performing weight*value analysis ...")
        self.split_predictions()
        # self.plot_instance_graphs(true_pos, "true_pos")
        # self.plot_instance_graphs(true_neg, "true_neg")
        # self.plot_instance_graphs(false_pos, "false_pos")
        # self.plot_instance_graphs(false_neg, "false_neg")

        #self.plot_final_graphs(self.true_pos, "true_pos")
        #self.plot_final_graphs(self.true_neg, "true_neg")
        self.plot_final_graphs(self.false_pos, "false_pos")
        self.plot_final_graphs(self.false_neg, "false_neg")

        print("Analysis done! Generated graphs in corresponding folder")

    def lime_expl(self, lime_obj,  model, _class, _cl):

        class_dict={}
        for i in range(len(self.featureNames)):
            class_dict[i] = 0
        for instance in _class:
            expl = lime_obj.explain_instance(instance, self.model.predict_proba, num_features=25, num_samples=500).as_map()
            for pair in expl[1]:
                class_dict[pair[0]] += 1
        #print(_cl + " : ", class_dict)
        class_dict = sorted(class_dict.items(), key=lambda x:x[1], reverse=True)[:25]
        _y = []
        _x = []
        for pair in class_dict:
            _y.append(self.featureNames[pair[0]])
            _x.append(pair[1])
        plt.figure(figsize=(5, 10))
        plt.barh(_y, _x)
        plt.ylabel("Attribute name")
        plt.xlabel("Attribute contribution count")
        plt.title(_cl+" Top positive weights")
        plt.savefig("../graphs/LIME_Top_Positive_Attributes_"+_cl+"_Final.jpeg")
        print("Done with ", _cl)
        return class_dict

    def lime_analysis(self, cat_features=None):

        self.split_predictions()
        self.limeObj = lime_tabular.LimeTabularExplainer(training_data=self.x_train, feature_names=self.featureNames, categorical_features=cat_features)
        print("Explaining... this might take some time")
        self.lime_expl(self.limeObj, self.model, self.false_pos, "False Positive")
        #self.lime_expl(self.limeObj, self.model, self.false_neg, "False Negative")
        print("Done! Required graphs are in corresponding folder")

    def lime_instance_expl(self, cl, num):

        tempDict = {1:self.true_pos, 2:self.true_neg, 3:self.false_pos, 4:self.false_neg}
        expl = self.limeObj.explain_instance(tempDict[cl][num], self.model.predict_proba, num_features=25, num_samples=500).save_to_file("../graphs/explanation_"+str(cl)+str(num)+".html")
        print("Saved your files as html!\n")