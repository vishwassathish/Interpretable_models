# Declare modules used in the program 

from tkinter.filedialog import askopenfilenames, askopenfilename
from backend import *
from sys import exit
from pickle import load

def exit_():
    exit(0)

# Train the model and plot the required curves for given dataset
def trainModel():
    '''
    Ask user to upload a pickle file containing data in the dictionary format:
    {
        'feature_names':featureNameList [f1, f2 ...]
        'x_train':x_train_numpy_array, 
        'y_train':y_train_numpy_array, 
        'x_validate':x_validate_numpy_array, 
        'y_validate':y_validate_numpy_array
    }
    '''
    dataFile = askopenfilename(title = "Select a pickle data file", filetypes = (("pickle files", "*.pickle"), ("all files","*.*")))
    f = open(dataFile, "rb")
    myData = load(f)
    x_train = myData['x_train']
    y_train = myData['y_train']
    x_validate = myData['x_validate']
    y_validate = myData['y_validate']

    # Create a custom logistic regression object. Check 'backend.py' for implementation. Method names are self explanatory
    logObj = LogRegression(x_train, y_train, x_validate, y_validate)
    logObj.train_and_test()
    logObj.print_confusion_matrix()
    logObj.plot_precision_recall_curve()
    logObj.plot_roc_curve()
    logObj.save_model()

def explain():
    # Ask user to upload the model as a pickle file
    print("PROVIDE THE DATA FILE USED DURING TRAINING THE MODEL")
    dataFile = askopenfilename(title = "Select a pickle data file", filetypes = (("pickle files", "*.pickle"), ("all files","*.*")))
    f = open(dataFile, "rb")
    myData = load(f)
    featureNames = myData['feature_names']
    x_train = myData['x_train']
    x_validate = myData['x_validate']
    y_validate = myData['y_validate']

    print("NOW PROVIDE THE TRAINED MODEL")
    model = askopenfilename(title = "Select a trained model file (pickle)", filetypes = (("pickle files", "*.pickle"), ("all files","*.*")))
    f = open(model, "rb")
    myModel = load(f)
    print(myModel)
    # Create Explaination object.
    '''
    There are 3 kinds of explanations:
        1) Simple weight analysis
        2) Weight*Attr_value analysis
        3) Lime tabular explanation
    ''' 
    explObj = ExplModel(myModel, x_train, x_validate, y_validate, featureNames)
    explObj.weight_analysis()
    explObj.weight_value_analysis()
    print("Success")

def lime():

    print("PROVIDE THE DATA FILE USED DURING TRAINING THE MODEL")
    dataFile = askopenfilename(title = "Select a pickle data file", filetypes = (("pickle files", "*.pickle"), ("all files","*.*")))
    f = open(dataFile, "rb")
    myData = load(f)
    featureNames = myData['feature_names']
    x_train = myData['x_train']
    x_validate = myData['x_validate']
    y_validate = myData['y_validate']

    print("NOW PROVIDE THE TRAINED MODEL")
    model = askopenfilename(title = "Select a trained model file (pickle)", filetypes = (("pickle files", "*.pickle"), ("all files","*.*")))
    f = open(model, "rb")
    myModel = load(f)
    print(myModel)

    print("Upload a file containing categorical features")
    cat_file = askopenfilename(title="Categorical features pickle file", filetypes = (("pickle files", "*.pickle"), ("all files","*.*")))
    with open(cat_file, "rb") as f:
        cat = load(f)
    cat_features = list(cat.values())

    explObj = ExplModel(myModel, x_train, x_validate, y_validate, featureNames)
    explObj.lime_analysis(cat_features)

    print("\n\nYOU CAN TEST EXPLANATION FOR CUSTOM INPUTS\nAll you have to do is input two SPACE SEPARATED integers\n1) class number from (1)true+, (2)true-, (3)false+, (4)false-\n2) Test example number in that class (zero indexed)")
    ch = input("Want to test with LIME on custom input?(y/n)\n")
    
    while(ch == 'y' or ch == 'Y'):

        cl, num = [int(x) for x in input("Give 2 space separated integers\n").strip().split(" ")]
        explObj.lime_instance_expl(cl, num)
        ch = input("Want to test with LIME on custom input?(y/n)\n")

    exit(0)