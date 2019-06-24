# Declare modules used in the program 

from tkinter import Tk, PhotoImage, Label, Button
from modules import *


### MAIN GUI TRIGGERS AND FUNCTIONS FOR THE TOOL ###
if __name__ == "__main__" :

    main = Tk()
    main.title("SBI : Explainable ML, Logistic Regression")
    
    #frame = tkinter.Frame(main, width = 100, height = 10).pack()
    logo = PhotoImage(file="../data/sbi2.png")
    l1 = Label(main, image = logo).pack(side = "top")
    text = "Explainable ML toolkit. Train a model for your dataset, explain feature contribution for each ."
    l2 = Label(main, text = text).pack(side = "top")


    print("Train a model by uploading training data!\nExplain a trained model")
    w0 = Button(main, text = "Train a logistic regression model", width = 25, command = trainModel)
    w1 = Button(main, text = "Explain model", width = 25, command = explain)
    w2 = Button(main, text = "Use Lime to analyze model", width = 25, command = lime)
    w3 = Button(main, text = "Exit", width = 25, command = exit_)
    w3.pack(side = "bottom")
    w2.pack(side = "bottom")
    w1.pack(side = "bottom")
    w0.pack(side = "bottom")
    
    main.mainloop()