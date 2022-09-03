import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class Model_Eval:
    def __init__(self) -> None:
        """
        Initialize Model evaluation class
        """

    def cross_val(self, model, x, y, cv=5):
        scoring = ['accuracy', 'precision', 'recall', 'f1']
        return cross_validate(estimator=model,
                              X=x,
                              y=y,
                              cv=cv,
                              scoring=scoring,
                              return_train_score=True,
                              return_estimator=True
                              )
    def plot_result(self,result,x_label,plot_title,image_name,_type):
        train_data = result['train_'+_type]
        val_data = result['test_'+_type]
        y_label = _type.capitalize()
        plot_title = y_label + ' Result of ' + plot_title
        
        plt.figure(figsize=(12, 6))
        labels = ["1st Fold", "2nd Fold", "3rd Fold", "4th Fold", "5th Fold"]
        X_axis = np.arange(len(labels))
        ax = plt.gca()
        plt.ylim(0.40000, 1)
        plt.bar(X_axis-0.2, train_data, 0.4, color='green', label='Training')
        plt.bar(X_axis+0.2, val_data, 0.4, color='yellow', label='Validation')
        plt.title(plot_title, fontsize=30)
        plt.xticks(X_axis, labels)
        plt.xlabel(x_label, fontsize=14)
        plt.ylabel(y_label, fontsize=14)
        plt.legend()
        plt.grid(True)
        # plt.savefig(image_name)
