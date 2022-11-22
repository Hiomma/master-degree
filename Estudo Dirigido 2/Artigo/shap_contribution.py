import numpy as np
import shap
import pandas as pd
import matplotlib.pyplot as plt


def get_shap_contribution(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)

    # Set the explainer with the model + shap
    explainer = shap.TreeExplainer(model)

    # Return the shap values with the X_test dataset
    shap_values = explainer.shap_values(X_test)

    # Constributions
    vals = np.abs(shap_values).mean(0)

    # Returning the values as DataFrame
    feature_importance = pd.DataFrame(list(zip([i for i in range(X_test.shape[1])], sum(vals))), columns = ['col_name','feature_importance_vals'])

    feature_importance.sort_values(by = ['feature_importance_vals'], ascending = False,inplace = True)
    feature_importance.set_index('col_name', inplace = True)

    return feature_importance

# Plot the selected variable
def plot_variable(X, variable):
    plt.plot(X[:, variable])
    plt.title("Variable " + str(variable))
    plt.show()