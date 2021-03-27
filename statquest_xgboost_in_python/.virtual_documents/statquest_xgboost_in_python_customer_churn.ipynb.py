import pandas as pd # load and manipulate data and for One-Hot Encoding
import numpy as np # calculate the mean and standard deviation
import xgboost as xgb # XGBoost stuff
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer # for scoring during cross validation
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import confusion_matrix # creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix


df = pd.read_csv('Telco_customer_churn.csv')


df.head()


df.drop(['Churn Label', 'Churn Score', 'CLTV', 'Churn Reason'],
        axis=1, inplace=True) ## set axis=0 to remove rows, axis=1 to remove columns
df.head()


df['Count'].unique()


df['Country'].unique()


df['State'].unique()


df['City'].unique()


df.drop(['CustomerID', 'Count', 'Country', 'State', 'Lat Long'],
        axis=1, inplace=True) ## set axis=0 to remove rows, axis=1 to remove columns
df.head()


df['City'].replace(' ', '_', regex=True, inplace=True)
df.head()


df['City'].unique()[0:10]


df.columns = df.columns.str.replace(' ', '_')
df.head()


df.dtypes


df['Phone_Service'].unique()


df['Total_Charges'].unique()


## NOTE: The next line is commented out, but you should
## uncomment it and run it so you can see the error that it
## generates.
##
## The only reason it is commented out is that if you want
## to just run all of the Python code at once, the next line
## will cause Python to quit early because it creates an error
## 
## However, the error is useful for learning how to spot missing
## data, so that is why you should uncomment it and run it if you can.

# df['Total Charges'] = pd.to_numeric(df['Total_Charges'])


len(df.loc[df['Total_Charges'] == ' '])


df.loc[df['Total_Charges'] == ' ']


df.loc[(df['Total_Charges'] == ' '), 'Total_Charges'] = 0


df.loc[df['Tenure_Months'] == 0]


df['Total_Charges'] = pd.to_numeric(df['Total_Charges'])
df.dtypes


df.replace(' ', '_', regex=True, inplace=True)
df.head()


df.size


X = df.drop('Churn_Value', axis=1).copy() # alternatively: X = df_no_missing.iloc[:,:-1]
X.head()


y = df['Churn_Value'].copy()
y.head()


X.dtypes


pd.get_dummies(X, columns=['Payment_Method']).head()


X_encoded = pd.get_dummies(X, columns=['City', 
                                       'Gender', 
                                       'Senior_Citizen', 
                                       'Partner',
                                       'Dependents',
                                       'Phone_Service',
                                       'Multiple_Lines',
                                       'Internet_Service',
                                       'Online_Security',
                                       'Online_Backup',
                                       'Device_Protection',
                                       'Tech_Support',
                                       'Streaming_TV',
                                       'Streaming_Movies',
                                       'Contract',
                                       'Paperless_Billing',
                                       'Payment_Method'])
X_encoded.head()


y.unique()


sum(y)/len(y)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)


sum(y_train)/len(y_train)


sum(y_test)/len(y_test)


clf_xgb = xgb.XGBClassifier(objective='binary:logistic', 
                            eval_metric="logloss", ## this avoids a warning...
                            missing=None, seed=42, 
                            use_label_encoder=False)
## NOTE: newer versions of XGBoost will issue a warning if you don't explitly tell it that
## you are not expecting it to do label encoding on its own (in other words, since we
## have ensured that the categorical values are all numeric, we do not expect XGBoost to do label encoding), 
## so we set use_label_encoder=False
clf_xgb.fit(X_train, 
            y_train,
            verbose=True,
            ## the next three arguments set up early stopping.
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])
## NOTE: In the video I got 50 or 60 lines of output. However, the newer version of XGBoost is faster and only prints out 20.


plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=["Did not leave", "Left"])


# ## NOTE: When data are imbalanced, the XGBoost manual says...
# ## If you care only about the overall performance metric (AUC) of your prediction
# ##     * Balance the positive and negative weights via scale_pos_weight
# ##     * Use AUC for evaluation
# ## ALSO NOTE: I ran GridSearchCV sequentially on subsets of parameter options, rather than all at once
# ## in order to optimize parameters in a short period of time.

# # ROUND 1
# param_grid = {
#     'max_depth': [3, 4, 5],
#     'learning_rate': [0.1, 0.01, 0.05],
#     'gamma': [0, 0.25, 1.0],
#     'reg_lambda': [0, 1.0, 10.0],
#     'scale_pos_weight': [1, 3, 5] # NOTE: XGBoost recommends sum(negative instances) / sum(positive instances)
# }
# # Output: max_depth: 4, learning_rate: 0.1, gamma: 0.25, reg_lambda: 10, scale_pos_weight: 3
# # Because learning_rate and reg_lambda were at the ends of their range, we will continue to explore those...

# ## ROUND 2
# param_grid = {
#     'max_depth': [4],
#     'learning_rate': [0.1, 0.5, 1],
#     'gamma': [0.25],
#     'reg_lambda': [10.0, 20, 100],
#      'scale_pos_weight': [3]
# }
# ## Output: max_depth: 4, learning_rate: 0.1, reg_lambda: 10.

# NOTE: To speed up cross validiation, and to further prevent overfitting.
# We are only using a random subset of the data (90get_ipython().run_line_magic(")", " and are only")
# using a random subset of the features (columns) (50get_ipython().run_line_magic(")", " per tree.")
# optimal_params = GridSearchCV(
#     estimator=xgb.XGBClassifier(objective='binary:logistic', 
#                                 eval_metric="logloss", ## this avoids a warning...
#                                 seed=42,
#                                 subsample=0.9,
#                                 colsample_bytree=0.5,
#                                 use_label_encoder=False),
#     param_grid=param_grid,
#     scoring='roc_auc', ## see https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#     verbose=0, # NOTE: If you want to see what Grid Search is doing, set verbose=2
#     n_jobs = 10,
#     cv = 3
# )

# optimal_params.fit(X_train, 
#                    y_train, 
#                    early_stopping_rounds=10,                
#                    eval_metric='auc',
#                    eval_set=[(X_test, y_test)],
#                    verbose=False)
# print(optimal_params.best_params_)


clf_xgb = xgb.XGBClassifier(seed=42,
                        objective='binary:logistic',
                        eval_metric="logloss", ## this avoids warning...
                        gamma=0.25,
                        learning_rate=0.1,
                        max_depth=4,
                        reg_lambda=10,
                        scale_pos_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.5,
                        use_label_encoder=False)
clf_xgb.fit(X_train, 
            y_train, 
            verbose=True, 
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])


plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=["Did not leave", "Left"])


## If we want to get information, like gain and cover etc, at each node in the first tree, 
## we just build the first tree, otherwise we'll get the average over all of the trees.
clf_xgb = xgb.XGBClassifier(seed=42,
                        eval_metric="logloss", ## this avoids another warning...
                        objective='binary:logistic',
                        gamma=0.25,
                        learning_rate=0.1,
                        max_depth=4,
                        reg_lambda=10,
                        scale_pos_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.5,
                        n_estimators=1, ## We set this to 1 so we can get gain, cover etc.)
                        use_label_encoder=False) 

clf_xgb.fit(X_train, y_train)

## now print out the weight, gain, cover etc. for the tree
## weight = number of times a feature is used in a branch or root across all trees
## gain = the average gain across all splits that the feature is used in
## cover = the average coverage across all splits a feature is used in
## total_gain = the total gain across all splits the feature is used in
## total_cover = the total coverage across all splits the feature is used in
## NOTE: Since we only built one tree, gain = total_gain and cover=total_cover
bst = clf_xgb.get_booster()
for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print('get_ipython().run_line_magic("s:", " ' % importance_type, bst.get_score(importance_type=importance_type))")

node_params = {'shape': 'box', ## make the nodes fancy
               'style': 'filled, rounded',
               'fillcolor': '#78cbe'} 
leaf_params = {'shape': 'box',
               'style': 'filled',
               'fillcolor': '#e48038'}
## NOTE: num_trees is NOT the number of trees to plot, but the specific tree you want to plot
## The default value is 0, but I'm setting it just to show it in action since it is
## counter-intuitive.
# xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10") 
## NOTE: I have heard that on windos you may need to add the
## the exectuables for graphviz to your path for this next code to work.
## For example, if graphviz is here: C:/Users/User/anaconda3/Library/bin/graphviz
## Then you may need to run the following line of code (adjusting the path as needed)
# os.environ["PATH"]+= os.pathsep + 'C:/Users/User/anaconda3/Library/bin/graphviz'
xgb.to_graphviz(clf_xgb, num_trees=0, 
                condition_node_params=node_params,
                leaf_node_params=leaf_params) 
## if you want to save the figure...
# graph_data = xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10", 
#                 condition_node_params=node_params,
#                 leaf_node_params=leaf_params) 
# graph_data.view(filename='xgboost_tree_customer_churn') ## save as PDF


import shap


clf_xgb = xgb.XGBClassifier(seed=42,
                        objective='binary:logistic',
                        eval_metric="logloss", ## this avoids warning...
                        gamma=0.25,
                        learning_rate=0.1,
                        max_depth=4,
                        reg_lambda=10,
                        scale_pos_weight=3,
                        subsample=0.9,
                        colsample_bytree=0.5,
                        use_label_encoder=False)
clf_xgb.fit(X_train, 
            y_train, 
            verbose=True, 
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])


explainer = shap.Explainer(clf_xgb)
shap_values = explainer(X_test)


shap.plots.beeswarm(shap_values)
