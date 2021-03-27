import pandas as pd # load and manipulate data and for One-Hot Encoding
import numpy as np # calculate the mean and standard deviation
import xgboost as xgb # XGBoost stuff
from sklearn.model_selection import train_test_split # split  data into training and testing sets
from sklearn.model_selection import GridSearchCV # cross validation
from sklearn.metrics import confusion_matrix # creates a confusion matrix
from sklearn.metrics import plot_confusion_matrix # draws a confusion matrix


df = pd.read_csv('default_of_credit_card_clients.tsv', 
                 header=1, ## NOTE: The second line contains column names, so we skip the first line
                 sep='\t') ## NOTE: Pandas automatically detects delimeters, but it never hurts to be specific

## NOTE: We can also read in the original MS Excel file directly from the website
# df = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00350/defaultget_ipython().run_line_magic("20of%20credit%20card%20clients.xls',", " ")
#                  header=1,
#                  sep='\t')


df.head()


df.rename({'default payment next month' : 'DEFAULT'}, axis='columns', inplace=True)
df.head()


df.drop('ID', axis=1, inplace=True) ## set axis=0 to remove rows, axis=1 to remove columns
df.head()


df.dtypes


df['SEX'].unique()


df['EDUCATION'].unique()


df['MARRIAGE'].unique()


len(df.loc[(df['EDUCATION'] == 0) | (df['MARRIAGE'] == 0)])


len(df)


X = df.drop('DEFAULT', axis=1).copy() # alternatively: X = df_no_missing.iloc[:,:-1]
X.head()


y = df['DEFAULT'].copy()
y.head()


X.dtypes


pd.get_dummies(X, columns=['MARRIAGE']).head()


X_encoded = pd.get_dummies(X, columns=['SEX',
                                       'EDUCATION',
                                       'MARRIAGE', 
                                       'PAY_0',
                                       'PAY_2',
                                       'PAY_3',
                                       'PAY_4',
                                       'PAY_5',
                                       'PAY_6'])
X_encoded.head()


X_encoded.dtypes


X_encoded.dtypes.unique()


y.dtypes


sum(y)/len(y)


X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify=y)


sum(y_train)/len(y_train)


sum(y_test)/len(y_test)


clf_xgb = xgb.XGBClassifier(objective='binary:logistic',
                            eval_metric="logloss", ## this avoids a warning...
                            seed=42,
                            use_label_encoder=False)
clf_xgb.fit(X_train, y_train)


plot_confusion_matrix(clf_xgb, 
                      X_test, 
                      y_test,
                      values_format='d',
                      display_labels=["Did not default", "Defaulted"])


# ## NOTE: When data are imbalanced, the XGBoost manual says...
# ## If you care only about the overall performance metric (AUC) of your prediction
# ##     * Balance the positive and negative weights via scale_pos_weight
# ##     * Use AUC for evaluation
# ## ALSO NOTE: I ran GridSearchCV sequentially on subsets of parameter options, rather than all at once
# ## in order to optimize parameters in a short period of time.

# ## ROUND 1
# # param_grid = {
# #     'max_depth': [3, 4, 5],
# #     'learning_rate': [0.1, 0.01, 0.05],
# #     'gamma': [0, 0.25, 1.0],
# #     'reg_lambda': [0, 1.0, 10.0],
# #     'scale_pos_weight': [1, 3, 5] # NOTE: XGBoost recommends sum(negative instances) / sum(positive instances)
# # }
# ## Output: max_depth: 5, learning_rate: 0.1, gamma: 0, reg_lambda: 10, scale_pos_weight: 3
# ## Because max_depth, learning_rate and reg_lambda were at the ends of their range, we will continue to explore those...

# ## ROUND 2
# # param_grid = {
# #     'max_depth': [5, 6, 7],
# #     'learning_rate': [0.1, 0.5, 1],
# #     'gamma': [0],
# #     'reg_lambda': [10.0, 20, 100],
# #      'scale_pos_weight': [3]
# # }
# ## Output: max_depth: 6, learning_rate: 0.1, reg_lambda: 100. Because reg_lambda was a the end, we will continue to explore

# ## Round 3
# param_grid = {
#     'max_depth': [6],
#     'learning_rate': [0.1],
#     'gamma': [0],
#     'reg_lambda': [100, 500, 1000],
#     'scale_pos_weight': [3]
# }

# ## NOTE: To speed up cross validiation, and to further prevent overfitting.
# ## We are only using a random subset of the data (90get_ipython().run_line_magic(")", " and are only")
# ## using a random subset of the features (columns) (50get_ipython().run_line_magic(")", " per tree.")
# optimal_params = GridSearchCV(
#     estimator=xgb.XGBClassifier(objective='binary:logistic', 
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
                        gamma=0,
                        learning_rate=0.1,
                        max_depth=6,
                        reg_lambda=500,
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
                      display_labels=["Did not default", "Defaulted"])


## If we want to get information, like gain and cover etc, at each node in the first tree, 
## we just build the first tree, otherwise we'll get the average over all of the trees.
clf_xgb = xgb.XGBClassifier(seed=42,
                            objective='binary:logistic',
                            eval_metric="logloss", ## this avoids a warning...
                            gamma=0,
                            learning_rate=0.1,
                            max_depth=6,
                            reg_lambda=500,
                            scale_pos_weight=3,
                            subsample=0.9,
                            colsample_bytree=0.5,
                            n_estimators=1, ## We set this to 1 so we can get gain, cover etc.
                            use_label_encoder=False
                            )
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
xgb.to_graphviz(clf_xgb, num_trees=0, 
                condition_node_params=node_params,
                leaf_node_params=leaf_params)
## if you want to save the figure...
# graph_data = xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10", 
#                 condition_node_params=node_params,
#                 leaf_node_params=leaf_params) 
# graph_data.view(filename='xgboost_tree_credit_card') ## save as PDF
