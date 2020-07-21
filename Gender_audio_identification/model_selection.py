from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier as Mlp
from sklearn.neighbors import KNeighborsClassifier as Knc
from sklearn.ensemble import RandomForestClassifier as Rf
from gather_data import DfCreator


def select_model(df, features, target):
    """Function to search for best model parameters.

    Input:
        df (pd.DataFrame): train dataframe.
        features (str): list of features columns.
        target (str): target column.

    Returns:
        str: list of the best parameters for the selected models.
    """

    x = df[features]
    y = df[target]

    models = [
        {
            "name": "MLPClassifier",
            "estimator": Mlp(random_state=1, activation='relu'),
            "hyperparameters":
                {
                    "hidden_layer_sizes": [(100, 100, 100,),
                                           (100, 100, 100, 100,),
                                           (50, 100, 200,)
                                           ],
                    "solver": ['lbfgs', 'sgd'],
                    "max_iter": [500, 1000]
                }
        },

        {
            "name": "KNeighborsClassifier",
            "estimator": Knc(n_jobs=4),
            "hyperparameters":
                {
                    "n_neighbors": [5, 7, 9, 11],
                    "algorithm": ['ball_tree', 'kd_tree'],
                    "leaf_size": [30, 40, 50]
                }
        },

        {
            "name": "RandomForestClassifier",
            "estimator": Rf(random_state=1, n_jobs=4),
            "hyperparameters":
                {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "max_leaf_nodes": [3, 4, 5],
                    "min_samples_split": [2, 3, 4]
                }
        }]

    for m in models:
        print(m['name'])
        print('-' * len(m['name']))

        grid = GridSearchCV(m["estimator"],
                            param_grid=m["hyperparameters"],
                            scoring='f1',
                            cv=10,
                            n_jobs=4)

        grid.fit(x, y)
        m["best_params"] = grid.best_params_
        m["best_score"] = grid.best_score_
        m["best_model"] = grid.best_estimator_

        print("Best Score: {}".format(m["best_score"]))
        print("Best Parameters: {}\n".format(m["best_params"]))

    return models


def model_train(db_t):
    """Function for model parameters search.

    Input:
        db_train (pd.DataFrame): dataframe for model train.

    Returns:
        str: models, parameters and F1 scores.
    """

    df_feat = db_t.columns[:-1]
    df_target = db_t.columns[-1]
    model_selection_result = select_model(db_train, df_feat, df_target)
    print(model_selection_result)


# Run this script to search for best algorithm.
# Use workflow.py to test model on test dataframe.
if __name__ == "__main__":
    srm = 8000  # 8000 hz
    num_features = 40  # number of MFCC features to gather

    # Create dataframe from the audio:
    dbc = DfCreator("wav_data", srm, num_features)
    features_f = dbc.file_reader("f.txt")
    features_m = dbc.file_reader("m.txt")
    full_df = dbc.make_db([features_f, features_m])
    # No nulls check:
    print("DB has nulls: ", dbc.db_check_for_nulls(full_df))

    # Get test and train dataframe:
    db_test = dbc.make_test_db(full_df, 20)  # ~10% from full_db
    db_train = dbc.make_train_db(full_df, db_test)
    df_feat_test = db_train.iloc[:, :-1]
    df_target_test = db_train.iloc[:, -1]
    # Delete "target" column from test_df:
    test_notarget = db_test.drop('target', axis=1)
    test_notarget.reset_index(drop=True, inplace=True)

    model_train(db_train)
