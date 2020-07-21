import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from gather_data import DfCreator


if __name__ == "__main__":
    sr = 8000  # 8000 hz are wav files
    num_features = 40  # number of MFCC features o gather

    # Create dataframe from the audio:
    dbc = DfCreator("wav_data", sr, num_features)
    features_f = dbc.file_reader("f.txt")
    features_m = dbc.file_reader("m.txt")
    full_df = dbc.make_db([features_f, features_m])
    print("Full dataframe columns count: ", full_df.shape[0])
    # No nulls check:
    print("Full dataframe has nulls: ", dbc.db_check_for_nulls(full_df))

    # Get test and train dataframe:
    db_test = dbc.make_test_db(full_df, 20)  # ~10% from full_db
    db_train = dbc.make_train_db(full_df, db_test)
    df_feat_test = db_train.iloc[:, :-1]
    df_target_test = db_train.iloc[:, -1]
    # Delete "target" column from test_df:
    test_notarget = db_test.drop('target', axis=1)
    test_notarget.reset_index(drop=True, inplace=True)

    # Best model with parameters from GridSearch:
    rf = RandomForestClassifier(max_depth=3, max_leaf_nodes=3,
                                min_samples_split=2, n_estimators=100)
    rf.fit(df_feat_test, df_target_test)
    predictor_rf = rf.predict(test_notarget)
    rf_df = pd.DataFrame(predictor_rf, columns=["Random Forest"])

    # Let's test algorithm:
    compare_df = pd.concat([db_test, rf_df], axis=1)
    print("""
    In "target" column is actual speaker: 1 - female, 2 - male
    In algorithm column - algorithm classification.
    """)
    print(compare_df.iloc[:, -2:])
