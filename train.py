import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_PATH = 'data/mammographic_mass.csv'
OUTPUT_FILE = 'model.bin'
C = 0.1

categorical = ['shape', 'margin', 'density']
numerical = ['age']


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.replace('-', '_')
    # bi_rads is the radiologist's own rating (non-predictive in UCI docs),
    # so we drop it to avoid data leakage
    df = df.drop(columns=['bi_rads'])
    return df


def prepare(df, age_median):
    df = df.copy()
    df['age'] = df['age'].fillna(age_median)
    for col in categorical:
        df[col] = df[col].apply(
            lambda v: 'unknown' if pd.isna(v) else str(int(float(v)))
        )
    return df


def main():
    df = load_data(DATA_PATH)

    # 60% train / 20% validation / 20% test
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.severity.values
    y_val = df_val.severity.values
    y_test = df_test.severity.values

    # median age comes from the training split only
    age_median = df_train.age.median()

    df_train = prepare(df_train, age_median)
    df_val = prepare(df_val, age_median)
    df_test = prepare(df_test, age_median)

    # final model: train on train + validation, evaluate once on test
    df_full_train = pd.concat([df_train, df_val])
    y_full_train = np.concatenate([y_train, y_val])

    full_train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
    test_dicts = df_test[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(full_train_dicts)
    X_test = dv.transform(test_dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_full_train, y_full_train)

    y_pred = model.predict_proba(X_test)[:, 1]
    print(f'Final AUC on test set: {roc_auc_score(y_test, y_pred):.4f}')

    with open(OUTPUT_FILE, 'wb') as f_out:
        pickle.dump((dv, model, age_median), f_out)
    print(f'Model saved to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()