import pandas as pd
from sklearn import preprocessing

def preprocess(df):
    
    columns_to_remove = []
    columns_to_label_encode = []
    
    for col in df.columns:
        if df[col].nunique() == 1:
            columns_to_remove.append(col)
            continue
        if df[col].nunique() == df.shape[0]:
            columns_to_remove.append(col)
            continue
        if df[col].isna().astype(int).sum() > df.shape[0] * 0.4:
            columns_to_remove.append(col)
            continue
        if df[col].dtype == 'object':
            columns_to_label_encode.append(col)
    
    df = df.drop(columns_to_remove, axis=1)
    #df = df.replace("?", np.NaN)
    df = df.dropna()
    
    
    for col in columns_to_label_encode:
        le = preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col]).astype(int)
    val = df.values #returns a numpy array
    print(df.shape[0])
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(val)
    df = pd.DataFrame(x_scaled, columns=df.columns)
    return df