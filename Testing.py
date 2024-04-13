import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset/water_potability.csv")

df.dropna(axis=0, inplace=True)

kmeans = KMeans(n_clusters = 2)
kmeans.fit(df)
labels = kmeans.labels_
df_kmeans = df[labels == 1]
df_kmeans.reset_index(inplace = True) # ???

X = df_kmeans.iloc[:, :-1]
y = df_kmeans.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

def accuracy(X_train, X_test, y_train, y_test):
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    y_gbc = gbc.predict(X_test)
    return accuracy_score(y_test, y_gbc)

accuracy(X_train_sc, X_test_sc, y_train, y_test)