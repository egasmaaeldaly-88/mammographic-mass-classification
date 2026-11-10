import pandas as pd
from ucimlrepo import fetch_ucirepo

# download the dataset from UCI (id=161)
dataset = fetch_ucirepo(id=161)

# the library returns two tables: features and target
X = dataset.data.features
y = dataset.data.targets

# put them side by side in one table
df = pd.concat([X, y], axis=1)

# save as csv
df.to_csv('data/mammographic_mass.csv', index=False)
print(df.shape)