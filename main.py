import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv('HR.csv')

sns.barplot(x="salary", y="left", data=df)
# plt.show()

sns.barplot(x="Department", y="left", data=df)
# plt.show()

sns.barplot(x="Department", y="time_spend_company", data=df)

sns.barplot(x="salary", y="average_montly_hours", data=df)
#plt.show()

sns.barplot(x="Department", y="satisfaction_level", data=df)
#plt.show()

x = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years']]
y = df.loc[:, 'left']

model = linear_model.LogisticRegression()

#x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.3)

model.fit(x, y)
print(model.score(x.tail(), y.tail()))







