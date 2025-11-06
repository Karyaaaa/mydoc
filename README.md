#2a  Removing Leading & Trailing Spaces


data = "   Data Science !!!   "
print('>', data, '<')
cleandata = data.strip()
print('>', cleandata, '<')

#2b  Removing Non-Printable String

import string

data = "Data\x00Science and\x02 ML is \x10fun!!!!"
clean = ''.join(c for c in data if c in string.printable)
print(clean)


#2c  Formatting Date

import datetime as dt

d = dt.date(2025, 7, 2)
print(d.strftime("%Y-%m-%d"))

d2 = dt.datetime.strptime("2025-07-02", "%Y-%m-%d")
print(d2.strftime("%d %B %Y"))



#5a   Handling Missing Values with Mean Imputation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
marks = np.random.randint(20, 100, 20).astype(float)
marks[[2,5,11,15,19]] = np.nan   # missing values

df = pd.DataFrame({"Marks": marks})
print("Original:\n", df)

df2 = df.fillna(df.Marks.mean())
print("\nAfter Mean Fill:\n", df2)

plt.hist(df.Marks.dropna())
plt.show()

plt.hist(df2.Marks)
plt.show()





#5b   Handling Missing Values with Median Imputation

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(24)
marks = np.random.randint(20,120,20).astype(float)
marks[[3,7,9,12,17]] = np.nan   # missing

df = pd.DataFrame({"Marks": marks})
print("Original:\n", df)

df2 = df.fillna(df.Marks.median())
print("\nAfter Median Fill:\n", df2)

plt.hist(df.Marks.dropna())
plt.show()

plt.hist(df2.Marks)
plt.show()



#5c  Handling Outliers using IQR Method (for non-normal/skewed data)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(101)
df = pd.DataFrame({"Value": np.random.exponential(2, 100)})

Q1, Q3 = df.Value.quantile([0.25, 0.75])
IQR = Q3 - Q1
low, high = Q1 - 1.5*IQR, Q3 + 1.5*IQR

df2 = df[(df.Value >= low) & (df.Value <= high)]

plt.boxplot(df.Value)
plt.show()

plt.boxplot(df2.Value)
plt.show()



#5d   Handling Outliers using 2-5core Method (for normal distribution)

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(202)
df = pd.DataFrame({"Value": np.random.normal(50, 10, 100)})

z = np.abs(stats.zscore(df.Value))
df2 = df[z < 3]

plt.hist(df.Value)
plt.show()

plt.hist(df2.Value)
plt.show()


#6a   Implement Normalization using Min-Max Scaling

import pandas as pd

df = pd.read_csv("sample_data1.csv")

df["Age_norm"]    = (df.Age    - df.Age.min())    / (df.Age.max()    - df.Age.min())
df["Income_norm"] = (df.Income - df.Income.min()) / (df.Income.max() - df.Income.min())
df["Score_norm"]  = (df.Score  - df.Score.min())  / (df.Score.max()  - df.Score.min())

print(df[["Age","Age_norm","Income","Income_norm","Score","Score_norm"]])



#6b  Implement Standardization using Z-score

import pandas as pd

df = pd.read_csv("sample_data1.csv")

df["Age_std"]    = (df.Age    - df.Age.mean())    / df.Age.std()
df["Income_std"] = (df.Income - df.Income.mean()) / df.Income.std()
df["Score_std"]  = (df.Score  - df.Score.mean())  / df.Score.std()

print(df[["Age","Age_std","Income","Income_std","Score","Score_std"]])



#6c  Implement Encoding - Convert categorical data to numeric

import pandas as pd

df = pd.read_csv("sample_data1.csv")

df["Gender_label"] = df.Gender.map({"Male":1, "Female":0})

print(df[["Gender","Gender_label"]])



#6d i  Transformation for Right-Skewed Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({"X": np.random.exponential(1000, 500)})

df["Log"]  = np.log1p(df.X)
df["Sqrt"] = np.sqrt(df.X)

plt.hist(df.X)
plt.show()

plt.hist(df.Log)
plt.show()

plt.hist(df.Sqrt)
plt.show()




#6d ii   Transformation for Left-Skewed Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
df = pd.DataFrame({"X": np.random.beta(6, 2, 500) * 100})

df["Square"] = df.X ** 2

plt.hist(df.X)
plt.show()

plt.hist(df.Square)
plt.show()




#7a   Implement One Sample t-Test

from scipy import stats

weights = [68, 72, 71, 69, 75, 70, 74, 73]

t, p = stats.ttest_1samp(weights, 70)

print("T:", t)
print("P:", p)

if p < 0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")




#7b   Implement Two Sample t-Test

from scipy import stats

g1 = [80,85,78,90,88]
g2 = [75,70,80,78,72]

t, p = stats.ttest_ind(g1, g2)

print("T:", t)
print("P:", p)

if p < 0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")




# 7c  Implement Chi-Square Test

from scipy.stats import chi2_contingency

data = [[30,10],
        [20,40]]

chi2, p, dof, exp = chi2_contingency(data)

print("Chi2:", chi2)
print("P:", p)
print("Expected:\n", exp)

if p < 0.05:
    print("Reject H0")
else:
    print("Fail to Reject H0")




#10 A   Univariate Analysis - Analyzing a single variable

import pandas as pd
import matplotlib.pyplot as plt


marks = [45, 60, 55, 70, 65, 50, 75, 80, 60, 55]
df = pd.DataFrame({"Marks": marks})


print("Mean:", df.Marks.mean())
print("Median:", df.Marks.median())
print("Mode:", df.Marks.mode()[0])
print("Std Dev:", df.Marks.std())
print("Min:", df.Marks.min())
print("Max:", df.Marks.max())


plt.hist(df.Marks)
plt.show()


plt.boxplot(df.Marks)
plt.show()




#10b   Bivariate Analysis - Analyzing the relationship between two variables
import pandas as pd
import matplotlib.pyplot as plt


data = {
    "Study_Hours": [2, 4, 6, 8, 10, 12],
    "Marks": [40, 60, 65, 75, 80, 90]
}
df = pd.DataFrame(data)


print("Correlation:", df.Study_Hours.corr(df.Marks))


plt.scatter(df.Study_Hours, df.Marks)
plt.show()


plt.plot(df.Study_Hours, df.Marks)
plt.show()





#10c   Multivariate Analysis - Analyzing 3 or more variables together

import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame({
    "Study":[2,3,4,5,6,7,8,9,10,11,12],
    "Sleep":[8,8,7,7,7,6,6,6,5,5,5],
    "Marks":[50,52,55,58,62,66,70,75,80,85,90]
})

print(df.corr())

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.scatter(df.Study, df.Sleep, df.Marks)
plt.show()
