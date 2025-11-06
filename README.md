10 A

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


10b
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
