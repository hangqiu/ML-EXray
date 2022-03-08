import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from MLEXray.Utils.params import ModelName, PipelineName

plot_dir = "python/plots/"

data = dict()

data[ModelName.MobileNetV2] = dict()
data[ModelName.MobileNetV2][PipelineName.Reference] = 0.7
data[ModelName.MobileNetV2][PipelineName.Mobile] = 0.57
data[ModelName.MobileNetV2][PipelineName.RefQuant] = 0.66
data[ModelName.MobileNetV2][PipelineName.MobileQuant] = 0.3

dataframe = pd.DataFrame.from_dict(data)
print(dataframe)

plt.figure(figsize=[6,2])
sns.barplot(data=dataframe.transpose(), order=[PipelineName.Reference, PipelineName.Mobile, PipelineName.RefQuant, PipelineName.MobileQuant])
# sns.barplot(data=dataframe, x=ModelName.MobileNetV2)
plt.ylabel("Accuracy")
# plt.legend()
plt.savefig(f"{plot_dir}accuracy_val.png")