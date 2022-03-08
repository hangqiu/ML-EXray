import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from MLEXray.Utils.params import ModelName, PipelineName

plot_dir = "python/plots/"

data = dict()

data['CNN1'] = dict()
data['CNN1'][PipelineName.Mobile] = 0.8
data['CNN1'][PipelineName.MobileResize] = 0.79
data['CNN1'][PipelineName.MobileNormalization] = 0.35

data['CNN2'] = dict()
data['CNN2'][PipelineName.Mobile] = 0.9326
data['CNN2'][PipelineName.MobileResize] = 0.9059
data['CNN2'][PipelineName.MobileNormalization] = 0.1259


dataframe = pd.DataFrame.from_dict(data)
dataframe = dataframe.transpose()
dataframe = dataframe.rename(columns={PipelineName.Mobile:"Mobile\n(Baseline)"})
dataframe = dataframe.transpose()
print(dataframe)
dataframe=dataframe.stack().reset_index()
dataframe.columns = ['Pipelines', 'Model', 'Accuracy']
print(dataframe)

ratio = 2.0
plt.figure(figsize=[6/ratio, 9/ratio])
# plt.figure()
sns.barplot(data=dataframe, x='Model', y='Accuracy',
            hue='Pipelines',
            # order=[PipelineName.Mobile, PipelineName.MobileResize, PipelineName.MobileChannel, PipelineName.MobileNormalization, PipelineName.MobileRotation],
            )
# sns.barplot(data=dataframe, x=ModelName.MobileNetV2)
plt.ylabel("Accuracy")
plt.xlabel("")

plt.legend(
    # ['Baseline(Mobile)', 'Resize', 'Channel', 'Normalization', 'Rotation'],
    loc='upper center', bbox_to_anchor=(0.4, 1.15), ncol=2,
    columnspacing=0.5
)
# plt.xticks(rotation=20)
plt.tight_layout()
plt.ylim([0,1])
plt.savefig(f"{plot_dir}speech_preprocess_bug.png")