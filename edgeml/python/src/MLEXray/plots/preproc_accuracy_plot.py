import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from MLEXray.Utils.params import ModelName, PipelineName

plot_dir = "python/plots/"

data = dict()

data[ModelName.MobileNetV1] = dict()
data[ModelName.MobileNetV1][PipelineName.Mobile] = 0.69
data[ModelName.MobileNetV1][PipelineName.MobileResize] = 0.69
data[ModelName.MobileNetV1][PipelineName.MobileChannel] = 0.63
data[ModelName.MobileNetV1][PipelineName.MobileNormalization] = 0.62
data[ModelName.MobileNetV1][PipelineName.MobileRotation] = 0.3

data[ModelName.MobileNetV2] = dict()
data[ModelName.MobileNetV2][PipelineName.Mobile] = 0.69
data[ModelName.MobileNetV2][PipelineName.MobileResize] = 0.7
data[ModelName.MobileNetV2][PipelineName.MobileChannel] = 0.6
data[ModelName.MobileNetV2][PipelineName.MobileNormalization] = 0.57
data[ModelName.MobileNetV2][PipelineName.MobileRotation] = 0.42


data[ModelName.MobileNetV3_Large] = dict()
data[ModelName.MobileNetV3_Large][PipelineName.Mobile] = 0.72
data[ModelName.MobileNetV3_Large][PipelineName.MobileResize] = 0.75
data[ModelName.MobileNetV3_Large][PipelineName.MobileChannel] = 0.65
data[ModelName.MobileNetV3_Large][PipelineName.MobileNormalization] = 0.62
data[ModelName.MobileNetV3_Large][PipelineName.MobileRotation] = 0.54

data[ModelName.MobileNetV3_Small] = dict()
data[ModelName.MobileNetV3_Small][PipelineName.Mobile] = 0.7
data[ModelName.MobileNetV3_Small][PipelineName.MobileResize] = 0.72
data[ModelName.MobileNetV3_Small][PipelineName.MobileChannel] = 0.51
data[ModelName.MobileNetV3_Small][PipelineName.MobileNormalization] = 0.54
data[ModelName.MobileNetV3_Small][PipelineName.MobileRotation] = 0.38

data[ModelName.InceptionV3] = dict()
data[ModelName.InceptionV3][PipelineName.Mobile] = 0.72
data[ModelName.InceptionV3][PipelineName.MobileResize] = 0.73
data[ModelName.InceptionV3][PipelineName.MobileChannel] = 0.69
data[ModelName.InceptionV3][PipelineName.MobileNormalization] = 0.73
data[ModelName.InceptionV3][PipelineName.MobileRotation] = 0.5

data[ModelName.DenseNet121] = dict()
data[ModelName.DenseNet121][PipelineName.Mobile] = 0.67
data[ModelName.DenseNet121][PipelineName.MobileResize] = 0.64
data[ModelName.DenseNet121][PipelineName.MobileChannel] = 0.57
data[ModelName.DenseNet121][PipelineName.MobileNormalization] = 0.62
data[ModelName.DenseNet121][PipelineName.MobileRotation] = 0.43


data[ModelName.ResNet50V2] = dict()
data[ModelName.ResNet50V2][PipelineName.Mobile] = 0.67
data[ModelName.ResNet50V2][PipelineName.MobileResize] = 0.65
data[ModelName.ResNet50V2][PipelineName.MobileChannel] = 0.6
data[ModelName.ResNet50V2][PipelineName.MobileNormalization] = 0.63
data[ModelName.ResNet50V2][PipelineName.MobileRotation] = 0.29

dataframe = pd.DataFrame.from_dict(data)
dataframe = dataframe.transpose()
dataframe = dataframe.rename(columns={PipelineName.Mobile:"Mobile\n(Baseline)"})
dataframe = dataframe.transpose()
print(dataframe)
dataframe=dataframe.stack().reset_index()
dataframe.columns = ['Pipelines', 'Model', 'Accuracy']
print(dataframe)

ratio = 2.0
plt.figure(figsize=[16/ratio, 9/ratio])
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
    loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=5,
    columnspacing=0.5
)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{plot_dir}image_preprocess_bug.png")