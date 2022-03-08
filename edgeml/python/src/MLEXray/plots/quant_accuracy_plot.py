import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from MLEXray.Utils.params import ModelName, PipelineName

plot_dir = "python/plots/"

data = dict()

data[ModelName.MobileNetV1] = dict()
data[ModelName.MobileNetV1][PipelineName.Reference] = 0.69
data[ModelName.MobileNetV1][PipelineName.Mobile] = 0.69
data[ModelName.MobileNetV1][PipelineName.RefQuant] = 0.66
data[ModelName.MobileNetV1][PipelineName.MobileQuant] = 0

data[ModelName.MobileNetV2] = dict()
data[ModelName.MobileNetV2][PipelineName.Reference] = 0.7
data[ModelName.MobileNetV2][PipelineName.Mobile] = 0.69
data[ModelName.MobileNetV2][PipelineName.RefQuant] = 0.66
data[ModelName.MobileNetV2][PipelineName.MobileQuant] = 0


data[ModelName.MobileNetV3_Large] = dict()
data[ModelName.MobileNetV3_Large][PipelineName.Reference] = 0.72
data[ModelName.MobileNetV3_Large][PipelineName.Mobile] = 0.72
data[ModelName.MobileNetV3_Large][PipelineName.RefQuant] = 0
data[ModelName.MobileNetV3_Large][PipelineName.MobileQuant] = 0

data[ModelName.MobileNetV3_Small] = dict()
data[ModelName.MobileNetV3_Small][PipelineName.Reference] = 0.69
data[ModelName.MobileNetV3_Small][PipelineName.Mobile] = 0.7
data[ModelName.MobileNetV3_Small][PipelineName.RefQuant] = 0
data[ModelName.MobileNetV3_Small][PipelineName.MobileQuant] = 0

data[ModelName.InceptionV3] = dict()
data[ModelName.InceptionV3][PipelineName.Reference] = 0.72
data[ModelName.InceptionV3][PipelineName.Mobile] = 0.72
data[ModelName.InceptionV3][PipelineName.RefQuant] = 0.74
data[ModelName.InceptionV3][PipelineName.MobileQuant] = 0.74

data[ModelName.DenseNet121] = dict()
data[ModelName.DenseNet121][PipelineName.Reference] = 0.69
data[ModelName.DenseNet121][PipelineName.Mobile] = 0.67
data[ModelName.DenseNet121][PipelineName.RefQuant] = 0.65
data[ModelName.DenseNet121][PipelineName.MobileQuant] = 0.66

data[ModelName.ResNet50V2] = dict()
data[ModelName.ResNet50V2][PipelineName.Reference] = 0.67
data[ModelName.ResNet50V2][PipelineName.Mobile] = 0.67
data[ModelName.ResNet50V2][PipelineName.RefQuant] = 0.66
data[ModelName.ResNet50V2][PipelineName.MobileQuant] = 0.64

dataframe = pd.DataFrame.from_dict(data)
dataframe = dataframe.transpose()
dataframe = dataframe.rename(columns={PipelineName.Reference:"Reference (Baseline)"})
dataframe = dataframe.transpose()
print(dataframe)
dataframe=dataframe.stack().reset_index()
dataframe.columns = ['Pipelines', 'Model', 'Accuracy']
print(dataframe)

ratio = 0.8
plt.figure(figsize=[4/ratio, 3.5/ratio])
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
    loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=2,
    # columnspacing=0.5
)
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{plot_dir}quant_bug.png")