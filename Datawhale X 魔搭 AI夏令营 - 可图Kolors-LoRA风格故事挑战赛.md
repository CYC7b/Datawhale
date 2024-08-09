# Datawhale X 魔搭 AI夏令营 - 可图Kolors-LoRA风格故事挑战赛

学习者手册:https://linklearner.com/activity/14/10/24

赛事地址:[http://competition.sais.com.cn/competitionDetail/532230/competitionData](https://tianchi.aliyun.com/competition/entrance/532254/information)

数据集下载:[http://competition.sais.com.cn/competitionDetail/532230/competitionData](https://tianchi.aliyun.com/competition/entrance/532254/information)

<details open>
  <summary><b>Task1：赛题解析&运行</b></summary>

  ### 1. 评分机制
  本次赛事使用的评分机制如以下代码所示。

  ```python
pip install simple-aesthetics-predictor

import torch, os
from PIL import Image
from transformers import CLIPProcessor
from aesthetics_predictor import AestheticsPredictorV2Linear
from modelscope import snapshot_download


model_id = snapshot_download('AI-ModelScope/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE', cache_dir="models/")
predictor = AestheticsPredictorV2Linear.from_pretrained(model_id)
processor = CLIPProcessor.from_pretrained(model_id)
device = "cuda"
predictor = predictor.to(device)


def get_aesthetics_score(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = predictor(**inputs)
    prediction = outputs.logits
    return prediction.tolist()[0][0]


def evaluate(folder):
    scores = []
    for file_name in os.listdir(folder):
        if os.path.isfile(os.path.join(folder, file_name)):
            image = Image.open(os.path.join(folder, file_name))
            scores.append(get_aesthetics_score(image))
    if len(scores) == 0:
        return 0
    else:
        return sum(scores) / len(scores)


score = evaluate("./images")
print(score)
  ```

  ## 2. 30分钟速通指南

1. 下载baseline文件[https://github.com/CYC7b/Datawhale/blob/Datawhale%E7%AC%AC%E5%9B%9B%E6%9C%9F/baseline.ipynb]
  ```python
git lfs install
git clone https://www.modelscope.cn/datasets/maochase/kolors.git
  ```

2. 进入kolors文件夹，打开baseline.ipynb文件

3. 运行第一个代码块，安装环境，然后重启kernel
  - 安装 Data-Juicer 和 DiffSynth-Studio
  - Data-Juicer：数据处理和转换工具，旨在简化数据的提取、转换和加载过程
  - DiffSynth-Studio：高效微调训练大模型工具

4. 调整prompt，设置你想要的图片风格
  - 正向描述词：你想要生成的图片应该包含的内容
  - 反向提示词：你不希望生成的图片的内容
![image](https://github.com/user-attachments/assets/9988fe31-6ef5-4afa-850e-1896b1b4788f)

5. 依次顺序运行剩余的代码块，点击代码框左上角执行按钮，等待代码执行

  下面的代码块按照功能主要分成这几类
  - 使用Data-Juicer处理数据，整理训练数据文件
  - 使用DiffSynth-Studio在基础模型上，使用前面整理好的数据文件进行训练微调
  - 加载训练微调后的模型
  - 使用微调后的模型，生成用户指定的prompt提示词的图片

6. 将模型上传到魔搭 

</details>
