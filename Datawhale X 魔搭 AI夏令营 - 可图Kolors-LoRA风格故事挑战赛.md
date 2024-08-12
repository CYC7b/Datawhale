# Datawhale X 魔搭 AI夏令营 - 可图Kolors-LoRA风格故事挑战赛

学习者手册:https://linklearner.com/activity/14/10/24

赛事地址:[http://competition.sais.com.cn/competitionDetail/532230/competitionData](https://tianchi.aliyun.com/competition/entrance/532254/information)

数据集下载:[http://competition.sais.com.cn/competitionDetail/532230/competitionData](https://tianchi.aliyun.com/competition/entrance/532254/information)

<details>
  <summary><b>Task1：赛题解析&运行</b></summary>

  ## 1. 评分机制
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

<details open>
  <summary><b>Task2：精读baseline&实战基于话剧的连环画制作</b></summary>
  
  ## 1. baseline精读
   
  #### 1. 环境设置与数据下载
  - 环境设置：首先，笔记本中的代码导入了必要的Python库，如os和json，这是为了确保能够处理文件和数据。
  - 数据下载：使用特定的库（如modelscope.msdatasets）来下载所需的数据集。这一步是准备数据的起点，确保后续的数据处理和模型训练有数据可用。

  #### 2. 数据处理
  - 配置数据处理：通过设定data_juicer_config来配置数据处理的参数。这一步是为了预处理数据，如格式转换、清洗等，以便于模型能够更好地学习。
  - 执行数据处理：代码中可能包含执行具体数据处理任务的命令，比如调用data-juicer的脚本或功能来处理数据。

  #### 3. 模型微调
  
  模型微调是指在预训练的模型基础上，通过继续训练来调整模型以适应新的任务。在baseline中，模型微调包括：

  - 下载预训练模型：通过使用diffsynth库来下载预训练的模型或模型架构。这是为了在训练前准备好模型框架。
  - 查看训练参数：运行训练脚本之前，通过打印出训练参数的帮助信息，确认每个参数的意义和设置。
  - 设置随机种子：多次设置随机种子确保模型训练的可重复性。
  - 启动训练脚本：通过命令行运行训练脚本开始模型训练。

  #### 4. 结果处理与生成图像
  - 加载模型：训练完成后，代码将加载训练好的模型，以便进行测试或生成结果。
  - 生成图像：最后，使用训练好的模型来生成图像或其他类型的输出。

  ## 2. 实战基于话剧的连环画制作

  #### 1. 使用通义千问设计最佳的提示词

  使用下面的提示词，让通义千问设计出八张图片的提示词。
  
  ```
你是一个文生图专家，我们现在要做一个实战项目，就是要编排一个文生图话剧
话剧由8张场景图片生成，你需要输出每张图片的生图提示词

具体的场景图片
1、女主正在上课
2、开始睡着了
3、进入梦乡，梦到自己站在路旁
4、王子骑马而来
5、两人相谈甚欢
6、一起坐在马背上
7、下课了，梦醒了
8、又回到了学习生活中

生图提示词要求
1、风格为古风
2、根据场景确定是使用全身还是上半身
3、人物描述
4、场景描述
5、做啥事情

例子：
古风，水墨画，一个黑色长发少女，坐在教室里，盯着黑板，深思，上半身，红色长裙
  ```

  自己在通义的返回的基础上，多多调整，争取打磨出一个最佳的提示词。

  这是我最终使用的提示词：

  | 场景描述          | 正向提示词 | 反向提示词 |
| ----------------- | ----------- | ----------- |
| **女主正在上课**  | 古风，水墨画，一个黑色长发少女，坐在古代书院的教室里，认真听讲，盯着老师手中的竹简，上半身，绿色长裙，背景有木质桌椅和古书架 | 现代风格，数字画，丑陋，变形，嘈杂，模糊，低对比度，扭曲的手指，多余的手指 |
| **开始睡着了**    | 古风，水墨画，一个黑色长发少女，伏在书桌上，眼睛半闭，显露出疲惫感，上半身，绿色长裙，周围书籍散乱，窗外夕阳透进来 | 现代风格，数字画，丑陋，变形，嘈杂，模糊，低对比度，扭曲的手指，多余的手指 |
| **进入梦乡**      | 古风，水墨画，一个黑色长发少女，站在一条青石板小路旁，四周桃花盛开，全身，白色飘逸长裙，背景是远处的古风建筑，天空中有几朵淡云 | 现代风格，数字画，丑陋，变形，嘈杂，模糊，低对比度，扭曲的手指，多余的手指 |
| **王子骑马而来**  | 古风，水墨画，一位身穿银色盔甲的年轻王子，骑着一匹白色骏马从远处走来，手持长剑，全身，背景为古风山野，桃花纷飞 | 现代风格，数字画，丑陋，变形，嘈杂，模糊，低对比度，扭曲的手指，多余的手指 |
| **两人相谈甚欢**  | 古风，水墨画，黑色长发少女与身穿银甲的王子站在路旁，微笑交谈，彼此注视，上半身，少女着白色长裙，王子手持缰绳，背景为桃花盛开的古风小路 | 现代风格，数字画，丑陋，变形，嘈杂，模糊，低对比度，扭曲的手指，多余的手指 |
| **一起坐在马背上**| 古风，水墨画，黑色长发少女与王子共骑一匹白马，少女侧坐在马背上，双手环抱王子的腰，全身，少女穿白色长裙，王子穿银甲，背景为青山绿水 | 现代风格，数字画，丑陋，变形，嘈杂，模糊，低对比度，扭曲的手指，多余的手指 |
| **下课了 梦醒了** | 古风，水墨画，黑色长发少女从书桌上抬起头，眼神迷茫刚从梦中醒来，上半身，绿色长裙，书桌上的竹简散乱，窗外阳光明媚 | 现代风格，数字画，丑陋，变形，嘈杂，模糊，低对比度，扭曲的手指，多余的手指 |
| **又回到了学习生活中** | 古风，水墨画，黑色长发少女站在教室中，手握毛笔在书案上认真书写，全身，绿色长裙，背景为书院的木质书架和摆满书籍的桌子 | 现代风格，数字画，丑陋，变形，嘈杂，模糊，低对比度，扭曲的手指，多余的手指 |

  

  #### 2. 生成图片

  打开baseline，替换原来的提示词。重新运行baseline，只需要从加载模型开始往下执行代码块即可。

</details>
