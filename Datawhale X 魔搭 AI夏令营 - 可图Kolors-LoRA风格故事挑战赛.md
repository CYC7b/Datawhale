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

<details>
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

<details open>
  <summary><b>Task3：进阶上分-实战优化</b></summary>

  ## 1. 工具初探一ComfyUI应用场景探索
  ComfyUI 是一个为生成对抗网络（GAN）和扩散模型设计的用户界面工具。它主要用于帮助用户在不需要编写复杂代码的情况下，直观地设计和操作机器学习模型，尤其是在图像生成和编辑领域。ComfyUI 提供了一种图形化的方式，用户可以通过拖放操作将不同的模型组件（如生成器、判别器、损失函数等）连接在一起，从而快速构建和实验不同的网络结构。

  #### 1.1 主要功能和特点
  
  图形化用户界面：ComfyUI 提供了一个友好的图形界面，用户可以通过拖放操作构建模型，而不必编写复杂的代码。
  
  模块化设计：它允许用户将模型设计成模块，每个模块代表模型的一部分（如层、激活函数、优化器等），用户可以自由组合这些模块。
  
  实时预览：在调整模型参数或结构后，用户可以立即看到生成结果，这有助于快速迭代和优化模型。
  
  支持多种模型架构：ComfyUI 支持各种流行的图像生成模型，如 GAN 和扩散模型等。

  #### 1.2 ComfyUI图片生成流程（以扩散模型为例）：

  1. 设置输入节点

    首先，用户需要在界面上添加输入节点。输入节点通常包括以下几种：
  
  - 噪声输入节点：用于生成随机噪声图像，作为扩散模型的初始输入。
  - 条件输入节点：如果你使用的是条件生成模型，例如基于文本的图像生成，输入节点将接收文本描述或其他条件信息。
  
  2. 选择和配置模型节点
  
    接下来，需要选择用于图像生成的模型节点。ComfyUI 支持多种模型架构，例如：
  
  - 生成模型节点（UNet、VAE 等）：这是核心部分，用于从输入噪声或条件中生成图像。
  - 预训练模型加载节点：加载你所需的预训练模型权重，这些模型会用于指导图像生成过程。
  
  3. 配置扩散流程节点
  
    在扩散模型的生成过程中，通常需要配置扩散流程相关的节点：
  
  - 扩散步数节点：设置扩散过程中的步数，步数越多，生成的图像细节越多，但生成时间也会增加。
  - 调度器节点：定义扩散过程中的时间步调度策略，如线性或指数调度器。
  
  4. 连接处理单元
  
    图像生成过程中可能还需要一些图像处理单元节点：
  
  - 图像后处理节点：用于图像生成后的处理，如裁剪、调整大小、颜色校正等。
  - 输出格式节点：设置生成图像的输出格式，如 PNG、JPEG 等。
  
  5. 配置输出节点
  
    最后，添加输出节点，用于保存或显示生成的图像：

  - 图像输出节点：将生成的图像保存到本地文件系统中，或直接在界面中显示。
  - 日志节点：记录生成过程中的信息，便于调试或记录生成结果。
  
  6. 执行生成流程
  
    配置完成后，用户可以运行整个数据流。ComfyUI 会自动执行连接的各个节点，逐步生成并输出图像。执行过程中，用户可以实时监控生成的中间结果和最终图像。


  **简单概括为如下流程：**
  
  噪声输入节点 -> 2. 生成模型节点 -> 3. 扩散步数节点 -> 4. 图像后处理节点 -> 5. 图像输出节点

  #### 1.3 20分钟速通安装ComfyUI

  下载安装ComfyUI的执行文件和task1中微调完成Lora文件
  ```
git lfs install
git clone https://www.modelscope.cn/datasets/maochase/kolors_test_comfyui.git
mv kolors_test_comfyui/* ./
rm -rf kolors_test_comfyui/
mkdir -p /mnt/workspace/models/lightning_logs/version_0/checkpoints/
mv epoch=0-step=500.ckpt /mnt/workspace/models/lightning_logs/version_0/checkpoints/   
  ```

  运行ComfyUI.ipynb。当执行到最后一个节点的内容输出了一个访问的链接的时候，复制链接到浏览器中访问。即可进入ComfyUI界面。

  加载工作流脚本，并完成第一次生图。

  可以替换工作流脚本，使用Task1中微调得到的Lora模型。


  ## 2. Lora微调

  #### 2.1 Lora简介
  LoRA (Low-Rank Adaptation) 微调是一种用于在预训练模型上进行高效微调的技术。它可以通过高效且灵活的方式实现模型的个性化调整，使其能够适应特定的任务或领域，同时保持良好的泛化能力和较低的资源消耗。这对于推动大规模预训练模型的实际应用至关重要。

  #### 2.2 Lora微调的原理
  LoRA通过在预训练模型的关键层中添加低秩矩阵来实现。这些低秩矩阵通常被设计成具有较低维度的参数空间，这样它们就可以在不改变模型整体结构的情况下进行微调。在训练过程中，只有这些新增的低秩矩阵被更新，而原始模型的大部分权重保持不变。

  #### 2.3 Lora微调的优势
  节省资源：LoRA 微调通过更新少量参数，显著降低了计算资源需求和显存占用。

  灵活性高：LoRA 可以快速适应不同任务，微调后的模型易于共享和迁移。

  降低过拟合风险：低秩分解减少模型自由度，帮助模型保持更好的泛化能力。

  #### 2.4 Lora详解

  Task1中的微调代码：

  ```python
  import os
  cmd = """
  python DiffSynth-Studio/examples/train/kolors/train_kolors_lora.py \ # 选择使用可图的Lora训练脚本DiffSynth-Studio/examples/train/kolors/train_kolors_lora.py
    --pretrained_unet_path models/kolors/Kolors/unet/diffusion_pytorch_model.safetensors \ # 选择unet模型
    --pretrained_text_encoder_path models/kolors/Kolors/text_encoder \ # 选择text_encoder
    --pretrained_fp16_vae_path models/sdxl-vae-fp16-fix/diffusion_pytorch_model.safetensors \ # 选择vae模型
    --lora_rank 16 \ # lora_rank 16 表示在权衡模型表达能力和训练效率时，选择了使用 16 作为秩，适合在不显著降低模型性能的前提下，通过 LoRA 减少计算和内存的需求
    --lora_alpha 4.0 \ # 设置 LoRA 的 alpha 值，影响调整的强度
    --dataset_path data/lora_dataset_processed \ # 指定数据集路径，用于训练模型
    --output_path ./models \ # 指定输出路径，用于保存模型
    --max_epochs 1 \ # 设置最大训练轮数为 1
    --center_crop \ # 启用中心裁剪，这通常用于图像预处理
    --use_gradient_checkpointing \ # 启用梯度检查点技术，以节省内存
    --precision "16-mixed" # 指定训练时的精度为混合 16 位精度（half precision），这可以加速训练并减少显存使用
  """.strip()
  os.system(cmd) # 执行可图Lora训练
  ```

  #### 2.5 如何改进微调代码

  1.优化超参数
  
  学习率调整：尝试使用学习率调度器（如 CosineAnnealingLR 或 ReduceLROnPlateau），以便动态调整学习率，提升模型的收敛速度和效果。
  
  批大小调整：适当调整批大小，平衡内存占用与训练速度。如果可能，使用渐进式批量增大策略。
  
  2. 增加数据增强
  
  对输入数据进行更多的增强操作（如随机裁剪、旋转、颜色抖动等），以提高模型的泛化能力，防止过拟合。
  
  3. 改进模型架构
  
  多任务微调：如果有多个相关任务，可以考虑多任务学习，通过共享部分模型参数来提升各任务的性能。
  
  更复杂的LoRA层：在 LoRA 微调中，尝试在更多模型层中应用 LoRA，或者针对特定任务调整 LoRA 的目标模块。
  
  4. 使用更高效的优化器
  
  尝试使用 AdamW 或 Lion 等更先进的优化器，它们在某些情况下可以比传统的 Adam 提供更好的性能和稳定性。
  
  5. 添加早停机制
  
  实现早停（Early Stopping）机制，根据验证集性能动态停止训练，避免模型过拟合。
  
  6. 调整精度
  
  如果适合，可以尝试使用混合精度训练（FP16 和 FP32），以加快训练速度并减少显存使用，同时保持数值稳定性。

  
  
    
