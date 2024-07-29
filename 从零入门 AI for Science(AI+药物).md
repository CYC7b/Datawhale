# Datawhale AI 夏令营-siRNA药物药效预测

学习者手册:[https://exn8g66dnwu.feishu.cn/docx/T7WGd7goqowRvFxwoApclo9Pn0b](https://linklearner.com/activity/12/4/4)

赛事地址:http://competition.sais.com.cn/competitionDetail/532230/competitionData

数据集下载:https://github.com/CYC7b/Datawhale/blob/main/siRNA_0715.zip

<details>
  <summary>Task1：赛题解析&背景入门</summary>

  ### 1. 赛题介绍
  本次比赛旨在利用机器学习技术，预测化学修饰后的siRNA序列在RNA干扰（RNAi）机制下对靶基因的沉默效率。RNAi是一种重要的基因表达调控机制，通过干扰特定基因的表达，可以用于疾病治疗。这次比赛的目标是通过构建并优化模型，准确预测siRNA的沉默效率，从而提升药物设计的效率和效果。

  **比赛流程**
  - 初赛阶段：提供一部分公开文献中提取的siRNA修饰序列和实验数据，参赛者需要使用这些数据训练模型并提交预测结果。初赛的重点是评估在训练集中出现过的目标mRNA序列，不同siRNA的沉默效率预测的准确性。
  - 复赛阶段：在初赛基础上，增加部分尚未公开的专利数据作为测试数据，评估模型在未见过的目标mRNA序列上的预测准确性。

  **数据集**
  数据集包括siRNA裸序列、经过化学修饰的siRNA序列、目标mRNA序列以及实验条件（如药物浓度、细胞系、转染方式等）。最重要的字段是mRNA_remaining_pct，这是我们模型的训练目标，表示siRNA对靶基因沉默后的剩余mRNA百分比，值越低表示沉默效率越好。

  ### 2. 评分机制
  在这个部分，我们会仔细介绍官方评分方案，让大家更加了解赛事官方平台是如何给我们的提交结果打分的~

  在这次比赛中，模型的评分由多个指标共同决定，以全面评估模型的性能。这些指标包括平均绝对误差（MAE）、区间内的平均绝对误差（Range MAE）和F1得分（F1 Score）。这些指标分别衡量模型在预测上的准确性和稳定性，以及在区间内的表现。最终的评分（Score）是综合这些指标的加权结果。通过下述代码，我们可以更加了解本次赛题的评分细节。

  ```python
  # score = 50% × (1−MAE/100) + 50% × F1 × (1−Range-MAE/100)
  def calculate_metrics(y_true, y_pred, threshold=30):
      mae = np.mean(np.abs(y_true - y_pred))

      y_true_binary = (y_true < threshold).astype(int)
      y_pred_binary = (y_pred < threshold).astype(int)

      mask = (y_pred >= 0) & (y_pred <= threshold)
      range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 100

      precision = precision_score(y_true_binary, y_pred_binary, average='binary')
      recall = recall_score(y_true_binary, y_pred_binary, average='binary')
      f1 = 2 * precision * recall / (precision + recall)
      score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
      return score
  ```

**代码实现解释**

1. 平均绝对误差（MAE）：
  ```python
  # 预测值和真实值之间绝对误差的平均值，衡量了模型预测的总体准确性。
  mae = np.mean(np.abs(y_true - y_pred))
  ```

2. 二值化处理：
  ```python
  # 这里将真实值和预测值进行二值化处理：如果值小于阈值（30），则为1，否则为0。
  y_true_binary = (y_true < threshold).astype(int)
  y_pred_binary = (y_pred < threshold).astype(int)
  ```

3. 区间内的平均绝对误差（Range MAE）：
  ```python
  # 这里计算在特定区间（0到阈值30）内的平均绝对误差。如果预测值在这个区间内，才计算其误差，否则设为100。这个指标评估了模型在重要预测区间内的表现。
  mask = (y_pred >= 0) & (y_pred <= threshold)
  range_mae = mean_absolute_error(y_true[mask], y_pred[mask]) if mask.sum() > 0 else 100
  ```

4. F1 分数
  ```python
  # 精确率是正确预测为正的样本占所有预测为正的样本的比例，召回率是正确预测为正的样本占所有真实为正的样本的比例。F1得分是精确率和召回率的调和平均数，综合考虑了两者的平衡。
  precision = precision_score(y_true_binary, y_pred_binary, average='binary')
  recall = recall_score(y_true_binary, y_pred_binary, average='binary')
  f1 = 2precision * recall / (precision + recall)
  ```

5. 综合评分（Score）
  ```python
  score = (1 - mae / 100) * 0.5 + (1 - range_mae / 100) * f1 * 0.5
  ```

最终的评分结合了MAE和区间内MAE的反比例值，以及F1得分。MAE和Range MAE越小，1减去它们的比值越大，表明误差小，模型表现好。F1得分高则表示模型分类性能好。最终评分是这几个值的加权平均数，权重各占50%。

### 3.相关生物背景知识

**AI与制药**

  长期以来，药物研发领域流传着“双十定律”，即从新药研发开始到产品获批上市，平均耗时十年，投入成本约十亿美元。所幸的是，大数据与人工智能（Artificial Intelligence，AI）的兴起，有望新药的研发走出这个“双十”困局，使药物研发的进度得以加速，成功率得以提高，同时成本也得以大大降低。如提升候选药物品质（改善靶点确证和先导分子优化流程、调整药物用途），优化临床试验设计（基于生物标志物的筛查、患者分级）等。AI制药企业英矽智能通过生成式人工智能筛选靶点并设计的小分子TNIK抑制剂候选药物INS018_055已完成Ⅱ期临床试验首例患者给药，给数以百万计特发性肺纤维化（IPF）病人群带去福音，其从靶点发现到人体临床开启仅用了18个月。AI辅助制药与生命科学研究已经成为一种新的范式。
  
  小干扰RNA (small interfering RNA,siRNA)生物学最重要生物技术之一，是发现能够通过一种被称为RNA干扰(RNA interference, RNAi)的现象来调节基因的表达。siRNA可用作研究体内和体外单基因功能的工具，是一类有吸引力的新型疗法，特别是针对治疗癌症和其他疾病的不可成药靶点。2018年，在结合靶向递送系统和经过高级化学修饰之后，全球首个siRNA药物Patisiran获批上市。如今，越来越多siRNA逐渐进入试验阶段甚至步入临床，这标志着临床领域实现精准医疗将不再只是一句口号，最终惠及更多患者。
  了解生物作用的机理与背景有助于设计更好的AI模型，来辅助siRNA的设计与优化。
  
**RNAi作用机制**

  生物体内，RNAi首先将较长的双链RNA加工和切割成 siRNA，通常在每条链的3'末端带有2个核苷酸突出端。负责这种加工的酶是一种RNase III样酶，称为Dicer。形成后，siRNA与一种称为RNA诱导的沉默复合物（RNAinduced silencing complex, RISC）的多蛋白组分复合物结合。在RISC复合物中，siRNA链被分离，具有更稳定的5′末端的链通常被整合到活性RISC复合物中。然后，反义单链siRNA组分引导并排列在靶mRNA上，并通过催化RISC蛋白（Argonaute family（Ago2））的作用，mRNA被切割，即对应基因被沉默，表达蛋白能力削弱。

  ![image](https://github.com/user-attachments/assets/ccd8bd2b-a039-4365-840b-1a36991750d3)
  

</details>
