# Datawhale AI 夏令营-siRNA药物药效预测

学习者手册:[https://exn8g66dnwu.feishu.cn/docx/T7WGd7goqowRvFxwoApclo9Pn0b](https://linklearner.com/activity/12/4/4)

赛事地址:http://competition.sais.com.cn/competitionDetail/532230/competitionData

数据集下载:https://github.com/CYC7b/Datawhale/blob/main/siRNA_0715.zip

<details>
  <summary><b>Task1：赛题解析&背景入门</b></summary>

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
  

</details>

<details>
  <summary><b>Task2：深入理解赛题，入门RNN和特征工程</b></summary>
  <br>
  
  **我在官方给出的lgb.ipynb基础上，改进后的算法见[improved_lgb.py](https://github.com/CYC7b/Datawhale/blob/main/improved_lgb.py)**
  
  本任务我们对官方的baseline进行分析解读，之后介绍RNN相关的基础知识，包括其适用范围和问题。随后，我们从特征工程构建的角度来重新分析赛题数据，并将其和lightgbm结合，最终给出一个更好的baseline。

  ### 官方baseline分析
  
  在baseline中，我们只用到了siRNA_antisense_seq和modified_siRNA_antisense_seq_list，它们都是由一串符号标记的序列，我们希望的是把这些序列特征能够输入RNN模型，因此需要对其做一定处理。在SiRNAModel类的forward方法中，展示了在得到序列特征的tensor表示后的处理步骤：
  
  ```python
  def forward(self, x):
    # 将输入序列传入嵌入层
    embedded = [self.embedding(seq) for seq in x]
    outputs = []
    ...
  ```

  那么这里的输入x是什么呢？我们可以通过train_loader来查看一个batch内的输入情况，这里的inputs和上面的x是一个东西。我们首先发现inputs包含两个元素，它们分别对应的是前面提到的两个使用的特征，每个元素的尺寸都是64*25，64代表batch的大小，25代表序列的长度。这里我们可以从inputs[0][0]看到每一行数据的siRNA_antisense_seq被向量化后的情况，这个例子中我们发现前面的7位是非零数，表示其序列编码后每一位的唯一标识；而后面都是0，这是因为RNN模型的输入需要每个样本的长度一致，因此我们需要事先算出一个所有序列编码后的最大长度，然后补0。
  
  ![image](https://github.com/user-attachments/assets/992c87e2-2dc4-4ddc-876c-1998608773ef)

  那么我们怎么能得到这个唯一标识呢？我们首先需要把序列给进行分词，siRNA_antisense_seq的分词策略是3个一组（GenomicTokenizer的ngram和stride都取3）进行token拆分，比如AGCCGAGAU会被分为[AGC, CGA, GAU]，而modified_siRNA_antisense_seq_list会进行按照空格分词（因为它本身已经根据空格分好了）。由此我们可以从整个数据集构建出一个词汇表，他负责token到唯一标识（索引）的映射：

  ```python
  # 创建词汇表
  all_tokens = []
  for col in columns:
      for seq in train_data[col]:
          if ' ' in seq:  # 修饰过的序列
              all_tokens.extend(seq.split())
          else:
              all_tokens.extend(tokenizer.tokenize(seq))
  vocab = GenomicVocab.create(all_tokens, max_vocab=10000, min_freq=1)
  ```

  有了这个词汇表，我们就可以
  - 来获得序列的最大长度
  
  ```python
  max_len = max(max(len(seq.split()) if ' ' in seq else len(tokenizer.tokenize(seq)) 
                    for seq in train_data[col]) for col in columns)
  ```

  - 在loader获取样本的时候把token转为索引

  ```python
  def __getitem__(self, idx):
      # 获取数据集中的第idx个样本
      row = self.df.iloc[idx]  # 获取第idx行数据
      
      # 对每一列进行分词和编码
      seqs = [self.tokenize_and_encode(row[col]) for col in self.columns]
      if self.is_test:
          # 仅返回编码后的序列（测试集模式）
          return seqs
      else:
          # 获取目标值并转换为张量（仅在非测试集模式下）
          target = torch.tensor(row['mRNA_remaining_pct'], dtype=torch.float)
          # 返回编码后的序列和目标值
          return seqs, target
  
  def tokenize_and_encode(self, seq):
      if ' ' in seq:  # 修饰过的序列
          tokens = seq.split()  # 按空格分词
      else:  # 常规序列
          tokens = self.tokenizer.tokenize(seq)  # 使用分词器分词
      
      # 将token转换为索引，未知token使用0（<pad>）
      encoded = [self.vocab.stoi.get(token, 0) for token in tokens]
      # 将序列填充到最大长度
      padded = encoded + [0] * (self.max_len - len(encoded))
      # 返回张量格式的序列
      return torch.tensor(padded[:self.max_len], dtype=torch.long)
  ```

  此时，对于某一行数据，其两个特征分别为AGCCUUAGCACA和u u g g u u Cf c，假设整个数据集对应token编码后序列的最大长度为10，那么得到的特征就可能是
  - [25, 38, 25, 24, 0, 0, 0, 0, 0, 0]
  - [65, 65, 63, 63, 65, 65, 74, 50, 0, 0]
  那么假设batch的大小为16，此时forword函数的x就会是两个列表，每个列表的tensor尺寸为16 * 10

  ### RNN模型分析
  
  我们在上一小节已经得到了数据的张量化表示，此时就要把它输入模型了。

```python
  class SiRNAModel(nn.Module):
      def __init__(self, vocab_size, embed_dim=200, hidden_dim=256, n_layers=3, dropout=0.5):
          super(SiRNAModel, self).__init__()
          
          # 初始化嵌入层
          self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
          # 初始化GRU层
          self.gru = nn.GRU(embed_dim, hidden_dim, n_layers, bidirectional=True, batch_first=True, dropout=dropout)
          # 初始化全连接层
          self.fc = nn.Linear(hidden_dim * 4, 1)  # hidden_dim * 4 因为GRU是双向的，有n_layers层
          # 初始化Dropout层
          self.dropout = nn.Dropout(dropout)
      
      def forward(self, x):
          # 将输入序列传入嵌入层
          embedded = [self.embedding(seq) for seq in x]
          outputs = []
          
          # 对每个嵌入的序列进行处理
          for embed in embedded:
              x, _ = self.gru(embed)  # 传入GRU层
              x = self.dropout(x[:, -1, :])  # 取最后一个隐藏状态，并进行dropout处理
              outputs.append(x)
          
          # 将所有序列的输出拼接起来
          x = torch.cat(outputs, dim=1)
          # 传入全连接层
          x = self.fc(x)
          # 返回结果
          return x.squeeze()
```

  我们首先第一步将得到的索引进行了embedding，token的embedding是将离散的符号（如单词、字符、或基因序列片段）映射到连续的向量空间的过程。这个过程通过将高维的稀疏表示（如独热编码）转换为低维的密集向量表示，使得相似的符号在向量空间中距离更近。此时，embed的尺寸会从BatchSize * Length成为BatchSize * Length * EmbeddingSize，此处EmbeddingSize即embed_dim=200。
RNN，全称为递归神经网络（Recurrent Neural Network），是一种人工智能模型，特别擅长处理序列数据。它和普通的神经网络不同，因为它能够记住以前的数据，并利用这些记忆来处理当前的数据。想象你在读一本书。你在阅读每一页时，不仅仅是单独理解这一页的内容，还会记住前面的情节和信息。这些记忆帮助你理解当前的情节并预测接下来的发展。这就是 RNN 的工作方式。假设你要预测一个句子中下一个单词是什么。例如，句子是：“我今天早上吃了一个”。RNN 会根据之前看到的单词（“我今天早上吃了一个”），预测下一个可能是“苹果”或“香蕉”等。它记住了之前的单词，并利用这些信息来做出预测。
- RNN 在处理序列数据时具有一定的局限性：
  - 长期依赖问题：RNN 难以记住和利用很久以前的信息。这是因为在长序列中，随着时间步的增加，早期的信息会逐渐被后来的信息覆盖或淡化。
  - 梯度消失和爆炸问题：在反向传播过程中，RNN 的梯度可能会变得非常小（梯度消失）或非常大（梯度爆炸），这会导致训练过程变得困难。
- LSTM 的改进
  - LSTM 通过引入一个复杂的单元结构来解决 RNN 的局限性。LSTM 单元包含三个门（输入门、遗忘门和输出门）和一个记忆单元（细胞状态），这些门和状态共同作用，使 LSTM 能够更好地捕捉长期依赖关系。
    1. 输入门：决定当前输入的信息有多少会被写入记忆单元。
    2. 遗忘门：决定记忆单元中有多少信息会被遗忘。
    3. 输出门：决定记忆单元的哪些部分会作为输出。
  - 通过这些门的控制，LSTM 可以选择性地保留或遗忘信息，从而有效地解决长期依赖和梯度消失的问题。
- GRU 的改进
  - GRU 是 LSTM 的一种简化版本，它通过合并一些门来简化结构，同时仍然保留了解决 RNN 局限性的能力。GRU 仅有两个门：更新门和重置门。
    1. 更新门：决定前一个时刻的状态和当前输入信息的结合程度。
    2. 重置门：决定忘记多少之前的信息。
  - GRU 的结构更简单，计算效率更高，同时在许多应用中表现出与 LSTM 类似的性能。
我们在pytorch的GRU文档中可以找到对应可选的参数信息，我们需要特别关注的参数如下，它们决定了模型的输入输出的张量维度
  - input_size（200）
  - hidden_size（256）
  - bidirectional（True）
  
假设输入的BatchSize为16，序列最大长度为10，即x尺寸为16 * 10 * 200，那么其输出的张量尺寸为 16 * 10 * (256 * 2)。
在从GRU模型输出后，x = self.dropout(x[:, -1, :])使得输出变为了BatchSize * (hidden_dim * 2)，此处取了序列最后一个位置的输出数据（注意RNN网络的记忆性），这里的2是因为bidirectional参数为True，随后x = torch.cat(outputs, dim=1)指定在第二个维度拼接后，通过全连接层再映射为标量，因此最后经过squeeze（去除维数为1的维度）后得到的张量尺寸为批大小，从而可以后续和target值进行loss计算，迭代模型。

  ### 数据的特征工程

  在提交官方baseline后，我们会发现得分并不好，这一方面原因可能在于数据用的特征还较为简单，序列特征的构造较为粗糙，再加上对于深度学习的RNN模型而言数据量可能还不太充足的因素，这是可以预见的output。下面我们介绍一种把序列特征的问题转化为表格问题的方法，并介绍在表格数据上如何做特征工程。
- 处理类别型变量
如何知道一个变量是类别型的呢，只需看下其值的分布，或者唯一值的个数

```python
df.gene_target_symbol_name.nunique()
```
```python
df.gene_target_symbol_name.value_counts()
```
如果相较于数据的总行数很少，那么其很可能就是类别变量了，比如gene_target_symbol_name。此时，我们可以使用get_dummie函数来实现one-hot特征的构造.
```python
# 如果有40个类别，那么会产生40列，如果第i行属于第j个类别，那么第j列第i行就是1，否则为0
df_gene_target_symbol_name = pd.get_dummies(df.gene_target_symbol_name)
df_gene_target_symbol_name.columns = [
    f"feat_gene_target_symbol_name_{c}" for c in df_gene_target_symbol_name.columns
]
```

- 可能的时间特征构造
在数据观察的时候发现，siRNA_duplex_id的编码方式很有意思，其格式为AD-1810676.1，我们猜测AD是某个类别，后面的.1是版本，当中的可能是按照一定顺序的序列号，因此可以构造如下特征

```python
siRNA_duplex_id_values = df.siRNA_duplex_id.str.split("-|\.").str[1].astype("int")
```

- 包含某些单词
  
```python
df_cell_line_donor = pd.get_dummies(df.cell_line_donor)
df_cell_line_donor.columns = [
    f"feat_cell_line_donor_{c}" for c in df_cell_line_donor.columns
]
# 包含Hepatocytes
df_cell_line_donor["feat_cell_line_donor_hepatocytes"] = (
    (df.cell_line_donor.str.contains("Hepatocytes")).fillna(False).astype("int")
)
# 包含Cells
df_cell_line_donor["feat_cell_line_donor_cells"] = (
    df.cell_line_donor.str.contains("Cells").fillna(False).astype("int")
)
```

- 根据序列模式提取特征
假设siRNA的序列为ACGCA...，此时我们可以根据上一个task中提到的rna背景知识，对碱基的模式进行特征构造
```python
def siRNA_feat_builder(s: pd.Series, anti: bool = False):
    name = "anti" if anti else "sense"
    df = s.to_frame()
    # 序列长度
    df[f"feat_siRNA_{name}_seq_len"] = s.str.len()
    for pos in [0, -1]:
        for c in list("AUGC"):
            # 第一个和最后一个是否是A/U/G/C
            df[f"feat_siRNA_{name}_seq_{c}_{'front' if pos == 0 else 'back'}"] = (
                s.str[pos] == c
            )
    # 是否已某一对碱基开头和某一对碱基结尾
    df[f"feat_siRNA_{name}_seq_pattern_1"] = s.str.startswith("AA") & s.str.endswith(
        "UU"
    )
    df[f"feat_siRNA_{name}_seq_pattern_2"] = s.str.startswith("GA") & s.str.endswith(
        "UU"
    )
    df[f"feat_siRNA_{name}_seq_pattern_3"] = s.str.startswith("CA") & s.str.endswith(
        "UU"
    )
    df[f"feat_siRNA_{name}_seq_pattern_4"] = s.str.startswith("UA") & s.str.endswith(
        "UU"
    )
    df[f"feat_siRNA_{name}_seq_pattern_5"] = s.str.startswith("UU") & s.str.endswith(
        "AA"
    )
    df[f"feat_siRNA_{name}_seq_pattern_6"] = s.str.startswith("UU") & s.str.endswith(
        "GA"
    )
    df[f"feat_siRNA_{name}_seq_pattern_7"] = s.str.startswith("UU") & s.str.endswith(
        "CA"
    )
    df[f"feat_siRNA_{name}_seq_pattern_8"] = s.str.startswith("UU") & s.str.endswith(
        "UA"
    )
    # 第二位和倒数第二位是否为A
    df[f"feat_siRNA_{name}_seq_pattern_9"] = s.str[1] == "A"
    df[f"feat_siRNA_{name}_seq_pattern_10"] = s.str[-2] == "A"
    # GC占整体长度的比例
    df[f"feat_siRNA_{name}_seq_pattern_GC_frac"] = (
        s.str.contains("G") + s.str.contains("C")
    ) / s.str.len()
    return df.iloc[:, 1:]
```

### 基于lightgbm的baseline
在得到了表格数据之后，我们可以使用任意适用于表格数据的机器学习回归模型来进行预测，此处我们简单使用了lightgbm模型：

```python
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

def print_validation_result(env):
    result = env.evaluation_result_list[-1]
    print(f"[{env.iteration}] {result[1]}'s {result[0]}: {result[2]}")

params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "root_mean_squared_error",
    "max_depth": 7,
    "learning_rate": 0.02,
    "verbose": 0,
}

gbm = lgb.train(
    params,
    train_data,
    num_boost_round=15000,
    valid_sets=[test_data],
    callbacks=[print_validation_result],
)
```
可一键运行的完整baseline见lgb.ipynb

</details>

<details open>
  <summary><b>Task3：特征工程进阶，持续上分</b></summary>
  
  ### 1. 特征工程改进

- 序列长度：lgb_revise 中添加了序列长度特征，并将其转换为整数类型。

- 特定位置的碱基类型：提取特定位置的碱基类型，并将其转换为整数类型。

- GC含量及局部GC含量：

  - lgb_revise 计算了全局GC含量和局部GC含量，后者采用了多种窗口大小（如6、10、15）进行计算。
  - 这些GC含量特征被进一步细化，确保模型能够捕捉到序列中的局部特性。

- 前后两端的碱基特征：

  - lgb_revise 对序列的首尾碱基进行了编码，使用数字映射（如A=1, U=2, G=3, C=4）代替原来的字符串表示。

- 熔解温度（Tm）：

  - 计算了siRNA序列的熔解温度（Tm），这一特性对RNA干扰效率有重要影响。

- 三联体重复模式：

  - 检查序列中是否存在重复模式，如三联体等，并将其编码为数值特征。

### 2. 使用Optuna进行超参数优化

在 lgb_revise 中，使用Optuna库进行超参数优化，以找到最佳的模型参数。这一过程包括定义目标函数，并通过多次试验找出最佳参数组合。

### 3. 使用最佳参数进行训练
在找到最佳参数后，使用这些参数进行模型的最终训练。

</details>
