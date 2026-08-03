+++
title = 'MobileBERT NLU 实战：从数据增强到多任务训练'
date = '2026-08-03T00:00:00+08:00'
draft = false
description = '记录一个 MobileBERT NLU 项目如何完成数据增强、模型选择、多任务训练、评估和推理。'
summary = '从意图识别与槽位填充出发，串起数据准备、MobileBERT 训练、checkpoint 选择和推理评估的完整流程。'
tags = ['NLP', 'BERT', 'MobileBERT', '意图识别', '槽位填充']
categories = ['AI 实战']
ShowToc = true
TocOpen = true
+++

这篇文章整理自一个 MobileBERT NLU 训练项目。目标不是只跑通一个分类模型，而是把“数据从哪里来、模型为什么这样选、训练日志怎么看、最后怎样接入推理”这条链路串起来。

项目面向智能对话、机器人指令理解和语音助手等场景，主要同时完成三个任务：

- Domain classification：判断输入属于哪个领域；
- Intent classification：判断用户想做什么；
- Slot filling：从文本中提取时间、温度、设备名等结构化参数。

例如输入 `Set AC to 20 degrees`，模型可以输出：

~~~json
{
  "domain": "ac",
  "intent": "temperature_set",
  "slots": ["(20)@[degree]"]
}
~~~

## 先说明项目边界

当前实战资料中包含数据生成、训练和推理脚本，但脚本本身不是完整的可独立运行项目，还依赖：

- `model.py`、`metric.py`、`slotnorm.py`；
- `dataset/` 下的配置、标签映射和训练数据；
- 本地 MobileBERT 预训练模型或可访问的 Hugging Face 模型；
- 训练输出的 checkpoint。

因此，本文更适合作为训练流程和工程方法的记录。拿到完整项目文件后，再按文中的命令运行。

## 一、NLU 任务如何拆分

### 1. Domain：先判断领域

Domain 是较粗粒度的分类，例如：

~~~text
light
ac
music
robot
~~~

它可以帮助后续系统缩小候选范围，减少不同领域之间的意图混淆。

### 2. Intent：判断具体意图

在 `ac` 领域中，输入可能对应：

~~~text
temperature_set
temperature_get
power_on
power_off
~~~

Intent 决定后续调用哪个能力或任务。

### 3. Slot：提取可执行参数

仅仅判断“用户想调节空调”还不够，系统还需要提取温度值。槽位填充通常采用 BIO 标注：

~~~text
Tokens: ["set", "AC", "to", "20", "degrees"]
Tags:   [  O,    O,    O,   B-degree, I-degree ]
~~~

推理后再把槽位还原成结构化结果，例如：

~~~json
{
  "temperature": 20
}
~~~

这个拆分让模型输出和后续业务动作之间有清晰的接口，也比只输出一句自然语言更容易评估。

## 二、为什么选择 MobileBERT

这个项目的目标场景包含机器人和端侧设备，所以模型不能只看准确率，还要考虑参数量、内存占用和推理延迟。

| 模型 | 大致参数量 | 特点 | 适合场景 |
| --- | ---: | --- | --- |
| MobileBERT | 约 25M | 小、快，面向资源受限设备 | 端侧、嵌入式 |
| ALBERT-Base | 约 12M | 参数共享，体积较小 | 资源受限 |
| DistilBERT | 约 66M | 蒸馏后的速度与效果折中 | 通用部署 |
| BERT-Base | 约 110M | 经典稳定 | 服务端 |
| RoBERTa-Base | 约 125M | 通常更偏向效果 | 效果优先 |

这些数字是不同模型的常见量级，实际速度还会受到 batch size、序列长度、硬件、量化方式和推理框架影响。选型时最好在目标设备上使用同一批输入实测。

项目通过配置文件指定预训练模型，因此模型替换不应该散落在训练代码里。切换模型时至少要同步确认：

1. `hidden_size` 是否和模型输出维度一致；
2. tokenizer 是否与预训练模型匹配；
3. 最大序列长度是否满足业务输入；
4. 任务头和标签数量是否仍然正确。

## 三、数据增强：让长尾意图有更多样本

真实 NLU 数据通常存在两个问题：

- 高频意图样本很多，长尾意图样本很少；
- 同一意图的表达方式比较单一，模型容易记住固定句式。

项目使用 LLM 生成同义表达，再与原始数据合并。例如原始样本：

~~~json
{
  "text": "Set the air-conditioner to 17 degrees",
  "label": {
    "domain": "ac",
    "intent": "temperature_set",
    "slots": ["(17)@[degree]"]
  }
}
~~~

生成的新样本仍然必须保持相同的 domain、intent 和 slot 语义：

~~~json
{
  "text": "Adjust the AC temperature to 20 degrees",
  "label": {
    "domain": "ac",
    "intent": "temperature_set",
    "slots": ["(20)@[degree]"]
  }
}
~~~

生成数据时最重要的不是数量，而是标签可靠性。建议至少做四步检查：

1. 生成前按意图分组，避免模型混淆相近意图；
2. 生成后去重，尤其是和原始验证集去重；
3. 检查槽位是否仍然出现在文本中；
4. 随机抽样人工审核，再加入训练集。

如果把验证集中的句子或近似句直接扩写进训练集，最终指标可能会虚高。因此，数据增强之后要重新确认训练集、验证集和测试集的边界。

示例命令如下：

~~~bash
python3 scripts/generate_synthetic_data.py \
  -i dataset/train.txt \
  -o dataset/train_augmented.txt \
  --mode api \
  --model Meta-Llama-3-8B-Instruct \
  --api-base http://YOUR_API_SERVER:8000 \
  --num-per-intent 50 \
  --temperature 0.8
~~~

第一次运行建议使用 `--dry-run`，先验证输入格式、API 地址和输出路径。

## 四、多任务学习：共享编码器，分开任务头

模型的总体结构可以抽象成：

~~~text
文本
  ↓
Tokenizer
  ↓
MobileBERT Encoder
  ├── Domain classifier
  ├── Intent classifier
  └── Slot classifier
~~~

编码器负责抽取共享语义特征，三个任务头分别处理句级分类和 token 级序列标注。

联合损失可以写成：

~~~text
L = α × L_domain + β × L_intent + θ × L_slot
~~~

权重需要结合任务难度和业务重要性调整。不能只盯着某一个子任务的准确率：如果 slot 很高但 intent 经常错，系统仍然无法选择正确的业务动作；如果 intent 很高但 slot 错误，执行参数仍然不可靠。

## 五、训练流程

安装依赖：

~~~bash
pip install -r requirements.txt
~~~

使用增强数据训练：

~~~bash
python3 scripts/train.py \
  -c dataset/model_config.json \
  -d dataset/train_augmented.txt \
  -b 32 \
  -e 10 \
  -l 128 \
  -o checkpoints/bert_augmented
~~~

如果显存不足，可以依次尝试：

- 将 batch size 从 32 调到 16 或 8；
- 将最大序列长度从 128 调到 96；
- 减少生成数据量或训练轮数；
- 确认训练是否真的使用了 GPU。

训练参数不是越大越好。序列长度过大，会增加显存和计算量；batch size 过小，梯度噪声会变大；训练轮数过多，则更容易过拟合。

## 六、如何读训练日志

训练日志至少要同时看 train 和 eval：

~~~text
train: loss=0.645, accuracy(domain=0.92, intent=0.88, slot=0.96, overall=0.85)
eval:  loss=0.612, accuracy(domain=0.93, intent=0.89, slot=0.96, overall=0.86) *
~~~

常见情况如下：

| 现象 | 可能原因 | 调整方向 |
| --- | --- | --- |
| train 和 eval 都低 | 欠拟合、数据不足或学习率不合适 | 增加有效数据、检查标签、调整学习率 |
| train 持续升高，eval 变差 | 过拟合 | 早停、增加正则化、检查数据泄漏 |
| domain 高、intent 低 | 细粒度意图相似或标签边界不清 | 合并混淆样本，补充难例 |
| slot 明显低 | BIO 标注或文本对齐有问题 | 检查 tokenization 和槽位边界 |

带 `*` 的 checkpoint 通常代表验证集整体指标最好。不要默认使用最后一个 checkpoint；最后一轮可能已经开始过拟合。

项目记录过一组“原始数据与增强数据”的对比，但这些结果依赖具体数据集、随机种子和硬件，不能直接当成普遍结论。实际项目应保存每次实验的配置、数据版本和评估结果。

## 七、推理与评估

交互式推理：

~~~bash
python3 scripts/run.py \
  -c checkpoints/bert_augmented/best.ckpt \
  --cfg dataset \
  -i
~~~

示例输入：

~~~text
Turn on the lights
Set AC to 20 degrees
~~~

批量评估：

~~~bash
python3 scripts/run.py \
  -c checkpoints/bert_augmented/best.ckpt \
  -e \
  --dataset dataset/test.txt \
  --cfg dataset \
  --output eval_results
~~~

评估时建议把指标拆开记录：

- Domain accuracy；
- Intent accuracy；
- Slot-level precision、recall、F1；
- 三个任务同时正确的 exact match；
- 按意图分组的长尾指标；
- CPU/目标设备上的单条延迟和峰值内存。

只看 overall accuracy 很容易掩盖长尾意图和槽位错误。

## 八、上线前检查清单

在把模型接入机器人或语音助手前，可以按下面的顺序检查：

1. 训练集和测试集是否存在重复或近似泄漏；
2. 每个 intent 是否都有足够的正例和难例；
3. 槽位值包含数字、单位、中文和英文混合文本时是否能正确对齐；
4. 未知意图和低置信度输入是否有兜底策略；
5. checkpoint 是否和 tokenizer、配置文件成套保存；
6. 目标设备上的延迟、内存和模型体积是否达标；
7. 线上是否记录了错误样本，方便回流训练。

## 总结

一个可用的端侧 NLU 系统，关键不只是“换一个更小的 BERT”，而是把数据、标签、模型、评估和部署作为一个闭环：

~~~text
原始数据
  → 数据增强与审核
  → MobileBERT 多任务训练
  → 验证集选择 checkpoint
  → 端侧推理评估
  → 错误样本回流
~~~

MobileBERT 适合对资源有约束的场景，但最终是否合适，仍然要用自己的数据和目标硬件验证。先建立稳定的评估集，再做模型压缩和延迟优化，通常比单纯追求更大的训练集或更高的单项准确率更可靠。
