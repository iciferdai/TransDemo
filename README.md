# TransDemo
> 手搓Transfomer，并通过简易英译中任务，完成模型训练与推理验证  
> Hand-craft the Transformer, and complete model training and inference verification through a simple English-Chinese translation task


## 1. 项目介绍 / Introduce


Transformer 是现代大语言模型（LLM）的基础架构，自提出以来彻底推动了自然语言处理（NLP）领域的范式变革。

本项目基于 PyTorch 从零手动实现基础 Transformer 结构，不依赖框架封装好的内置模块，旨在从底层理解自注意力、多头注意力等核心组件的运行机制。

结合完成一个极简的英中翻译任务，覆盖模型搭建、数据集处理、词表构建、训练、监控、调优与推理全流程，实现理论与工程实践结合，深入掌握 Transformer 底层原理并积累实战经验。


## 2.  效果 / Effect Demo

#### 验证集测试情况：
```log
Input_[1|5][0|4] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Src: <BOS> I like football <EOS> <PAD> <PAD> <PAD> <PAD> <PAD>
Tgt: 我喜欢足球<EOS><PAD><PAD><PAD>
Pred: 我喜欢足球<EOS><EOS><EOS><EOS>
Probs: 1.0000,1.0000,0.9998,0.9996,0.9999,0.9996,0.9913,0.9997,0.9968
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
...(共20条测试集，略)...
Input_[5|5][3|4] >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
Src: <BOS> they clean letters <EOS> <PAD> <PAD> <PAD> <PAD> <PAD>
Tgt: 他们清理信件<EOS><PAD><PAD>
Pred: 他们整理信件<EOS><EOS><EOS>
Probs: 1.0000,0.9999,0.9895,1.0000,0.9991,0.8448,0.9999,0.9996,0.9998
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```

#### 手动输入验证：

```log
load checkpoint...
2026-03-24 23:01:57.946|INFO|modelMgmt.py:144|load_checkpoint -> checkpoint: ./saves/CheckPoint_Ep100_0.0464_0.0481.pth Loaded

Press send input: you buy movies
Next token: [你|1.000][买|0.000][踢|0.000]
Next token: [买|1.000][喝|0.000][你|0.000]
Next token: [电|0.656][买|0.281][<EOS>|0.041]
Next token: [影|0.856][<EOS>|0.108][电|0.016]
Next token: [<EOS>|1.000][件|0.000][影|0.000]

进程已结束，退出代码为 0
```

> 仅构建极简的<主谓宾>数据集，目的是学习&验证，非正式功能  
> 验证集及手动输入的验证句式，均未在训练集中出现


## 3. 训练 / Training

#### 训练相关信息：
- 训练数据集200条英-中对，测试集20条英-中对，均为<主谓宾>短句
- Batch Size设置为4，共训练100个epoch：
- 采用Adam优化器：lr=1e-4, weight_decay=5e-6
- 采用交叉熵损失为loss：nn.CrossEntropyLoss(ignore_index=PAD_ID)
- 通过硬回滚+软检测算法处理过拟合

#### Loss Dashboard：
![Loss_Dashboard](https://github.com/iciferdai/TransDemo/blob/main/pictures/Loss_Dashboard.png)

#### 最佳均衡loss：
- 训练集：0.0464
- 验证集：0.0481


## 4. 实现介绍 / Implementation Introduction
>  以下为概述性的介绍，具体实现请结合代码

#### 1.  手搓Transformer基本结构（模型组件）：
- 定义基础模型全局设置到base_params，采用全局静态变量，如：D_MODEL = 1024
- 实现核心点积函数：dot_att，mask由外层传递，在函数里计算
- 基于点积函数封装多头注意力类MultiHeadAttention(nn.Module)
- 实现位置函数封装到PosEncoding(nn.Module)，实际为固定算法，无需训练，采用register_buffer快速使用
- 实现FFN(nn.Module)，最基础的神经网络，激活函数采用原论文的ReLU
- 基于MLA和FFN封装EncoderLayer(nn.Module)：注意力->残差->LayerNorm->FFN->残差->LayerNorm，含dropout，完全透传mask
- 基于MLA和FFN封装DecoderLayer(nn.Module)：注意力->残差->LayerNorm->FFN->注意力(交叉)->残差->LayerNorm->FFN->残差->LayerNorm，含dropout，完全透传mask（2种）

#### 2.  手搓简化版Transformer模型：MyTransf(nn.Module)：
- Encoder采用3层EncoderLayer，输入x先经Embedding->叠加位置编码->三层EncoderLayer->输出，mask由外层传入
- Decoder采用3层DecoderLayer，输入x先经Embedding->叠加位置编码->三层DecoderLayer（+encoder输出作为输入）->输出，mask(2种)由外层传入
- 最终模型，输入src->Encoder->y；  输入tgt(自回归)+y->Decoder->输出o，o经Linear得到词表输出概率，mask(对应src和tgt)由外层传入

#### 3.  准备试验数据demo_data_train & demo_data_test，直接写入data_dict，采用格式：[(EN, CN),(EN, CN)…]
#### 4. 在tools中，离线处理demo_data_train & demo_data_test数据，简化分词处理逻辑（英文按空格分单词，中文逐字）；生成词表后补充到data_dict文件
#### 5.  在processData中实现数据处理相关任务：准备TensorDataset和DataLoader的函数，以及2种mask的生成函数
#### 6.  封装核心模型训推管理类class ModelManagement()：
- 管理训练epoch进度
- 管理loss算法：硬回滚+软检测
- 记录管理状态数据
- 训练过程loss曲线监控
- 支持保存/加载状态和Checkpoint，支持退出时保存，支持续训
- 处理模型/数据显示转移到训练设备
- 主训练循环：取数据-传数据-前向-计算loss-反向-优化器更新
- 推理（手动）：根据输入进行自回归推理
- 推理（自动）：对demo_data_test数据集进行遍历推理并输出对比结果

#### 7.  实现外层极简的训练调用控制：Main_Train，组装调用模型初始化、数据初始化、模型管理类初始化后，调用模型管理类初始化训练参数与loss看板、启动N个epoch的训练、及后继的状态保存。
#### 8.  实现外层极简的推理调用控制：Main_eval，组装调用模型初始化、数据初始化（手动时提示用户输入）、模型管理类初始化后，调用模型管理类初始化推理，调用相应的推理函数。


## 5. 过程总结 / Conclusion

按最初目标达成学习和试验的目的，仅需进行很少几个epoch的训练，把训练loss降到较低，然后做下推理即可；但实际引入了测试集提升了任务挑战，即模拟（注1）模型的泛化能力。

引入测试集挑战后难度大大提升，因为实际训练时极易过拟合，初期训练loss和验证loss接近，但几个epoch后训练loss快速下降甚至很快接近0，与验证loss快速拉开，甚至是数量级差异，因此正好实践一些泛化指标与算法。

具体本项目的实践总结为硬回滚+软检测，其实现均封装在了模型管理类中的loss_algorithm方法中，概述如下：

- **绝对比GAP（ = test_loss / train_loss ）判断 （硬回滚指标）：**
	- 限制一定的绝对比例，但比例极大，避免间接关联作弊，比如绝对比GAP数量级（本例中设置为10）差异
	- 本次绝对比GAP对比之前的对比GAP，容许一定的扩大，但不允许跳变，比如上次比值是2，本次为4，代表跳变

- **滑动窗口归一化后的绝对差（硬回滚+软检测）：**
	- 计算最近的MEAN_EPOCH个epoch的loss训练集平均值作为基数，不含本epoch，把loss按base归一化到同一数量级，计算偏差（相差的值及比例）
	- 设置2级指标，过大直接硬回滚，中等则消耗耐心值（注2）
	
- **背离（硬回滚+软检测）：**
	- 背离：泛化损失（Generalization Loss，GL），训练损失还在下降，但测试损失已经停止下降甚至上升
	- 设置2级指标，过大直接硬回滚，中等则消耗耐心值
	
- **最佳Checkpoint保存及回滚：**
	- 保存：记录最佳测试loss，本次loss比最佳测试loss低，则进入待保存判断，如果满足最小绝对GAP，则直接保存并刷新最佳loss，如果不满足，则先做上述软检测判断，软检测判断都通过则可以保存并刷新最佳loss
	- 回滚：回滚到上次保存的最佳loss点，并恢复耐心值

> 注1：如此小规模的模型以及数据集上，并非真正具备泛化能力，以上仅为模拟试验  
> 注2：耐心值，在模型管理类中预设，对于软检测不通过则消耗耐心值但不立刻回滚，直至耐心值消耗完才回滚

---
##  结束 / End
>:loudspeaker: Notice：  
>本项目为个人学习与实验性项目  
> This is personal learning and experimental project
