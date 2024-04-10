# DynaCat TESLA

## 中文描述
使用预设的主动学习训练策略，自动训练适用于 DynaCat 研究的 DeepMD 势函数。

## 英文描述
DynaCat TESLA is a tool that automatically trains DeepMD potential for DynaCat research using predefined active learning training strategies.

## App 详情页

### App 介绍
TESLA 是一款执行 训练（Training）、结构探索（Exploration）、结构筛选（Screening）、数据标注（Labeling）的主动学习（Activate Learning）工作流。通过提供简单的初始结构和初始数据集，即可自动迭代完成函数的训练。

DynaCat TESLA 针对动态催化研究的特点预置了一组训练策略，并提供简洁的用户界面帮助用户快速上手势函数训练工作流。

### 用户手册

#### 准备
在使用前，用户需要自行准备以下数据：
1. 用于势函数训练的启动数据集，其格式为经过 deepmd/npy, 需要压缩为 tgz 或者 zip 格式。
2. 用于结构搜索的初始数据集，其格式需要为 extxyz (以.xyz为后缀)。
3. 用于执行 CP2K 标注的配置文件。
4. 用于执行 DeepMD 训练的配置文件，其格式为 JSON, 可参考 [DeepMD-kit](https://docs.deepmodeling.com/projects/deepmd/en/r2/train/index.html)

其中 1、2、3 可借助另一应用 CP2K Lightning 进行准备。

需要注意的是，如果手动编写 CP2K 配置文件，用户需要使用 `@include coord_n_cell.inc` 引入结构文件，例如

```
   &SUBSYS
      @include coord_n_cell.inc
      &KIND O
         BASIS_SET  DZVP-MOLOPT-SR-GTH
         POTENTIAL  GTH-PBE-q6
      &END
      &KIND H
         BASIS_SET  DZVP-MOLOPT-SR-GTH
         POTENTIAL  GTH-PBE-q1
      &END
   &END
```

`coord_n_cell.inc` 文件会在工作流中自动生成供用户引用。

相关数据文件的准备可借助 ai2-kit 命令行工具，详见 [ase tool](https://github.com/chenggroup/ai2-kit/blob/main/doc/manual/ase.md) and [dpdata tool](https://github.com/chenggroup/ai2-kit/blob/main/doc/manual/dpdata.md)


#### 作业提交
上述文件准备完毕后，即可到 App 页面根据说明提交作业。

需要注意的是，为了避免用户误启动作业导致机时浪费，默认 `Dry Run` 为被勾选状态，此时只会生成配置文件，不会真正提交作业。如要真正提交作业，请将 `Dry Run` 选项取消勾选。

目前由于平台限制部分输入选项介绍不清晰，使用时请可参考下方的图例。

![](img/dynacat_tesla_form_1.png)
![](img/dynacat_tesla_form_2.png)

