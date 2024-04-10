# TESLA Lightning

## 中文描述

根据给定结构生成CP2K输入并执行基础AIMD产生初始数据集。

## 英文描述

Generate CP2K input file from given structure and run AIMD task to generate short trajectory as an initial training dataset.

## App详情页

### App介绍

TESLA Lightning基于AI4EC联合实验室开发的催化领域全自动基元反应势函数训练工作流[ai<sup>2</sup>-cat](https://github.com/chenggroup/ai2-kit)中生成CP2K AIMD任务的流程，结合dflow-galaxy对dflow的灵活封装演进而来。
用户可以上传一个结构初始文件，推荐为 `extxyz` 或者 `POSCAR` 格式，即包含对应原子的元素、位置坐标和周期性设置的结构，然后设定需要运行的CP2K任务和一些具体设置，如对角化算法、MD的时间步和步长、控温设置、基组、泛函等，并选择所需要的机型、镜像等，便可自助提交一个AIMD任务，用于后续研究。

### 使用手册

#### 参数说明

- `dry_run`: 选择是否仅生成输入文件、不真正提交计算任务。
- `system_file` : 用于AIMD模拟的初始结构，需要包含元素种类、原子坐标、周期性设置，推荐使用 `extxyz` 或者 `POSCAR`。 
- `system_type` : 根据带隙选择体系的类型及对应的对角化方法，若为半导体体系请选择 `semi` 并使用OT方法，若为金属体系则选择 `metal` 并使用DIIS方法。默认为 `metal`。
- `accuracy` : 选择DFT计算时的收敛精度要求，选项包括 `low` , `medium` 和 `high`，分别使用了不同大小的CUTOFF和收敛精度要求，通常精度越高需要的模拟时间越长。默认为 `metal`。
- `temperature` : 设定模拟时热浴的温度，单位是K，默认为 300.0。
- `steps` : 设定模拟的步数，默认为1000步。
- `timestep` : 设定模拟的时间步长，单位是飞秒（fs），默认为 0.5 fs。
- `basis_set` : 选择使用的基组文件，默认为 `BASIS_MOLOPT`。
- `potential` : 选择使用的赝势文件，默认为 `GTH_POTENTIALS`。

#### 使用指南

上传需要的结构信息文件，正确配置参数即可提交。
