# Dynacat MD

## 中文描述

从给定的设置和Plumed输入生成结合机器学习势函数的增强采样MD任务。

## 英文描述

Generate enhanced sampling task from given MD settings as well as Plumed input.

## App详情页

### App介绍

Dynacat MD基于AI4EC联合实验室开发的催化领域全自动基元反应势函数训练工作流[ai<sup>2</sup>-cat](https://github.com/chenggroup/ai2-kit)中生成增强采样任务的流程，结合dflow-galaxy对dflow的灵活封装演进而来。
用户可以上传一个结构初始文件，推荐为 extxyz 或者 POSCAR 格式，即包含对应原子的元素、位置坐标和周期性设置的结构；以及一个DeePMD训练得到的势函数，用于后续MLMD模拟。然后设定需要运行增强任务和一些具体设置，如MD的时间步和步长、系综、Plumed输入文件等，并选择所需要的机型、镜像等，便可自助提交一个增强采样MD任务，并可结合输出的偏置、CV等进行后续分析。

### 使用手册

#### 参数说明

- `dry_run`: 选择是否仅生成输入文件、不真正提交计算任务
- `system_file` : 用于LAMMPS模拟的初始结构，需要包含元素种类、原子坐标、周期性设置，推荐使用 `extxyz` 或者 `POSCAR`
- `deepmd_model` : 用于MD模拟的机器学习势函数模型文件
- `ensemble` : MD模拟使用的系综或者热浴，默认为 `csvr` ，即采用NVT系综并选用CSVR热浴
- `temperature` : 设定的模拟温度，单位是K，默认是 330.0
- `pressure` : 设定的模拟压力，选择NPT系综时需要设置，否则请保持为-1
- `steps` : MD模拟的时间步数
- `step_size` : MD模拟的步长，单位是皮秒（ps），默认为 0.0005 ps，即 0.5 fs
- `sample_freq` : 输出轨迹的间隔，默认每10步输出一次
- `plumed_config` : MD模拟过程中使用的Plumed输入，包含广义坐标的设置、需要输出的信息、采用增强采样的方法等等。
- `extra_args`（不建议修改）: LAMMPS模拟需要的额外设定，例如 `tau_t`, `tau_p`, `time_const` 等。除非用户真正了解这些参数的含义，否则请保持默认值且不要删除。

#### 使用指南

请上传需要的初始结构及机器学习势函数，正确设置参数，即可提交。
