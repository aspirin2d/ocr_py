# Grounded

## 安装WSL系统（可选）
在[这里](https://learn.microsoft.com/en-us/windows/wsl/install)按照说明进行安装。PS. 因为有些依赖可能在windows上还存在冲突，最好是在linux系统上进行开发。

## 安装miniconda3和python
1. 在[这里](https://docs.anaconda.com/miniconda/miniconda-install/)按照说明进行安装。
1. 安装python：`conda install python`

## 安装项目
1. 下载项目： `git clone https://github.com/aspirin2d/ocr_py.git`，进入到对应的目录：`cd ocr_py`
1. 创建虚拟环境：`conda env create -f environment.yml`，并激活：`conda activate grounded`。注意：每次进入新的shell时都需要手动激活虚拟环境（默认的环境是'base'而不是'grounded'）。
1. 运行命令行：`python main.py detect -img "TEST_IMAGE.jpg" -prt "YOUR_DETECTION_PROMPT" -cls="YOUR_DETECTION_LABEL" -flt 2 -dbd`（具体使用说明输入`python main.py -h`查看）。
