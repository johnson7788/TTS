TTS是基于深度学习的text2speech解决方案。 它偏向于简单而不是复杂的大型模型，然而，它旨在达到最先进的结果。

目前，我们提出了两种模型架构，分别绘制在Tacotron和Tacotron2上。 特别是对于注意力模块，对初始体系结构有很多改进。

基于Tacotron的模型较小，并且针对更快的训练/推理，而基于Tacotron2的模型几乎大3倍，但通过使用神经声码器（WaveRNN，WaveNet等）可获得更好的结果。 注意选择适合您需求的正确体系结构。

到目前为止，根据我们的实验，与其他开源text2speech解决方案相比，TTS能够提供同等或更好的性能。 它也支持多种语言（英语，德语，中文等），性能几乎没有变化。

Tacotron 架构图

1. 下载模型
        从我们生成的TTS模型中选择一个模型。您可以听音频样本以帮助选择。如有疑问，请选择最新的，即列表中的最后一个。
https://github.com/mozilla/TTS/wiki/Released-Models

单击“ TTS型号”列中的链接。这将是一个下载页面，例如Google云端硬盘。下载编号最大的.tar文件，
并将其用作步骤2中的“checkpoint”文件。此外，下载config.json并将其用作“模型配置”。

2. Build the package with the server and model
Check out mozilla / TTS的源代码：   git clone git@github.com:mozilla/TTS.git
如果您在步骤1.1中选择了较旧的型号，则需要从“commit”列中签出相应的服务器源代码。

为构建创建一个python环境, cd TTS; virtualenv -p python3 . ; source bin/activate

运行 python setup.py bdist_wheel --checkpoint /path/to/checkpoint.tar --model_config /path/to/config.json

这将在dist /中创建一个新的WHL。 这是一个Python wheel 包，其中包含server和model。

cd ..

3.  Install and run the server

可选：使用以下命令创建一个新的python环境以运行服务器：virtualenv -p python3 tacotron2，而“ tacetron”是您可以选择的目录名称。 然后，进入新的python环境
cd tacotron2; source bin/activate, 注意：每次打开新控制台时，都需要调用source bin/activate。

安装您在步骤2中刚创建的服务器和模型程序包, pip install ../TTS/dist/TTS-0.0.1+4f61539-py3-none-any.whl (your filename will differ)

Run the server: python -m TTS.server.server

Open http://localhost:5002 in your browser.