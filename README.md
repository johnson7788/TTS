<p align="center"><img src="https://user-images.githubusercontent.com/1402048/52643646-c2102980-2edd-11e9-8c37-b72f3c89a640.png" data-canonical-src="![TTS banner](https://user-images.githubusercontent.com/1402048/52643646-c2102980-2edd-11e9-8c37-b72f3c89a640.png =250x250)
" width="320" height="95" /></p>

<img src="https://travis-ci.org/mozilla/TTS.svg?branch=dev"/>
该项目是[Mozilla Common Voice]（https://voice.mozilla.org/en）的一部分。 TTS的目标是基于深度学习的Text2Speech引擎，其成本低且质量高。 首先，您可以从[此处]（https://soundcloud.com/user-565970875/commonvoice-loc-sens-attn）听到示例声音。

TTS包括两种基于[Tacotron]（https://arxiv.org/abs/1703.10135）和[Tacotron2]（https://arxiv.org/abs/1712.05884）的不同模型实现。 Tacotron体积更小，效率更高且更容易训练，但Tacotron2提供了更好的效果，尤其是与Neural声码器结合使用时。 因此，请根据您的项目要求进行选择。

如果您是新手，也可以在[这里]（http://www.erogol.com/text-speech-deep-learning-architectures/）上找到有关TTS体系结构及其比较的简短文章。

[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/0)](https://sourcerer.io/fame/erogol/erogol/TTS/links/0)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/1)](https://sourcerer.io/fame/erogol/erogol/TTS/links/1)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/2)](https://sourcerer.io/fame/erogol/erogol/TTS/links/2)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/3)](https://sourcerer.io/fame/erogol/erogol/TTS/links/3)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/4)](https://sourcerer.io/fame/erogol/erogol/TTS/links/4)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/5)](https://sourcerer.io/fame/erogol/erogol/TTS/links/5)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/6)](https://sourcerer.io/fame/erogol/erogol/TTS/links/6)[![](https://sourcerer.io/fame/erogol/erogol/TTS/images/7)](https://sourcerer.io/fame/erogol/erogol/TTS/links/7)

## TTS Performance 
<p align="center"><img src="https://camo.githubusercontent.com/9fa79f977015e55eb9ec7aa32045555f60d093d3/68747470733a2f2f646973636f757273652d706161732d70726f64756374696f6e2d636f6e74656e742e73332e6475616c737461636b2e75732d656173742d312e616d617a6f6e6177732e636f6d2f6f7074696d697a65642f33582f362f342f363432386639383065396563373531633234386535393134363038393566373838316165633063365f325f363930783339342e706e67"/></p>

[Details...](https://github.com/mozilla/TTS/wiki/Mean-Opinion-Score-Results)

## Features
-Torch和Tensorflow 2.0上的高性能Text2Speech模型。
-高性能扬声器编码器，可有效计算扬声器嵌入。
-与各种神经声码器（PWGAN，MelGAN，WaveRNN）集成
-发布预训练的模型。
-PyTorch的有效训练代码。 （很快用于Tensorflow 2.0）
-将Torch模型转换为Tensorflow 2.0的代码。
-有关console和Tensorboard的训练分析。
-用于在``dataset_analysis''下管理Text2Speech数据集的工具。
-用于模型测试的Demo server。
-用于广泛的模型基准测试的notebooks。
-模块化（但不太多）的代码库，可以轻松测试新想法。

## Requirements and Installation
Highly recommended to use [miniconda](https://conda.io/miniconda.html) for easier installation.
  * python>=3.6
  * pytorch>=0.4.1
  * librosa
  * tensorboard
  * tensorboardX
  * matplotlib
  * unidecode

使用```setup.py```安装TTS。 它将自动安装所有要求，并使TTS作为普通python模块可用于所有python环境。

```python setup.py develop```

或者您可以使用```requirements.txt```来仅安装需求。

```pip install -r requirements.txt```

### Docker
一个barebone的Dockerfile存在于项目的根目录下，这应该可以让您快速设置环境。 默认情况下，它将启动server并让您查询它。 确保使用`nvidia-docker`来使用您的GPU。 在构建映像之前，请确保遵循[`服务器自述文件]（server / README.md）中的说明，以便服务器可以在映像中找到模型。

```
docker build -t mozilla-tts .
nvidia-docker run -it --rm -p 5002:5002 mozilla-tts
```

## Checkpoints and Audio Samples
Please visit [our wiki.](https://github.com/mozilla/TTS/wiki/Released-Models)

## Example Model Outputs
在下面，您可以看到具有LJSpeech数据集的批次大小为32的16K迭代后的Tacotron模型状态。

> "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase the grey matter in the parts of the brain responsible for emotional regulation and learning."

Audio examples: [soundcloud](https://soundcloud.com/user-565970875/pocket-article-wavernn-and-tacotron2)

<img src="images/example_model_output.png?raw=true" alt="example_output" width="400"/>

## Runtime
最耗时的部分是在CPU上运行的声码器算法（Griffin-Lim）。 通过将其迭代次数设置得较低，您可能会更快地执行，而质量损失很小。 一些实验值如下。

Sentence: "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."

Audio length is approximately 6 secs.

| Time (secs) | System | # GL iters | Model
| ---- |:-------|:-----------| ---- |
|2.00|GTX1080Ti|30|Tacotron|
|3.01|GTX1080Ti|60|Tacotron|
|3.57|CPU|60|Tacotron|
|5.27|GTX1080Ti|60|Tacotron2|
|6.50|CPU|60|Tacotron2|


## Datasets and Data-Loading
TTS提供了易于用于新数据集的通用dataloder。 您需要编写一个预处理器函数来集成自己的数据集，检查``datasets / preprocess.py''以查看一些示例。 在该功能之后，您需要在“ config.json”中设置“ dataset”字段。 也不要忘记其他与数据相关的字段。

Some of the open-sourced datasets that we successfully applied TTS, are linked below.

- [LJ Speech](https://keithito.com/LJ-Speech-Dataset/)
- [Nancy](http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/)
- [TWEB](https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset)
- [M-AI-Labs](http://www.caito.de/2019/01/the-m-ailabs-speech-dataset/)
- [LibriTTS](https://openslr.org/60/)
- [Spanish](https://drive.google.com/file/d/1Sm_zyBo67XHkiFhcRSQ4YaHPYM0slO_e/view?usp=sharing) - thx! @carlfm01

## Training and Fine-tuning LJ-Speech
Here you can find a [CoLab](https://gist.github.com/erogol/97516ad65b44dbddb8cd694953187c5b) notebook for a hands-on example, training LJSpeech. Or you can manually follow the guideline below. 

首先，将“ metadata.csv”分为训练和验证子集“ metadata_train.csv”和“ metadata_val.csv”。 请注意，对于文本到语音，验证性能可能会产生误导，因为损耗值不会直接衡量人耳的语音质量，也不会衡量关注模块的性能。 因此，使用新句子运行模型并聆听结果是最好的方法。

```
shuf metadata.csv > metadata_shuf.csv
head -n 12000 metadata_shuf.csv > metadata_train.csv
tail -n 1100 metadata_shuf.csv > metadata_val.csv
```
要训练新模型，您需要定义自己的```config.json```文件（请参见示例）并使用以下命令进行调用。 您还可以在```config.json```中设置模型架构。

```train.py --config_path config.json```

To fine-tune a model, use ```--restore_path```.

```train.py --config_path config.json --restore_path /path/to/your/model.pth.tar```

对于多GPU训练，请使用```distribute.py```。 它启用了基于进程的多GPU培训，其中每个进程都使用一个GPU。

```CUDA_VISIBLE_DEVICES="0,1,4" distribute.py --config_path config.json```

每次运行都会创建一个新的输出文件夹，并在该文件夹下复制“ config.json”。
如果发生任何错误或执行被截获，如果输出文件夹下还没有检查点，则将删除整个文件夹。
如果将Tensorboard参数`--logdir`指向实验文件夹，您也可以使用Tensorboard。

## [Testing and Examples](https://github.com/mozilla/TTS/wiki/Examples-using-TTS)

## Contribution guidelines
This repository is governed by Mozilla's code of conduct and etiquette guidelines. For more details, please read the [Mozilla Community Participation Guidelines.](https://www.mozilla.org/about/governance/policies/participation/)

Please send your Pull Request to ```dev``` branch. Before making a Pull Request, check your changes for basic mistakes and style problems by using a linter. We have cardboardlinter setup in this repository, so for example, if you've made some changes and would like to run the linter on just the changed code, you can use the follow command:

```bash
pip install pylint cardboardlint
cardboardlinter --refspec master
```

## Collaborative Experimentation Guide
如果您想使用TTS尝试一个新的想法并希望与社区分享您的实验，我们敦促您使用以下指南进行更好的协作。
(If you have an idea for better collaboration, let us know)
- Create a new branch.
- Open an issue pointing your branch. 
- Explain your experiment.
- Share your results as you proceed. (Tensorboard log files, audio results, visuals etc.)
- Use LJSpeech dataset (for English) if you like to compare results with the released models. (It is the most open scalable dataset for quick experimentation)

## [Contact/Getting Help](https://github.com/mozilla/TTS/wiki/Contact-and-Getting-Help)

## Major TODOs
- [x] Implement the model.
- [x] Generate human-like speech on LJSpeech dataset.
- [x] Generate human-like speech on a different dataset (Nancy) (TWEB).
- [x] Train TTS with r=1 successfully.
- [x] Enable process based distributed training. Similar to (https://github.com/fastai/imagenet-fast/).
- [x] Adapting Neural Vocoder. TTS works with WaveRNN and ParallelWaveGAN (https://github.com/erogol/WaveRNN and https://github.com/erogol/ParallelWaveGAN)
- [ ] Multi-speaker embedding.
- [ ] Model optimization (model export, model pruning etc.)

<!--## References
- [Efficient Neural Audio Synthesis](https://arxiv.org/pdf/1802.08435.pdf)
- [Attention-Based models for speech recognition](https://arxiv.org/pdf/1506.07503.pdf)
- [Generating Sequences With Recurrent Neural Networks](https://arxiv.org/pdf/1308.0850.pdf)
- [Char2Wav: End-to-End Speech Synthesis](https://openreview.net/pdf?id=B1VWyySKx)
- [VoiceLoop: Voice Fitting and Synthesis via a Phonological Loop](https://arxiv.org/pdf/1707.06588.pdf)
- [WaveRNN](https://arxiv.org/pdf/1802.08435.pdf)
- [Faster WaveNet](https://arxiv.org/abs/1611.09482)
- [Parallel WaveNet](https://arxiv.org/abs/1711.10433)
-->

### References
- https://github.com/keithito/tacotron (Dataset pre-processing)
- https://github.com/r9y9/tacotron_pytorch (Initial Tacotron architecture)
