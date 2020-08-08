TTS是基于深度学习的text2speech解决方案。 它偏向于简单而不是复杂的大型模型，然而，它旨在达到最先进的结果。

目前，我们提出了两种模型架构，分别绘制在Tacotron和Tacotron2上。 特别是对于注意力模块，对初始体系结构有很多改进。

基于Tacotron的模型较小，并且针对更快的训练/推理，而基于Tacotron2的模型几乎大3倍，但通过使用神经声码器（WaveRNN，WaveNet等）可获得更好的结果。 注意选择适合您需求的正确体系结构。

到目前为止，根据我们的实验，与其他开源text2speech解决方案相比，TTS能够提供同等或更好的性能。 它也支持多种语言（英语，德语，中文等），性能几乎没有变化。

Tacotron 架构图

中文数据集语料, 2.1GB

https://www.data-baker.com/open_source.html

1. 下载模型
        从我们生成的TTS模型中选择一个模型。您可以听音频样本以帮助选择。如有疑问，请选择最新的，即列表中的最后一个。
https://github.com/mozilla/TTS/wiki/Released-Models

单击“ TTS型号”列中的链接。这将是一个下载页面，例如Google云端硬盘。下载编号最大的.tar文件，
并将其用作步骤2中的“checkpoint”文件。此外，下载config.json并将其用作“模型配置”, 也要下载scale_stats.npy文件

2. Build the package with the server and model
Check out mozilla / TTS的源代码：   git clone git@github.com:mozilla/TTS.git
如果您在步骤1.1中选择了较旧的型号，则需要从“commit”列中签出相应的服务器源代码。

为构建创建一个python环境, cd TTS; virtualenv -p python3 . ; source bin/activate

安装espeak; brew install espeak

运行 python setup.py bdist_wheel --checkpoint /path/to/checkpoint.tar --model_config /path/to/config.json

这将在dist /中创建一个新的WHL。 这是一个Python wheel 包，其中包含server和model。
dist/
TTS-0.0.4+feaaa5a-py3-none-any.whl

cd ..

3.  Install and run the server

不用管的操作
可选操作：使用以下命令创建一个新的python环境以运行服务器：virtualenv -p python3 tacotron2，而“ tacetron”是您可以选择的目录名称。 然后，进入新的python环境
cd tacotron2; source bin/activate, 注意：每次打开新控制台时，都需要调用source bin/activate。

安装您在步骤2中刚创建的服务器和模型程序包,dist下生成的whl文件
pip install dist/TTS-0.0.4+feaaa5a-py3-none-any.whl 

Run the server: python -m TTS.server.server

Open http://localhost:5002 in your browser.


Dataset 数据集
什么使一个好的数据集

    剪辑和文本长度上的属于高斯分布。因此，如果片段长度很长，请绘制分布，并检查它是否覆盖了足够的短片段和长片段。
    没有错误。删除任何错误或损坏的文件。检查注释，比较transcript和音频长度。
    无噪音。背景噪声可能会导致您的模型难以学习特别好的对齐方式。即使它学习了对齐方式，最终结果也可能比您预期的要差得多。
    语音片段之间兼容的音调和音高。例如，如果您在项目中使用有声书录音，则该录音可能会模仿书中的不同字符。实例之间的这种差异会降低模型性能。
    良好的音素覆盖率。确保您的数据集覆盖了音素，双音素以及某些语言的三音素。根据您的使用情况，如果音素覆盖率较低，则该模型可能很难发音新颖的难听单词。
    录音的自然性。您的模型将学习数据集中的任何内容。因此，如果您希望在所有音调和音高差异（例如标点符号不同）下听到尽可能自然的声音，则您的数据集也应包含相似的属性
    

预处理数据集
如果您想使用直言不讳的数据集，则可能需要在训练之前执行两次质量检查。 
TTS提供了几个notebooks（CheckSpectrograms，AnalyzeDataset）来帮助您加快这一步。

AnalyzeDataset用于检查片段和转录本长度方面的数据集分布。最好找到异常值实例（太长，短文本但很长的语音剪辑等）并在训练前将其删除。请记住，我们希望在长短剪辑之间保持良好的平衡，以防止在转换时出现任何偏见。如果您只有短片（1-3秒），那么您的模型可能会在推理时间内遭受较长的句子；如果您的实例很长，则可能无法学习对齐方式，或者可能花费太长时间来训练模型。

CheckSpectrograms用于测量片段的噪声水平并找到良好的音频处理参数。
通过检查频谱图可以观察到噪声水平。如果频谱图看起来很杂乱，尤其是在无声部分，
则该数据集可能不是TTS项目的理想候选者。如果您的语音剪辑在背景中太嘈杂，
则会使模型更难于学习对齐方式，并且最终结果可能与您所给的语音有所不同。
如果频谱图看起来不错，则下一步是找到在config.json中定义的好的音频处理参数集。
在notebooks中，您可以比较不同的参数集，并查看与给定ground-truth相关的重新合成结果。
找到可以提供最佳综合性能的最佳参数。

另一个重要的实用细节是语音片段的量化。如果您的数据集的比特率很高，
则可能会导致数据加载时间变慢，从而导致训练速度变慢。最好在16000-22050附近降低数据集的采样率。

Setting up Dataloader

在训练之前，您需要确保数据加载器（TTSDataset.py）与数据集兼容。 通常，对于任何数据集就足够了，除非您需要考虑一些特定的事项。 然后，最好查看一下并进行必要的编辑。

如果加载的数据看起来不错，那么您需要为自己的数据集中的dataset / preprocess.py实现一个预处理器。 大多数开放数据集已经有一些示例预处理器。

设置config.json

config.json是有关模型和训练的所有内容的配置文件。 
在完成所有前面的步骤后，您需要在该文件中填写与数据集相关的参数。 
我们尝试使config.json尽可能具有描述性。 请遵循那里的注释以更好地了解参数。


How can I train my own model?
    Check your dataset with notebooks under dataset_analysis. Use this notebook to find the right audio processing parameters. The best parameters are the ones with the best GL synthesis.
    Write your own dataset formatter in datasets/preprocess.py or format your dataset as one of the supported datasets like LJSpeech.
        preprocessor parses the metadata file and converts a list of training samples.
    If you have a dataset with a different alphabet than English Latin, you need to add your alphabet in utils.text.symbols.
        If you use phonemes for training and your language is supported here, you don't need to do that.
    Write your own text cleaner in utils.text.cleaners. It is not always necessary to expect you have a different alphabet or language-specific requirements.
        This step is used to expand numbers, abbreviations and normalizing the text.
    Setup config.json for your dataset. Go over each parameter one by one and consider it regarding the commented explanation.
        'sample_rate', 'phoneme_language' (if phoneme enabled), 'output_path', 'datasets', 'text_cleaner' are the fields you need to edit in most of the cases.
    Write down your test sentences in a txt file as a sentence per line and set it in config.json test_sentences_file.
    Train your model.
        SingleGPU training: python train.py --config_path config.json
        MultiGPU training: CUDA_VISIBLE_DEVICES="0,1,2" python distribute.py --config_path config.json
            This command uses all the GPUs given in CUDA_VISIBLE_DEVICES. If you don't specify, it uses all the GPUs available.

如何训练自己的模型？
    使用数据集_分析下的笔记本检查数据集。使用此笔记本查找正确的音频处理参数。最佳参数是具有最佳GL合成的参数。
    在datasets / preprocess.py中编写自己的数据集格式化程序，或将数据集格式化为受支持的数据集之一，例如LJSpeech。
        预处理器解析元数据文件并转换训练样本列表。
    如果数据集的字母与英语拉丁字母不同，则需要在utils.text.symbols中添加字母。
        如果您使用音素进行训练并且此处支持您的语言，则无需这样做。
    在utils.text.cleaners中编写自己的文本清理器。不一定总是期望您对字母或语言有不同的要求。
        此步骤用于扩展数字，缩写和规范化文本。
    为数据集设置config.json。逐一遍历每个参数，并考虑有关注释的说明。
        在大多数情况下，“ sample_rate”，“ phoneme_language”（如果启用了音素），“ output_path”，“ datasets”，“ text_cleaner”是您需要编辑的字段。
    在txt文件中以每行一个句子的形式写下您的测试句子，然后在config.json test_sentences_file中进行设置。
    训练模型。
        SingleGPU培训：python train.py --config_path config.json
        MultiGPU培训：CUDA_VISIBLE_DEVICES =“ 0,1,2” python distribution.py --config_path config.json
            此命令使用CUDA_VISIBLE_DEVICES中给定的所有GPU。如果未指定，它将使用所有可用的GPU。