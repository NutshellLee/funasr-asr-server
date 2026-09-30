### 需求

* 需求背景

  离线和实时的语音识别、断句、添加标点符号，离线说话人区分、时间戳

* 需求能力列表或相关细节

  - [实时语音识别、时间戳] 能力1   

  - [离线语音识别、说话人区分、时间戳] 能力2   

  - [2pass实时语音识别] 能力3  


### 模型环境

* python3.8.12: [requirements](./requirements.txt)

```shell
# 创建虚拟环境后，下载虚拟环境里没有的资源库
pip install -r ./requirements.txt
```

### 模型说明

1. 数据说明

   * 测试数据集:    
   [aishell4-test](/home/share/dataset/base_data/asr/paraformer_related_datasets/aishell4-test)   
   aishell4的test数据集13小时，涉及20个大中小型会议室的多人讨论音频，有20种议题的讨论。 
   [cantonese_audio](/home/share/dataset/base_data/asr/paraformer_related_datasets/cantonese_audio)   
   cantonese_audio有六段我在网上合成的以及样例下载的音频，可用于参考粤语的效果，其中sample3和8k_canto为重复内容、不同的人讲的。
   [chinese_audio](/home/share/dataset/base_data/asr/paraformer_related_datasets/chinese_audio)   
   cantonese_audio有六段我在网上合成的以及样例下载的音频，可用于参考粤语的效果，其中sample3和8k_canto为重复内容、不同的人讲的。
   [ST-CMDS-20170001_1-OS](/home/share/dataset/base_data/asr/paraformer_related_datasets/ST-CMDS-20170001_1-OS)   
   ST-CMDS-20170001_1-OS是一个142小时的文学小说里面节选短句子的音频数据集，吐字清晰、在安静环境下录制，可作为benchmark重要参考之一。

   * 数据预处理: `无需单独预处理`

2. 模型来源

   paraformer系列的断句、加标点符号、实时、离线模型均来源于[modelscope魔搭社区框架paraformer](https://modelscope.cn/docs/)，   
   paraformer是2022年论文的自动无回归语音识别模型，在ASR领域里没有涉及断句和标点符号的功能，因此需要集成断句和符号模型实现，单纯ASR的准确率效果齐平主流sota的conformer，并且速度快10倍。   

   cam++也来源于[modelscope魔塔社区框架campplus](https://modelscope.cn/models/iic/speech_campplus_speaker-diarization_common/summary)，建议人数在10人以内，音频长度大于30秒，效果为佳。

   推理方式均可见modelscope或者funasr的[github开源项目首页](https://github.com/alibaba-damo-academy/FunASR)

   2pass模式的部署采用docker直接部署，勾选配置然后启动，详细方式见funasr的[2pass实时操作文档](https://github.com/alibaba-damo-academy/FunASR/blob/main/runtime/docs/SDK_advanced_guide_online_zh.md)

3. 权重说明

   - 以下模型均来源于modelscope官方文档提供的下载方式，链接：https://modelscope.cn/docs/%E6%A8%A1%E5%9E%8B%E7%9A%84%E4%B8%8B%E8%BD%BD
   - 模型集成包含如下权重，模型库列表页面在这里，链接：https://modelscope.cn/models?page=1   
   * punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727 实时读取文字，添加标点符号
   * punc_ct-transformer_zh-cn-common-vocab272727-pytorch 离线读取文字，添加标点符号
   * speech_fsmn_vad_zh-cn-16k-common-pytorch 离线实时通用的读取音频进行断句
   * speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch 离线的集成了ASR、断句、标点的多个模型
   * speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online 实时的ASR识别，适用于处理好断句的实时音频，推理后再添加标点符号
   * speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch 离线推理ASR纯文字的识别
   * speech_campplus_sv_zh-cn_16k-common 说话人确认模型
   * speech_timestamp_prediction-v1-16k-offline 给定文字和音频，预测对应信息的时间戳位置
   * speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch 支持热词的离线ASR推理
   

### 评估说明
   
1. 简易评测
   
   ```shell
   #评估方法需要在启动docker和fastapi服务之后

   cd FunASR/tests
   # 测试实时推理的效果，支持接收与发送分开进行、多文件格式、8k/16k采样率，模拟音频的实时等待播放
   python test_online.py 

   # 测试离线推理的效果
   python test_offline.py 

   # 测试2pass推理的效果
   python test_2pass.py 


2. 数据集评测
   
   ```shell
   cd FunASR/
   # 拷贝需要评测的数据集到项目根目录里
   sudo cp -r /home/share/dataset/base_data/asr/paraformer_related_datasets/aishell4-test ./additional    #该数据集较大，可以需要评测时再拷贝
   sudo cp -r /home/share/dataset/base_data/asr/paraformer_related_datasets/ST-CMDS-20170001_1-OS ./additional    #该数据集较大，可以需要评测时再拷贝

   # 运行ST-CMDS-20170001_1-OS数据集的推理评测脚本（老版本脚本），选取的是离线模式的加断句和标点的paraformer_large模型
   # 会生成包含推理结果的txt文件，叫“paraformer_punc_ST.txt”
   # 运行该数据集推理脚本指令如下：
   cd FunASR/tests
   python example_paraformer_large_ST.py

   # 调用API进行数据集推理评测：
   cd FunASR/tests
   python example_paraformer_large_API_ST.py
   ```

3. 自定义数据集
   ```
   # 选取ST-CMDS-20170001_1-OS推理脚本作为评测的示例，
   # 该数据集的格式简单，自定义数据集可参考：
   ├── ST-CMDS-20170001_1-OS
   │   ├── file
   │   │   ├── 1111111111111.txt
   │   │   └── 2222222222222.txt
   │   ├── wav
   │   │   ├── 1111111111111.wav
   │   │   └── 2222222222222.wav

   # wav文件规格是16kHz，单通道, 16bits位深。
   # txt文件规格是，每个文件一句话，不带空格、回车，不带标点符号。
   # txt文件名与wav文件名保持一致。
   ```

   * paraformer-large在aishell4测试集的CER错误率在0.16954566，推理总时长2653653ms
   * 集成断句和标点符号以后，CER在0.22595654，推理总时长5526023ms。

   结论（我的取舍）：

   * 加了断句和标点后 CER 从 0.170 升到 0.226，这不是识别变差，而是评测口径变了——标点符号也计入错误位，两行数字不能直接比较精度。
   * 代价在延迟：推理总耗时从 2654 s 涨到 5526 s，约 2.1 倍（按 aishell4-test 13 小时音频算，约从 17 倍实时降到 8.5 倍实时）。
   * 因此离线整段识别用「ASR + 断句 + 标点」；实时场景只跑纯 ASR，标点交给 2pass 的第二次离线推理补——这也是本项目同时保留离线 / 实时 / 2pass 三套接口的原因。


### Docker部署

1. 镜像构建

   ```shell
   # 从远端仓库拉取项目到本地的或者服务器的当前路径
   git clone -b main https://github.com/NutshellLee/funasr-asr-server.git
   cd funasr-asr-server
   # 模型权重与以下 4 个 wheel 未纳入仓库，构建镜像前请先下载放入本目录：
   #   torch-2.2.1+cu118-cp38-cp38-linux_x86_64.whl
   #   torchaudio-2.2.1+cu118-cp38-cp38-linux_x86_64.whl
   #   nvidia_cudnn_cu11-8.7.0.84-py3-none-manylinux1_x86_64.whl
   #   nvidia_cublas_cu11-11.11.3.6-py3-none-manylinux1_x86_64.whl

   # 执行构建docker镜像
   cd FunASR/docker/
   DOCKER_BUILDKIT=1 ./docker_build.sh
   # 镜像 funasr_docker:3.2 构建完毕
   ```

2. 启动容器服务

   ```shell
   #在执行构建docker镜像之后
   #再次进入docker文件夹
   cd docker
   #启动容器就自动后台运行api_server.py的服务，只使用CPU
   ./docker_cpu_run.sh 

   #启动容器就自动后台运行api_server.py服务，使用GPU
   ./docker_gpu_run.sh

   #启动后进入容器，
   ./docker_run.sh
   #或者用下述步骤重复进入容器

   #可以开启和执行container_id来重复进入已存在的容器环境
   #开启已经存在但没运行的container_id
   docker start xxxxxxxxxxxxx
   #已经在运行的容器直接用下述命令指定contianer_id进入
   docker exec -ti xxxxxxxxxxxxxxx bash
   ```

3. 简易服务开启

   ```
   # 如果没有使用docker自动开启服务在后台运行，手动开启服务命令如下：
   cd FunASR/
   python fast_api_server.py
   ```
