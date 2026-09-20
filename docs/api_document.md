2pass、实时和离线语音识别、断句、加标点符号、说话人区分、时间戳接口



### 一、功能

实时和离线语音识别、断句、加标点符号接口
* 语音识别流式实时推理、时间戳
* 语音识别离线推理和说话人区分、时间戳
* 2pass实时语音识别


### 二、语音识别流式实时推理

1. 接口描述   
<br> 
用于发送bytes音频流切分片段然后实时推理断句，再做语音识别，再做添加标点符号，最后返回带断句的标点符号的文字结果。  
<br> 

2. 请求方法

Websockets方法： 
请求URL：`ws://0.0.0.0:8081/ai/asr/online_inference`   
 

发送请求时的数据类型:   

| 数据类型       |
| ------------ |
| json         |
| string       |

online_inference POST请求参数： 
<br>  
请求方式：    

```python

# 定义要作为URL参数传递的参数
query_params = {
  "sample_rate": 16000,  # 8000或者16000
  "bits_per_sample": 16,  # 这个会是2*8=16
  "channels": 1,  # 这个是1
}

# 将参数编码为查询字符串
query_string = urlencode(query_params)

# 构建包含查询字符串的WebSocket URL
url = f"ws://0.0.0.0:8081/ai/asr/online_inference?{query_string}"

data_json = {
"is_final": is_final,
}
data_json_str = json.dumps(data_json)

await websocket.send(data_json_str)
await websocket.send(speech_chunk)

```

POST参数说明：   

| 字符          | 必选 | 类型   | 说明       |
| ------------- | ---- | ---- | -------- |
| data_json_str | 是   | str | json信息存储了循环的索引，转成字符串 |
| speech_chunk | 是   | bytes | 切分成小段的音频流转成原生bytes |

<br>
<br>


1. 请求示例代码---Python： 
```python
import asyncio
import websockets
import json
import time
import wave
from urllib.parse import urlencode
from pydub import AudioSegment


# 定义发送音频数据的异步函数
async def send_audio_data_to_ws(websocket, speech_chunks):
    for i, speech_chunk in enumerate(speech_chunks[:-1]):
        is_final = i == len(speech_chunks) - 2
        data_json = {
            "is_final": is_final,
        }
        data_json_str = json.dumps(data_json)

        await asyncio.sleep(
            len(speech_chunk) / 32000
        )  # 16k采样率用 32000 或者 8k采样率用16000
        await websocket.send(data_json_str)
        await websocket.send(speech_chunk)


# 定义接收响应的异步函数
async def recv_responses(websocket):
    async for response in websocket:
        print(json.loads(response))


async def main():
    # 获取音频数据并切分成块
    audio_path = "16k.aac"  # 支持的文件格式：.wav .pcm .mp3 .aac
    if audio_path.endswith(".wav"):
        with wave.open(audio_path, "rb") as wav_file:
            params = wav_file.getparams()
            frames = wav_file.readframes(wav_file.getnframes())
            speech = bytes(frames)
    elif audio_path.endswith(".pcm"):
        with open(audio_path, "rb") as pcm_file:
            speech = pcm_file.read()
        params = wave._wave_params(
            1, 2, 16000, len(speech) // 2, "NONE", "not compressed"
        )
    else:
        audio = AudioSegment.from_file(audio_path, format=audio_path.split(".")[-1])
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        speech = audio.raw_data
        params = wave._wave_params(
            1, 2, 16000, len(speech) // 2, "NONE", "not compressed"
        )

    # 定义要作为URL参数传递的参数
    query_params = {
        "sample_rate": params.framerate,  # 8000或者16000
        "bits_per_sample": params.sampwidth * 8,  # 这个会是2*8=16
        "channels": params.nchannels,  # 这个是1
    }

    # 将参数编码为查询字符串
    query_string = urlencode(query_params)

    # 构建包含查询字符串的WebSocket URL
    url = f"ws://0.0.0.0:8081/ai/asr/online_inference?{query_string}"

    chunk_stride = 19200  # 8k 9600 ; 16k 19200
    speech_chunks = [
        speech[i * chunk_stride : (i + 1) * chunk_stride]
        for i in range((len(speech) - 1) // chunk_stride + 1)
    ]

    for _ in range(1000):  # 修改循环次数以适应你的需求
        start_time = time.time()
        async with websockets.connect(url) as websocket:
            # 创建发送和接收任务
            send_task = asyncio.create_task(
                send_audio_data_to_ws(websocket, speech_chunks)
            )
            recv_task = asyncio.create_task(recv_responses(websocket))

            # 等待发送任务完成
            await send_task
            print("send_task completed")

        end_time = time.time()
        total_elapsed_time = end_time - start_time

        print(f"模型推理耗时: {total_elapsed_time:.4f} 秒")

    # 确保所有任务都已完成
    await recv_task
    print("recv_task completed")


# 运行主函数
asyncio.run(main())

```
4. 返回结果示例

实时语音识别成功示例：
检测到语音：
```json
{
 'statusCode': 200, 
 'message': 'OK', 
 'response': '{
     "asr_status": "success", 
     "test_voice_result": "请问你想咨询的是信用卡还是借记卡服务呢？信用卡。请问您想咨询的是信用卡还是借记卡服务呢？信用卡。请问您想咨询的是信用卡还是借记卡服务呢？信用方。信用卡卡片激活完毕，请问您还有其他问题吗？卡片激活。",
     "duration": [0, 28200],
     "is_final": True
     }'
 }
```
未检测到语音示例：
```json
{
 'statusCode': 200, 
 'message': 'OK', 
 'response': '{
     "asr_status": "failed", 
     "test_voice_result": null,
     "duration": null,
     "is_final": null
     }'
 }
```

<br>
实时语音识别返回结果参数说明   

| 字段          | 类型     | 说明           |
| ------------- | -------- | -------------- |
| statusCode    | int      | 请求状态码     |
| message       | string   | 请求状态信息   |
| response      | string    | 响应结果的字符串 |
| +asr_status   | string | 推理是否成功 |
| +test_voice_result   | string | 语音返回的文字结果 |
| +duration   | list | 音频的时长 |

### 三、语音识别离线推理
1. 接口描述
<br> 
用于发送装有音频的filed对象，接口读取file文件对象，做离线推理语音识别、离线断句，然后离线添加标点符号，返回带断句的标点符号的文字结果，以及说话人区分和时间戳的结果。 
<br> 


2. 请求方法   

HTTP方法：POST   
请求URL：`http://IP:8081/ai/asr/offline_inference`   
  

Header 设置:   

| 参数         | 值                              |
| ------------ | ------------------------------- |
| Content-Type | multipart/form-data(请求方式一) |



offline_inference POST请求参数：
<br>  
请求方式：    

```python
files = {
    "audio_file": open(audio_file_path, "rb")
    }
```

POST参数说明：   

| 字符          | 必选 | 类型   | 说明       |
| ------------- | ---- | ------ | ---------- |
| audio_file | 是   | _io.BufferedReader | 读取音频文件后的文件对象 |
<br>


3. 请求示例代码---Python：   

```python
import requests


# 假设你有一个音频文件的路径
audio_file_path = "16k.wav"

files = {"audio_file": open(audio_file_path, "rb")}

url = "http://localhost:8081/ai/asr/offline_inference"
response = requests.post(url, files=files)

if response:
    print(response.json())

```


4. 返回结果示例

离线语音识别的成功示例：   
```json
{
    "statusCode": 200,
    "message": "OK",
    "response":
    "{
        \"asr_status\": \"success\", \"test_voice_result\": \"请问您想咨询的是信用卡还是借记卡服务呢？信用卡，请问您想咨询的是信用卡还是借记卡服务呢？信用卡，请问您想咨询的是信用卡还是借记卡服务呢？信用卡信用卡卡片激活完毕，请问您还有其他问题吗？卡片激活。\", \"timestamps\": [[50, 150], [150, 290], [290, 450], [450, 690], [710, 790], [790, 970], [970, 1090], [1090, 1230], [1230, 1390], [1390, 1550], [1550, 1790], [1930, 2090], [2090, 2290], [2290, 2390], [2390, 2550], [2550, 2770], [2770, 2910], [2910, 3150], [3150, 3415], [4140, 4280], [4280, 4500], [4500, 4805], [7000, 7180], [7180, 7320], [7320, 7480], [7480, 7720], [7740, 7820], [7820, 8000], [8000, 8120], [8120, 8260], [8260, 8420], [8420, 8580], [8580, 8820], [8920, 9120], [9120, 9260], [9260, 9420], [9420, 9520], [9520, 9760], [9800, 9940], [9940, 10120], [10120, 10445], [11200, 11440], [11440, 11600], [11600, 11875], [13970, 14110], [14110, 14250], [14250, 14410], [14410, 14650], [14670, 14750], [14750, 14930], [14930, 15050], [15050, 15190], [15190, 15350], [15350, 15510], [15510, 15750], [15870, 16050], [16050, 16250], [16250, 16350], [16350, 16510], [16510, 16730], [16730, 16870], [16870, 17110], [17110, 17405], [18110, 18330], [18330, 18550], [18550, 18885], [20970, 21150], [21150, 21330], [21330, 21570], [21610, 21750], [21750, 21990], [22010, 22150], [22150, 22290], [22290, 22450], [22450, 22690], [23170, 23330], [23330, 23450], [23450, 23690], [23750, 23870], [23870, 24010], [24010, 24150], [24150, 24310], [24310, 24450], [24450, 24630], [24630, 24925], [25670, 25850], [25850, 26090], [26090, 26230], [26230, 26645]], \"speaker_diarization_result\": [{\"text\": \"请问您想咨询的是信用卡还是借记卡服务呢？\", \"start\": 50, \"end\": 3415, \"timestamp\": [[50, 150], [150, 290], [290, 450], [450, 690], [710, 790], [790, 970], [970, 1090], [1090, 1230], [1230, 1390], [1390, 1550], [1550, 1790], [1930, 2090], [2090, 2290], [2290, 2390], [2390, 2550], [2550, 2770], [2770, 2910], [2910, 3150], [3150, 3415]], \"spk\": 0}, {\"text\": \"信用卡，\", \"start\": 3415, \"end\": 4805, \"timestamp\": [[4140, 4280], [4280, 4500], [4500, 4805]], \"spk\": 1}, {\"text\": \"请问您想咨询的是信用卡还是借记卡服务呢？\", \"start\": 4805, \"end\": 10445, \"timestamp\": [[7000, 7180], [7180, 7320], [7320, 7480], [7480, 7720], [7740, 7820], [7820, 8000], [8000, 8120], [8120, 8260], [8260, 8420], [8420, 8580], [8580, 8820], [8920, 9120], [9120, 9260], [9260, 9420], [9420, 9520], [9520, 9760], [9800, 9940], [9940, 10120], [10120, 10445]], \"spk\": 0}, {\"text\": \"信用卡，\", \"start\": 10445, \"end\": 11875, \"timestamp\": [[11200, 11440], [11440, 11600], [11600, 11875]], \"spk\": 1}, {\"text\": \"请问您想咨询的是信用卡还是借记卡服务呢？\", \"start\": 11875, \"end\": 17405, \"timestamp\": [[13970, 14110], [14110, 14250], [14250, 14410], [14410, 14650], [14670, 14750], [14750, 14930], [14930, 15050], [15050, 15190], [15190, 15350], [15350, 15510], [15510, 15750], [15870, 16050], [16050, 16250], [16250, 16350], [16350, 16510], [16510, 16730], [16730, 16870], [16870, 17110], [17110, 17405]], \"spk\": 0}, {\"text\": \"信用卡信用卡卡片激活完毕，\", \"start\": 17405, \"end\": 22690, \"timestamp\": [[18110, 18330], [18330, 18550], [18550, 18885], [20970, 21150], [21150, 21330], [21330, 21570], [21610, 21750], [21750, 21990], [22010, 22150], [22150, 22290], [22290, 22450], [22450, 22690]], \"spk\": 0}, {\"text\": \"请问您还有其他问题吗？\", \"start\": 22690, \"end\": 24925, \"timestamp\": [[23170, 23330], [23330, 23450], [23450, 23690], [23750, 23870], [23870, 24010], [24010, 24150], [24150, 24310], [24310, 24450], [24450, 24630], [24630, 24925]], \"spk\": 0}, {\"text\": \"卡片激活，\", \"start\": 24925, \"end\": 26645, \"timestamp\": [[25670, 25850], [25850, 26090], [26090, 26230], [26230, 26645]], \"spk\": 1}]
        }"
}
```
离线语音识别的未识别到示例：   
```json
{
    "statusCode": 200,
    "message": "OK",
    "response": "{\"asr_status\": \"failed\", \"test_voice_result\": null, \"timestamps\": null, \"speaker_diarization_result\": null}"
}
```

<br>
离线语音识别返回结果参数说明   

| 字段          | 类型     | 说明           |
| ------------- | -------- | ------------ |
| statusCode    | int      | 请求状态码     |
| message       | string   | 请求状态信息   |
| response      | string    | 响应数据的字符串 |
| +asr_status   | string | 语音识别是否有数据 |
| +test_voice_result   | string | 语音返回的文字结果以及每个字的时间戳 |
| +speaker_diarization_result   | list | 用列表输出了文段、起始、结束点、每个字的时间戳、说话人编号 |


### 四、2pass语音识别实时推理

1. 接口描述   
<br> 
用于发送bytes音频流切分片段然后先实时语音识别推理，再做断句、离线语音识别、添加标点符号，最后返回带断句的标点符号的文字结果。  
<br> 

2. 请求方法

Websockets方法： 
请求URL：`ws://0.0.0.0:8081/ai/asr/two_pass`   
 

发送请求时的数据类型:   

| 数据类型       |
| ------------ |
| json         |
| string       |

online_inference POST请求参数： 
<br>  
请求方式：    
```python
message = json.dumps(
    {
        "is_speaking": True,
    }
)

self.websocket.send(message)
self.websocket.send(chunk, ABNF.OPCODE_BINARY)
```

POST参数说明：   

| 字符          | 必选 | 类型   | 说明       |
| ------------- | ---- | ---- | -------- |
| message | 是   | str | json信息存储了循环的索引，转成字符串 |
| chunk | 是   | bytes | 切分成小段的音频流转成原生bytes |

<br>
<br>


1. 请求示例代码---Python： 
```python
import ssl
from websocket import ABNF
from websocket import create_connection
from queue import Queue
import threading
import traceback
import json
import time
import numpy as np


# class for recognizer in websocket
class Funasr_websocket_recognizer:
    """
    python asr recognizer lib

    """

    def __init__(
        self,
        host="0.0.0.0",
        port="8081",
        is_ssl=False,
    ):
        """
        host: server host ip
        port: server port
        is_ssl: True for wss protocal, False for ws
        """
        try:
            if is_ssl == True:
                ssl_context = ssl.SSLContext()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                uri = "wss://{}:{}/ai/asr/two_pass".format(host, port)
                ssl_opt = {"cert_reqs": ssl.CERT_NONE}
            else:
                uri = "ws://{}:{}/ai/asr/two_pass".format(host, port)
                ssl_context = None
                ssl_opt = None
            self.host = host
            self.port = port

            self.msg_queue = Queue()  # used for recognized result text

            print("connect to url", uri)
            self.websocket = create_connection(uri, ssl=ssl_context, sslopt=ssl_opt)

            self.thread_msg = threading.Thread(
                target=Funasr_websocket_recognizer.thread_rec_msg, args=(self,)
            )
            self.thread_msg.start()

            message = json.dumps(
                {
                    "is_speaking": True,
                }
            )

            self.websocket.send(message)

            print("send json", message)

        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()

    # threads for rev msg
    def thread_rec_msg(self):
        try:
            while True:
                msg = self.websocket.recv()
                if msg is None or len(msg) == 0:
                    continue
                msg = json.loads(msg)

                self.msg_queue.put(msg)
        except Exception as e:
            print("client closed")

    # feed data to asr engine, wait_time means waiting for result until time out
    def feed_chunk(self, chunk, wait_time=0.01):
        try:
            self.websocket.send(chunk, ABNF.OPCODE_BINARY)
            # loop to check if there is a message, timeout in 0.01s
            while True:
                msg = self.msg_queue.get(timeout=wait_time)
                if self.msg_queue.empty():
                    break

            return msg
        except:
            return ""

    def close(self, timeout=1):
        message = json.dumps({"is_speaking": False})
        self.websocket.send(message)
        # sleep for timeout seconds to wait for result
        time.sleep(timeout)
        msg = ""
        while not self.msg_queue.empty():
            msg = self.msg_queue.get()

        self.websocket.close()
        # only resturn the last msg
        return msg


if __name__ == "__main__":

    print("example for Funasr_websocket_recognizer")
    import wave

    wav_path = "16k.wav"
    with wave.open(wav_path, "rb") as wav_file:
        params = wav_file.getparams()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_bytes = bytes(frames)

    stride = int(60 * 10 / 10 / 1000 * 16000 * 2)
    chunk_num = (len(audio_bytes) - 1) // stride + 1
    # create an recognizer
    rcg = Funasr_websocket_recognizer(
        host="0.0.0.0", port="8081", is_ssl=False
    )
    # loop to send chunk
    for i in range(chunk_num):

        beg = i * stride
        data = audio_bytes[beg : beg + stride]

        text = rcg.feed_chunk(data, wait_time=0.02)
        if len(text) > 0:
            print("text", text)
        time.sleep(0.05)

    # get last message
    text = rcg.close(timeout=3)
    print("text", text)


```
4. 返回结果示例

实时语音识别成功示例：
检测到语音：
```json
{
    'mode': '2pass-online', 
    'text': '请问您想咨询的是信用卡还是借记卡服务呢', 
    'is_final': False,
    'duration': [0, 4200]
    }

{
    'mode': '2pass-offline', 
    'text': '请问您想咨询的是信用卡还是借记卡服务呢？', 
    'is_final': False,
    'duration': [0, 3610]
    }
```
未检测到语音示例：
```json
不返回结果
```

<br>
实时语音识别返回结果参数说明   

| 字段          | 类型     | 说明           |
| ------------- | -------- | -------------- |
| mode    | string      | 表示是实时还是离线推理  |
| text       | string   | 识别结果返回   |
| is_final      | bool    | 是否音频出现停顿 |



### 五、监控

1. 请求方法及参数

HTTP方法：POST

请求URL：`http://IP:8081/test/asr/survival`

​		

2. 请求示例代码---Python：

```shell
import requests
request_url = "http://IP:8081/test/asr/survival"

response = requests.post(request_url, timeout=200)
if response:
    print(response.json())

```



3. 返回结果示例

成功示例（若非这个，均为失败）

```json
{
 "statusCode": 200,
 "message": "success",
 "response": {}
}
```


返回结果参数说明

| 字段       | 类型   | 说明         |
| ---------- | ------ | ---------- |
| statusCode | int    | 请求状态码   |
| message    | string | 请求状态信息 |
| response   | dict  | 响应数据     |