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

