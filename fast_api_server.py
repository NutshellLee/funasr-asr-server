from fastapi import FastAPI, File, UploadFile
from fastapi.websockets import WebSocket, WebSocketDisconnect
import logging
from fastapi.middleware.cors import CORSMiddleware
from utils.utils_function import (
    message_return,
    read_wavfile,
    read_otherfile,
)
import json
import os
import uvicorn
import numpy as np
from urllib.parse import parse_qs
from itn.chinese.inverse_normalizer import InverseNormalizer

# Paths are configurable so the image can be built/run from any host directory.
# Defaults match the paths used inside the container.
MODELSCOPE_CACHE = os.environ.get("MODELSCOPE_CACHE", "/root/asr/models_from_modelscope")
ITN_CACHE_DIR = os.environ.get("ITN_CACHE_DIR", "/root/asr/WeTextProcessing/itn")


def model_path(name):
    return os.path.join(MODELSCOPE_CACHE, "iic", name)


invnormalizer = InverseNormalizer(cache_dir=ITN_CACHE_DIR)
from typing import Dict
from scipy.signal import resample
import traceback
import re


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("model loading")
from funasr import AutoModel

# 这是离线接口的
model_asr_contextual_vad_punc_sv = AutoModel(
    model=model_path("speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
    model_revision="v2.0.4",
    vad_model=model_path("speech_fsmn_vad_zh-cn-16k-common-pytorch"),
    vad_model_revision="v2.0.4",
    punc_model=model_path("punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
    punc_model_revision="v2.0.4",
    spk_model=model_path("speech_campplus_sv_zh-cn_16k-common"),
    spk_model_revision="v2.0.2",
    ngpu=1,
    ncpu=4,
    device="cuda",
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)

# 这是2pass服务的
model_asr = AutoModel(
    model=model_path("speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"),
    model_revision="v2.0.4",
    ngpu=1,
    ncpu=4,
    device="cuda",
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)
model_asr_stream = AutoModel(
    model=model_path("speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"),
    model_revision="v2.0.4",
    ngpu=1,
    ncpu=4,
    device="cuda",
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)
model_vad = AutoModel(
    model=model_path("speech_fsmn_vad_zh-cn-16k-common-pytorch"),
    model_revision="v2.0.4",
    ngpu=1,
    ncpu=4,
    device="cuda",
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)
model_punc = AutoModel(
    model=model_path("punc_ct-transformer_zh-cn-common-vocab272727-pytorch"),
    model_revision="v2.0.4",
    ngpu=1,
    ncpu=4,
    device="cuda",
    disable_pbar=True,
    disable_log=True,
    disable_update=True,
)

print("model loaded! only support one client at the same time now!!!!")


# 非实时接口：HTTP RESTful接口
@app.post("/ai/asr/offline_inference")
async def non_realtime_asr(
    audio_file: UploadFile = File(..., description="音频文件流")
) -> str:
    """ASR离线推理, 采用JSON提交数据

    Args:
        audio_file (UploadFile, optional): 音频文件流. Defaults to File(None, description="音频文件流").

    Returns:
        Dict: 结果格式
    """

    try:
        file_type = "wav"  # 先默认是wav
        file_types = {
            b"\x46\x4F\x52\x4D": "aiff",  # FORM for AIFF
            b"\x23\x21\x41\x4D\x52": "amr",  # #!AMR for AMR
            b"\x46\x4C\x41\x43": "flac",  # FLaC for FLAC
            b"\x4D\x54\x68\x64": "mid",  # MThd for MIDI
            b"\x49\x44\x33": "mp3",  # ID3 for MP3
            b"\x4F\x67\x67\x53": "ogg",  # OggS for OGG/Vorbis
            b"\x00\x00\x00\x1C": "mp4a",  # Common header start for MP4/AAC
            b"\x66\x74\x79\x70": "mp4",  # ftyp for MP4 container
            # Additional formats...
        }

        content = audio_file.file.read()
        header = content[:5]
        for type_literal, filetype in file_types.items():
            if header.startswith(type_literal):
                file_type = filetype
                break

        if file_type != "wav":
            data_info = read_otherfile(content, file_type)
        else:
            data_info = read_wavfile(content)

        offline_result = model_asr_contextual_vad_punc_sv.generate(
            input=data_info, batch_size_s=300, hotword="激活 信用"
        )

        asr_status = "success"
        asr_result = offline_result[0]["text"]
        timestamp = offline_result[0]["timestamp"]
        sv_result = offline_result[0]["sentence_info"]
    except Exception:
        asr_status = "failed"
        asr_result = None
        timestamp = None
        sv_result = None

    result = {
        "asr_status": asr_status,
        "test_voice_result": asr_result,
        "timestamps": timestamp,
        "speaker_diarization_result": sv_result,
    }

    return message_return("OK", 200, result)


@app.post("/test")
def get_server_status():
    """
    监控部分, 确保flask存活
    """

    return message_return("Success", 200)


connections: Dict[WebSocket, dict] = {}


async def online_inference_endpoint(websocket: WebSocket):
    await websocket.accept()

    # 从URL中提取参数
    query_params = parse_qs(websocket.url.query)
    sample_rate = int(query_params.get("sample_rate", [8000])[0])
    bits_per_sample = int(query_params.get("bits_per_sample", [16])[0])
    channels = int(query_params.get("channels", [1])[0])

    # 参数验证
    if sample_rate not in [8000, 16000]:
        await websocket.send_text(
            message_return("Failed", 400, {"error": "Invalid sample rate"})
        )
        await websocket.close()
        return
    if bits_per_sample != 16:
        await websocket.send_text(
            message_return("Failed", 400, {"error": "Invalid bits per sample"})
        )
        await websocket.close()
        return
    if channels != 1:
        await websocket.send_text(
            message_return("Failed", 400, {"error": "Invalid number of channels"})
        )
        await websocket.close()
        return

    connection_state = connections.setdefault(
        websocket,
        {
            "index": 0,
            "overall_text": "",
            "temp_sentence": "",
            "cache_asr": {},
            "cache_vad": {},
            "sample_rate": sample_rate,
            "bits_per_sample": bits_per_sample,
            "channels": channels,
        },
    )

    while True:
        try:
            data_json = await websocket.receive_json()
            chunk = await websocket.receive_bytes()

            is_final = bool(data_json["is_final"])

            connection_state["index"] += 1

            overall_duration = [0, 0]

            # 根据采样率信息进行处理
            if connection_state["sample_rate"] != 16000:
                audio_array = np.frombuffer(chunk, dtype=np.int16)
                # 计算新的长度
                new_length = int(
                    len(audio_array) * 16000 / connection_state["sample_rate"]
                )
                # 进行重采样（这里假设你有一个 resample 函数）
                audio_array = resample(audio_array, new_length)
                chunk = audio_array.astype(np.int16).tobytes()

            vad_size = 300
            res_vad = [[], []]
            vad_final = False
            for i in range(2):
                if is_final and i == 1:
                    vad_final = True
                temp_vad = model_vad.generate(
                    input=chunk[i * 9600 : (i + 1) * 9600],
                    cache=connection_state["cache_vad"],
                    is_final=vad_final,
                    chunk_size=vad_size,
                )
                if len(temp_vad[0]["value"]):
                    res_vad[i] = temp_vad[0]["value"]
            overall_duration[1] = connection_state["index"] * vad_size * 2
            print(len(chunk), chunk[:100])
            res = model_asr_stream.generate(
                input=chunk,
                cache=connection_state["cache_asr"],
                is_final=is_final,
                chunk_size=[0, 10, 5],  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )

            previous_sentence = connection_state["temp_sentence"]
            connection_state["temp_sentence"] += res[0]["text"]

            print(res_vad)
            print(res)

            if len(res_vad[0]):
                for pair in res_vad[0]:
                    if pair[0] == -1 and previous_sentence != "":
                        overall_duration[1] = pair[1]
                        res_punc = model_punc.generate(input=previous_sentence)
                        connection_state["overall_text"] += res_punc[0]["text"]
                        connection_state["temp_sentence"] = res[0]["text"]
                        break
            if len(res_vad[1]):
                for pair in res_vad[1]:
                    if pair[0] == -1 and connection_state["temp_sentence"] != "":
                        overall_duration[1] = pair[1]
                        res_punc = model_punc.generate(
                            input=connection_state["temp_sentence"]
                        )
                        connection_state["overall_text"] += res_punc[0]["text"]
                        connection_state["temp_sentence"] = ""
                        break
            if not is_final:
                asr_result = (
                    connection_state["overall_text"] + connection_state["temp_sentence"]
                )
            else:
                asr_result = connection_state["overall_text"]

            asr_result = asr_result[-10000:]
            asr_status = "success"
            result = {
                "asr_status": asr_status,
                "test_voice_result": asr_result,
                "duration": overall_duration,
                "is_final": is_final,
            }

            await websocket.send_text(message_return("OK", 200, result))
        except WebSocketDisconnect:
            # 捕获到断开连接事件，跳出循环
            connections.pop(websocket, None)
            break
        except Exception as e:
            connections.pop(websocket, None)
            asr_status = "failed"
            result = {
                "asr_status": asr_status,
                "test_voice_result": None,
                "duration": None,
                "is_final": None,
            }

            await websocket.send_text(message_return("OK", 200, result))
            print(f"Error: {e}\n{traceback.format_exc()}")


async def two_pass_endpoint(websocket: WebSocket):
    await websocket.accept()

    print("new user connected", flush=True)

    connection_state = connections.setdefault(
        websocket,
        {
            "frames": [],
            "frames_asr": [],
            "frames_asr_online": [],
            "vad_pre_idx": 0,
            "speech_start": False,
            "speech_end_i": -1,
            "status_dict_asr": {},
            "status_dict_asr_online": {"cache": {}, "is_final": False},
            "status_dict_vad": {"cache": {}, "is_final": False},
            "status_dict_punc": {"cache": {}},
            "text_print": "",
            "text_print_2pass_online": "",
            "duration": [0, 0],  # 用于记录音频段的时长
        },
    )

    while True:
        try:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if message["type"] == "websocket.receive":
                if "text" in message:
                    messagejson = json.loads(message["text"])

                    if "is_speaking" in messagejson:
                        websocket.is_speaking = messagejson["is_speaking"]
                        connection_state["status_dict_asr_online"][
                            "is_final"
                        ] = not websocket.is_speaking

                elif "bytes" in message:
                    binary_message = message["bytes"]
                    connection_state["frames"].append(binary_message)
                    duration_ms = len(binary_message) // 32  # 每次接收到60ms的音频段
                    connection_state["vad_pre_idx"] += duration_ms

                    # asr online
                    connection_state["frames_asr_online"].append(binary_message)
                    connection_state["status_dict_asr_online"]["is_final"] = (
                        connection_state["speech_end_i"] != -1
                    )
                    if (
                        len(connection_state["frames_asr_online"]) % 10 == 0
                        or connection_state["status_dict_asr_online"]["is_final"]
                    ):
                        audio_in = b"".join(connection_state["frames_asr_online"])

                        connection_state["duration"][1] += len(audio_in) // 32
                        try:
                            await async_asr_online(
                                websocket, connection_state, audio_in
                            )
                        except Exception as e:
                            print(
                                f"error in asr streaming, {connection_state['status_dict_asr_online']}, {e}"
                            )
                        connection_state["frames_asr_online"] = []

                    if connection_state["speech_start"]:
                        connection_state["frames_asr"].append(binary_message)

                    # vad online
                    try:
                        speech_start_i, connection_state["speech_end_i"] = (
                            await async_vad(connection_state, binary_message)
                        )
                    except Exception as e:
                        print(f"error in vad, {e}")
                    if speech_start_i != -1:
                        connection_state["speech_start"] = True
                        connection_state["duration"][0] = speech_start_i
                        beg_bias = (
                            connection_state["vad_pre_idx"] - speech_start_i
                        ) // duration_ms
                        frames_pre = connection_state["frames"][-beg_bias:]
                        connection_state["frames_asr"] = []
                        connection_state["frames_asr"].extend(frames_pre)

                # asr punc offline
                if connection_state["speech_end_i"] != -1 or not websocket.is_speaking:
                    audio_in = b"".join(connection_state["frames_asr"])
                    connection_state["duration"][1] = connection_state["speech_end_i"]
                    try:
                        await async_asr(websocket, connection_state, audio_in)
                    except Exception as e:
                        print(f"error in asr offline, {e}")
                    connection_state["frames_asr"] = []
                    connection_state["speech_start"] = False
                    connection_state["frames_asr_online"] = []
                    connection_state["status_dict_asr_online"]["cache"] = {}

                    if not websocket.is_speaking:
                        connection_state["vad_pre_idx"] = 0
                        connection_state["frames"] = []
                        connection_state["status_dict_vad"]["cache"] = {}
                    else:
                        connection_state["frames"] = connection_state["frames"][-20:]

        except WebSocketDisconnect:
            print("Connection closed")
            break


async def replace_chars_with_symbols(text):
    replacements = {
        "杠": "-",
        "一": "1",
        "二": "2",
        "三": "3",
        "四": "4",
        "五": "5",
        "六": "6",
        "七": "7",
        "八": "8",
        "九": "9",
        "零": "0",
        ". ": "。",
        " ": "",
    }
    text = text.upper()
    # 遍历字典中的键值对，对文本中的字符进行替换
    for original_char, symbol in replacements.items():
        text = text.replace(original_char, symbol)
    text = re.sub(r"(\d)\s+(\d)", r"\1\2", text)  # 数字之间的空格去掉
    text = re.sub(r"(?<!\d)\.(?!\d)", "。", text)  # 非小数点的英文句号，转成中文句号

    return text


async def message(websocket, connection_state, text, mode):
    try:
        if mode == "2pass-online":
            connection_state["text_print_2pass_online"] += text
            connection_state["text_print"] = connection_state["text_print_2pass_online"]
        else:
            connection_state["text_print_2pass_online"] = ""
            connection_state["text_print"] = text

        connection_state["text_print"] = connection_state["text_print"][
            -10000:
        ]  # 防止过长
        connection_state["text_print"] = invnormalizer.normalize(
            connection_state["text_print"]
        )
        connection_state["text_print"] = await replace_chars_with_symbols(
            connection_state["text_print"]
        )
        print("\rpid" + ": " + connection_state["text_print"])

        message = json.dumps(
            {
                "mode": mode,
                "text": connection_state["text_print"],
                "is_final": not websocket.is_speaking,
                "duration": connection_state["duration"],
            }
        )
        await websocket.send_text(message)
    except Exception as e:
        print(f"error in message method inside, {e}")


async def async_asr_online(websocket, connection_state, audio_in):
    if len(audio_in) > 0:
        rec_result = model_asr_stream.generate(
            input=audio_in,
            cache=connection_state["status_dict_asr_online"]["cache"],
            is_final=connection_state["status_dict_asr_online"]["is_final"],
            chunk_size=[5, 10, 5],
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=0,
        )[0]

        if connection_state["status_dict_asr_online"].get("is_final", False):
            return

        if len(rec_result["text"]):
            print("online: ", rec_result)
            await message(
                websocket, connection_state, rec_result["text"], "2pass-online"
            )


async def async_vad(connection_state, audio_in):

    segments_result = model_vad.generate(
        input=audio_in,
        cache=connection_state["status_dict_vad"]["cache"],
        is_final=connection_state["status_dict_vad"]["is_final"],
        chunk_size=60,
    )[0]["value"]

    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


async def async_asr(websocket, connection_state, audio_in):
    if len(audio_in) > 0:

        rec_result = model_asr.generate(input=audio_in, hotword="魔搭社区 魔搭")[0]

        if len(rec_result["text"]) > 0:

            rec_result = model_punc.generate(
                input=rec_result["text"],
                cache=connection_state["status_dict_punc"]["cache"],
            )[0]
        if len(rec_result["text"]) > 0:
            await message(
                websocket, connection_state, rec_result["text"], "2pass-offline"
            )


# 注册WebSocket路由
app.websocket("/ai/asr/online_inference")(online_inference_endpoint)
app.websocket("/ai/asr/two_pass")(two_pass_endpoint)

if __name__ == "__main__":
    uvicorn.run("fast_api_server:app", host="0.0.0.0", port=8081)
