# -*- coding: utf-8 -*-
# utils_function.py
# @Description
#   配置文件
# @created 2024-04-12T16:27:51.220Z+08:00
#
import os
import json
import base64
import io
import librosa
from fastapi.responses import JSONResponse
import ffmpeg
import librosa
import numpy as np


def executor_callback(one_done):
    one_done.result()


def executor_callback_file(file_path, uuid_str):
    def callback(one_done):
        info = one_done.result()

        with open(os.path.join(file_path, f"{uuid_str}.json"), "w") as f:
            json.dump(info, f, ensure_ascii=False, indent=4)

    return callback


def message_return(message, code, response=None):
    """
        接口调用信息返回体
    Args:
        message: 状态信息
        code: 状态码
        response: 相应信息

    Returns:

    """
    if response is not None:
        return json.dumps(
            {"statusCode": code, "message": message, "response": response},
            skipkeys=True,
            ensure_ascii=False,
            indent=4,
        )

    return json.dumps(
        {"statusCode": code, "message": message, "response": {}},
        skipkeys=True,
        ensure_ascii=False,
        indent=4,
    )


def read_base64(base64_data: str):
    """
        解析base64，转为语音bytes数据
    Args:
        base64_data: base64编码后的字符串

    Returns:
        librosa object
    """

    decoded_data = base64.b64decode(base64_data)
    speech = io.BytesIO(decoded_data)
    wav, s = librosa.load(speech, sr=16000)

    return wav


def read_wavfile(wav_file):
    """
        读取音频文件对象

    Args:
        wav_file : io.BufferedReader

    Returns:
        librosa object
    """

    speech = io.BytesIO(wav_file)
    wav, s = librosa.load(speech, sr=16000)

    return wav


def convert_audio_ffmpeg(input_path, output_path):
    (
        ffmpeg.input(input_path)
        .output(output_path, ar=16000, ac=1, format="wav")
        .overwrite_output()
        .run()
    )


def read_otherfile(other_file, file_type):
    """
        读取音频文件对象

    Args:
        wav_file : io.BufferedReader

    Returns:
        librosa object
    """
    wav_file_path = "temp.wav"
    new_file_path = "temp." + file_type

    with open(new_file_path, "wb") as new_file:
        new_file.write(other_file)

    convert_audio_ffmpeg(new_file_path, wav_file_path)

    wav, s = librosa.load(wav_file_path, sr=16000)

    return wav
