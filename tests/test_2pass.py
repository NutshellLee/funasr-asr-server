"""
  Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
  Reserved. MIT License  (https://opensource.org/licenses/MIT)
  
  2022-2023 by zhaomingwork@qq.com  
  Modifications by NutshellLee, 2024
  对传送参数作了删减
  修改了接口地址uri
"""

# pip install websocket-client
import ssl
from websocket import ABNF
from websocket import create_connection
from queue import Queue
import threading
import traceback
import json
import time
import numpy as np
from loguru import logger


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

            # self.msg_queue = Queue()  # used for recognized result text

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

                # self.msg_queue.put(msg)
                logger.info(f"{msg}")  # 在这里接收来自服务端返回的结果
        except Exception as e:
            print("client closed")

    # feed data to asr engine, wait_time means waiting for result until time out
    def feed_chunk(self, chunk, wait_time=0.01):
        try:
            self.websocket.send(chunk, ABNF.OPCODE_BINARY)
            # loop to check if there is a message, timeout in 0.01s把这段注释掉从而接收和发送分开来互不影响
            # while True:
            #     msg = self.msg_queue.get(timeout=wait_time)
            #     if self.msg_queue.empty():
            #         break

            # return msg
            return ""
        except:
            return ""

    def close(self, timeout=1):
        message = json.dumps({"is_speaking": False})
        self.websocket.send(message)
        # sleep for timeout seconds to wait for result
        time.sleep(timeout)
        msg = ""
        # while not self.msg_queue.empty():
        #     msg = self.msg_queue.get()

        self.websocket.close()
        # only return the last msg
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
    # create a recognizer
    rcg = Funasr_websocket_recognizer(host="0.0.0.0", port="8081", is_ssl=False)
    import time

    for j in range(1000):
        start_time = time.time()
        for i in range(chunk_num):

            beg = i * stride
            data = audio_bytes[beg : beg + stride]

            text = rcg.feed_chunk(data, wait_time=0.02)

            # if len(text) > 0:
            #     print("text", text)

            time.sleep(len(data) / 32000)  # 音频字节一共 16k * 2 = 32000

        end_time = time.time()
        total_elapsed_time = end_time - start_time

        print(f"模型推理耗时: {total_elapsed_time:.4f} 秒")

    # get last message
    text = rcg.close(timeout=3)
    print("text", text)
