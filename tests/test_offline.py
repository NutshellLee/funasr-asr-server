# import requests


# # 假设你有一个音频文件的路径
# audio_file_path = "1.m4a"

# files = {"audio_file": open(audio_file_path, "rb")}

# url = "http://localhost:8081/ai/asr/offline_inference"
# response = requests.post(url, files=files)

# if response:
#     print(response.json())
    
    
    
import requests
import time


audio_file_path = "16k.wav"

# 初始化总耗时
total_elapsed_time = 0

url = "http://localhost:8081/ai/asr/offline_inference"

start_time = time.time()


with open(audio_file_path, 'rb') as audio_file:
    files = {"audio_file": audio_file}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"请求失败，状态码：{response.status_code}")

end_time = time.time()
total_elapsed_time = end_time - start_time

print(f"模型推理耗时: {total_elapsed_time:.4f} 秒")
