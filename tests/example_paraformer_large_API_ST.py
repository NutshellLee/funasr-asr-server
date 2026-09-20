from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import time
import os
import re
import requests
import base64
import json


url = "http://0.0.0.0:8083/ai/asr/offline_inference"


def edit_distance(a, b):
    dp_list = [
        [i if j == 0 else -1 for i in range(len(a) + 1)] for j in range(len(b) + 1)
    ]  # 2D list used to store the edit distance values for substrings of a and b.
    # first row represents the distance of transforming an empty string into substrings of a,
    # first colu represents the distance of transforming an empty string into substrings of b.
    for i in range(
        len(b) + 1
    ):  # The distance of transforming an empty string into substrings of b is simply the length of the substring (insertions).
        dp_list[i][0] = i

    for j in range(1, len(a) + 1):  # iterates over characters of a
        for i in range(1, len(b) + 1):  # iterates over characters of b.
            if (
                b[i - 1] == a[j - 1]
            ):  # if the characters at the current positions in a and b are the same, the edit distance remains the same as the previously calculated values.
                dp_list[i][j] = min(
                    dp_list[i - 1][j] + 1, dp_list[i][j - 1] + 1, dp_list[i - 1][j - 1]
                )
            else:  # if they are different, the minimum of the three possible operations (ins, del, and sub) is chosen, and 1 is added to it to update the current edit distance.
                dp_list[i][j] = min(
                    dp_list[i - 1][j] + 1,
                    dp_list[i][j - 1] + 1,
                    dp_list[i - 1][j - 1] + 1,
                )

    return dp_list[-1][
        -1
    ]  # returns the edit distance for the entire strings a and b, which is stored in [-1][-1]


log = open("paraformer_punc_ST.txt", "a")  # append mode

offline_inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model="../models_from_modelscope/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
)
errors = 0

duration = 0
folder = "../additional/ST-CMDS-20170001_1-OS/file"
filenames = [os.path.join(folder, file) for file in os.listdir(folder)]
# filenames = sorted(filenames, key=os.path.getmtime)
total = 0
lines = []
for path in filenames:
    # log.write(path+'\n')
    with open(path) as f:
        for line in f:
            words = list(line.rstrip())
            total += len(words)
            lines.append(line.rstrip())
log.write("total words = " + str(total) + "\n")
log.write("total lines = " + str(len(lines)) + "\n")
wav = "../additional/ST-CMDS-20170001_1-OS/wav"
wavs = [os.path.join(wav, file[:-4] + ".wav") for file in os.listdir(folder)]
for i in range(len(wavs)):
    # log.write(wavs[i]+'\n')
    start_time = time.perf_counter()
    data = base64.b64encode(open(wavs[i], "rb").read())
    data_form = {"wav_base64": data}
    response = requests.post(url, data=data_form)
    end_time = time.perf_counter()
    duration += end_time - start_time
    result = {}
    if response:
        result = json.loads(response.json()["response"])
        temp = re.sub(r"[^\w\s]", "", result["test_voice_result"])
        errors += edit_distance(temp, lines[i])
    else:
        errors += edit_distance("", lines[i])

    log.write(str(result) + " " + lines[i] + "\n")
log.write("duration:" + str(int(duration * 1000)) + "ms\n")
log.write("errors =" + str(errors) + "\n")
log.write("CER = " + str(errors / total) + "\n")
log.close()
