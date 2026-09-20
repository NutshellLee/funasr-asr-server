# from funasr import AutoModel
# import soundfile
# import numpy as np
# import time

# chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
# encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
# decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention

# model = AutoModel(model="../models_from_modelscope/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
#         model_revision="v2.0.4",
#         ngpu=1,
#         ncpu=4,
#         device='cuda',
#         disable_pbar=True,
#         disable_log=True,
#         disable_update=True)
# model_vad = AutoModel(model="../models_from_modelscope/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
#         model_revision="v2.0.4",
#         ngpu=1,
#         ncpu=4,
#         device='cuda',
#         disable_pbar=True,
#         disable_log=True,
#         disable_update=True
#                       )
# model_punc = AutoModel(model="../models_from_modelscope/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
#         model_revision="v2.0.4",
#         ngpu=1,
#         ncpu=4,
#         device='cuda',
#         disable_pbar=True,
#         disable_log=True,
#         disable_update=True)

# speech, sample_rate = soundfile.read('tests/16k.wav')

# chunk_stride = 4800  # 200ms  #chunk_size[1] * 960 # 600ms
# start_time = time.perf_counter()
# cache = {}
# cache_vad = {}
# accu_text = ''
# accu_time = [0, 0]
# temp_sentence = ''
# total_chunk_num = int(len((speech)-1)/chunk_stride+1)
# for i in range(total_chunk_num):
#     vad_speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
#     print(vad_speech_chunk[:10])
#     is_final = i == total_chunk_num - 1
#     res_vad = model_vad.generate(input=vad_speech_chunk, cache=cache_vad, is_final=is_final, chunk_size=200)
#     accu_time[1] = (i+1)*200
#     print('res_vad: ',res_vad)
#     if (i+1) % 2 == 0:
#         speech_chunk = speech[(i-1)*chunk_stride:(i+1)*chunk_stride]
#         res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
#         temp_sentence += res[0]['text']
#         print(temp_sentence)
#     if len(res_vad[0]['value']):
#         for pair in res_vad[0]['value']:
#             if pair[0] == -1:
#                 accu_time[1] = pair[1]
#                 if temp_sentence != '':
#                     res_punc = model_punc.generate(input=temp_sentence)
#                     accu_text += res_punc[0]['text']
#                 temp_sentence = ''
#     print(accu_text+temp_sentence)
#     print(accu_time)
# end_time = time.perf_counter()
# execution_time = end_time - start_time
# print(f"代码执行时间: {execution_time} �?)


from funasr import AutoModel

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="../models_from_modelscope/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        model_revision="v2.0.4",
        ngpu=1,
        ncpu=4,
        device='cuda',
        disable_pbar=True,
        disable_log=True,
        disable_update=True)

import soundfile
import os

wav_file = os.path.join("16k.wav")
speech, sample_rate = soundfile.read(wav_file)
chunk_stride = chunk_size[1] * 960 # 600ms

cache = {}
total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk, cache=cache, is_final=is_final, chunk_size=chunk_size, encoder_chunk_look_back=encoder_chunk_look_back, decoder_chunk_look_back=decoder_chunk_look_back)
    print(res)