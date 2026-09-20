from itn.chinese.inverse_normalizer import InverseNormalizer
import os
invnormalizer = InverseNormalizer(cache_dir=os.environ.get("ITN_CACHE_DIR", "/root/asr/WeTextProcessing/itn"))
result = invnormalizer.normalize("发行人零二零二三。测量时间。二零二四年七月二十七日。二零二三年八月三日。二零二一年五月十五日。")
print(result)