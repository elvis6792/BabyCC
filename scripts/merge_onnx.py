"""
scripts/merge_onnx.py
合併 asthma_monitor_final.onnx + .onnx.data 為單一檔，並驗證輸出一致。
執行時 .onnx 與 .onnx.data 必須在同一目錄。
"""
import numpy as np
import onnx
import onnxruntime as ort
from onnx.external_data_helper import convert_model_from_external_data

SRC = 'asthma_monitor_final.onnx'
DST = 'asthma_monitor_final_merged.onnx'

# 1. 載入圖 + 外部權重
model = onnx.load(SRC, load_external_data=True)

# 2. 權重嵌回圖內（移除 external data 引用）
convert_model_from_external_data(model)

# 3. 存單一檔
onnx.save(model, DST)
print(f'已合併儲存：{DST}')

# 4. 驗證：合併前後輸出必須一致
def input_shape(sess):
    d = sess.get_inputs()[0].shape
    return [s if isinstance(s, int) else 1 for s in d]

s_old = ort.InferenceSession(SRC)
s_new = ort.InferenceSession(DST)
x = np.random.RandomState(42).randn(*input_shape(s_old)).astype(np.float32)

o_old = s_old.run(None, {s_old.get_inputs()[0].name: x})
o_new = s_new.run(None, {s_new.get_inputs()[0].name: x})
for a, b, n in zip(o_old, o_new, ['wheeze_logit', 'cough_logit']):
    diff = float(np.max(np.abs(a - b)))
    print(f'{n}: max diff = {diff:.2e}')
    assert diff < 1e-5
print('驗證通過：合併前後輸出一致')


if __name__ == '__main__':
    pass