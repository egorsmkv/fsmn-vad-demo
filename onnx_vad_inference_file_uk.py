from funasr_onnx import Fsmn_vad

model_dir = "./fsmn-vad"
# model = Fsmn_vad(model_dir, quantize=True)
model = Fsmn_vad(model_dir, quantize=False)

wav_path = "./vad_example_ukrainian.wav"

result = model(wav_path)

print(result)
