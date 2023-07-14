from pydub import AudioSegment
from funasr_onnx import Fsmn_vad

model_dir = "./fsmn-vad"
# model = Fsmn_vad(model_dir, quantize=True)
model = Fsmn_vad(model_dir, quantize=False)

wav_path = "./vad_example_ukrainian.wav"
wav = AudioSegment.from_wav(wav_path)

result = model(wav_path)

for x in result:
    for y in x:
        start, end = y
        
        print(start, end)

        chunk = wav[start:end]
        chunk.export(f"./chunks/{start}_{end}.wav", format="wav")
