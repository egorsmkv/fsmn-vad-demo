# Demo of FSMN-VAD for Ukrainian

## Install requirements

```
conda create -n funasr python=3.8
conda activate funasr

pip3 install torch torchaudio

pip3 install -U modelscope
pip3 install modelscope[audio] -f https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html

pip3 install -U pydub
```

## Run inference scripts

Chinese sample (original):

```
python vad_inference_file.py
```

Ukrainian sample:

```
python vad_inference_file_uk.py
```

## ONNX

### Install requirements

```
pip3 install -U funasr_onnx
```

### Run inference scripts

Chinese sample (original):

```
python onnx_vad_inference_file.py
```

Ukrainian sample:

```
python onnx_vad_inference_file_uk.py
```
