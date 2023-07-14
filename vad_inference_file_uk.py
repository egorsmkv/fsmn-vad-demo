from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

inference_pipeline = pipeline(
    task=Tasks.voice_activity_detection,
    model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
    device='cuda:1',
)

segments_result = inference_pipeline(audio_in='vad_example_ukrainian.wav')

print(segments_result)
