@REM python asr_client.py --url ws://39.105.10.158:8000/ws/asr 

@REM python asr_client.py ^
@REM   --url ws://39.105.10.158:8000/ws/asr ^
@REM   --vad-threshold 0.015 ^
@REM   --silence-duration 0.8 --min-speech-frames 1 --pre-roll-sec 0 --two-pass

python asr_client.py --url ws://39.105.10.158:8000/ws/asr --vad-threshold 0.095 --silence-duration 0.8 --min-speech-frames 1 --pre-roll-sec 0 --two-pass