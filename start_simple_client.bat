@REM python asr_client.py --url ws://39.105.10.158:8000/ws/asr 

python asr_client.py ^
  --url ws://10.184.60.127:8000/ws/asr ^
  --vad-threshold 0.015 ^
  --silence-duration 0.8 --min-speech-frames 1 --pre-roll-sec 0