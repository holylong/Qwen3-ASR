# 音频协议改为 int16

**日期**: 2026-05-07

**背景**: asr_server.py 此前要求客户端发送 raw float32 PCM 音频（每样本 4 字节）。大部分语音识别 API 和音频处理工具链默认使用 16bit（int16）PCM，导致外部客户端需要额外做 float32 转换，不够友好。

## 修改内容

### asr_server.py

**协议声明** (line 13)：
```
- BINARY: raw float32 PCM → raw int16 PCM
```

**音频解析** (line 294)：收到 int16 二进制帧后，服务端自行转为 float32 再送入模型：
```python
# 旧
pcm = np.frombuffer(data, dtype=np.float32).reshape(-1)

# 新
pcm = np.frombuffer(data, dtype=np.int16).astype(np.float32).reshape(-1) / 32768.0
```

### asr_client.py

**发送前转换** (lines 301, 309)：麦克风采集的 float32 在发送前转为 int16：
```python
# 旧
await ws.send(c.tobytes())
await ws.send(chunk.tobytes())

# 新
await ws.send((c * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
await ws.send((chunk * 32767).clip(-32768, 32767).astype(np.int16).tobytes())
```

## 数据流

```
mic (float32)
  → *32767 → clip → int16 → tobytes()
    → WebSocket binary frame
      → np.frombuffer(int16) → astype(float32) / 32768.0
        → Qwen3ASRModel (float32)
```

模型输入依旧是 float32，传输层改为 int16（带宽减半，兼容主流音频工具链）。
