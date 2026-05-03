# 在线实时语音识别服务 — 修改记录

> Date: 2026-05-03
> Project: Qwen3-ASR
> Branch: master

## 概述

基于 Qwen3-ASR vLLM backend 实现了完整的在线语音识别服务，包括：

- **WebSocket 服务端** — 支持流式和 2-pass 两种模式
- **Python 麦克风客户端** — 实时采集音频、VAD 端点检测、终端实时显示
- **ModelScope 模型下载支持** — 可直接从 ModelScope 拉取模型

## 新增文件

| 文件 | 行数 | 说明 |
|---|---|---|
| `asr_server.py` | 379 | WebSocket ASR 服务端（FastAPI + vLLM） |
| `asr_client.py` | 375 | Python 麦克风客户端（sounddevice + websockets） |
| `start_server.sh` | 59 | 一键启动脚本（支持 HF / ModelScope） |
| `download_modelscope.sh` | 29 | ModelScope 模型批量下载脚本 |

## 设计决策

### 1. 通信协议：WebSocket

选择 WebSocket 而非 HTTP REST 的原因：

- **低延迟**：单个长连接，无需每次 HTTP 握手开销
- **全双工**：服务端可主动推送流式中间结果，无需客户端轮询
- **二进制传输**：PCM 音频以 raw bytes 传输，无序列化开销

已有代码 `qwen_asr/cli/demo_streaming.py` 使用 Flask REST API（`/api/start`, `/api/chunk`, `/api/finish`），每次 chunk 都是一次 HTTP 请求。对于实时麦克风场景，WebSocket 更适合。

### 2. Single-Pass vs 2-Pass 模式

**结论：默认推荐 streaming（单 pass）；对精度要求极高时启用 2-pass。**

分析依据：

- Qwen3-ASR 架构是 encoder-decoder（非 CTC/Transducer），流式模式每帧都重新编码**全部**累积音频（full-context re-encode）
- 流式质量损失主要来自 prefix rollback 策略的边界误差，而非模型能力损失
- README benchmark 显示流式 WER 与离线 WER 差距很小

| 特性 | Single-Pass (streaming) | 2-Pass (streaming + offline) |
|---|---|---|
| 延迟 | 实时（~500ms 字级） | 实时 + 1-2s 离线修正 |
| 精度 | 已接近离线 | 最高（消除 prefix 边界误差） |
| max_new_tokens | 受 prefix 约束，自然短 | 256，完整生成 |
| 适用场景 | 实时字幕、对话 | 会议纪要、精准记录 |

2-Pass 实现逻辑（`asr_server.py:247-282`）：

```
Pass 1: streaming_transcribe (prefix rollback, fast)
   ↓
Pass 2: transcribe() on accumulated audio (offline, full attention, no prefix)
```

### 3. VAD 端点检测

客户端使用**基于能量（RMS）的简单 VAD**，无额外依赖：

- `VAD_THRESHOLD = 0.02` — 高于此阈值视为语音帧
- `MIN_SPEECH_FRAMES = 4`（2 秒）— 需连续语音帧才开始识别，避免噪音误触发
- `SILENCE_DURATION_SEC = 0.8` — 连续静音 0.8 秒后自动结束当前语音段

每段语音自动在服务端创建新 session，语音段之间互不影响。

### 4. 服务端会话管理

- 基于 `uuid4` 的 session ID
- TTL = 10 分钟自动回收
- `ThreadPoolExecutor(max_workers=4)` 处理阻塞的 vLLM 推理调用
- 每个 session 维护独立的 `ASRStreamingState`

## WebSocket 协议定义

### 客户端 → 服务端

| 消息类型 | 格式 | 说明 |
|---|---|---|
| 开始会话 | `{"type": "start", "mode": "streaming\|two-pass"}` | 文本（JSON） |
| 音频块 | `float32 PCM16 raw bytes` | 二进制 |
| 结束语音 | `{"type": "finish"}` | 文本（JSON） |

### 服务端 → 客户端

| 消息类型 | 格式 | 说明 |
|---|---|---|
| 会话已创建 | `{"type": "started", "session_id": "...", "mode": "..."}` | 文本（JSON） |
| 识别结果 | `{"type": "result", "language": "English", "text": "...", "is_partial": true\|false, "pass": 1\|2}` | 文本（JSON） |
| 错误 | `{"type": "error", "message": "..."}` | 文本（JSON） |

## 使用方式

### 前置依赖

```bash
pip install qwen-asr[vllm] fastapi uvicorn         # 服务端
pip install sounddevice websockets numpy            # 客户端
pip install modelscope                              # ModelScope（可选）
```

### 从 HuggingFace 启动

```bash
# 服务端
python asr_server.py --asr-model-path Qwen/Qwen3-ASR-1.7B --port 8000

# 客户端（单 pass）
python asr_client.py --url ws://localhost:8000/ws/asr

# 客户端（2-pass）
python asr_client.py --url ws://localhost:8000/ws/asr --two-pass
```

### 从 ModelScope 启动

```bash
# 方式 A：先下载，再启动
./download_modelscope.sh
python asr_server.py --asr-model-path ./models/Qwen3-ASR-1.7B --port 8000

# 方式 B：启动时自动下载
python asr_server.py --asr-model-path Qwen/Qwen3-ASR-1.7B --use-modelscope --port 8000

# 方式 C：脚本启动
./start_server.sh ms
```

### 便捷脚本

| 脚本 | 用途 |
|---|---|
| `./start_server.sh` | 默认启动（HF）
| `./start_server.sh ms` | ModelScope 启动
| `MODEL=X PORT=Y ./start_server.sh ms` | 自定义参数启动
| `./download_modelscope.sh` | 批量下载所有模型到 `./models/`

## 服务端参数说明

```
--asr-model-path         模型名称或本地路径 (默认: Qwen/Qwen3-ASR-1.7B)
--use-modelscope         从 ModelScope 下载模型
--modelscope-cache-dir   ModelScope 缓存目录
--host                   绑定 IP (默认: 0.0.0.0)
--port                   监听端口 (默认: 8000)
--gpu-memory-utilization GPU 显存利用率 (默认: 0.8)
--max-new-tokens         最大生成 token 数 (默认: 256)
--unfixed-chunk-num      前 N 个 chunk 不使用 prefix (默认: 2)
--unfixed-token-num      prefix 回退 token 数 (默认: 5)
--chunk-size-sec         chunk 大小/秒 (默认: 1.0)
--use-modelscope         从 ModelScope 下载模型

## Bugfix (2026-05-03)

### 问题：客户端连上服务器但说话无文本返回

**根本原因分析：**

1. **`drain_audio` 错误合并音频块**
   - 旧实现把所有 callback 累积的 250ms 块 concat 成一个大 chunk
   - VAD 对大 chunk 只计为 1 帧，导致 `MIN_SPEECH_FRAMES=4` 永远达不到
   - 修复：`drain_one_chunk()` 每次只返回一个 250ms 块

2. **VAD 预热时间过长**
   - 旧 `MIN_SPEECH_FRAMES=4` × 500ms = 2 秒预热 → 短测试语无法触发
   - 修复：降为 3 × 250ms = 0.75s

3. **`read_server_results` timeout 过短**
   - 旧实现每次只等 50ms，vLLM 推理（200-500ms）时结果被丢弃
   - 修复：主循环每次迭代都 drain 服务端消息，不依赖单次 send 后的短超时

4. **客户端 chunk 频率太低**
   - 旧 CHUNK_DURATION=0.5s → 延迟高
   - 修复：降为 0.25s，发送更频繁

### 调式方式

```bash
# 服务端 — 查看日志（加 --debug）
python asr_server.py --asr-model-path ./models/Qwen3-ASR-1.7B --port 8000 --debug

# 客户端 — 查看 VAD RMS 值（加 --verbose）
python asr_client.py --url ws://SERVER_IP:8000/ws/asr --verbose
```

`--verbose` 输出示例：
```
VAD: rms=0.0345 thr=0.015 speech=True state=idle sf=2/3
VAD: rms=0.0281 thr=0.015 speech=True state=idle sf=3/3
VAD: rms=0.0192 thr=0.015 speech=True state=speaking sf=3/3
```

如果 RMS 值始终低于 threshold，调低 `--vad-threshold`（尝试 0.005 或 0.01）。

### 问题：丢前两个字/单词

**根本原因：VAD 预热延迟导致前端削波（front-end clipping）**

- VAD 需要 3 帧连续语音（0.75s）才触发 SPEAKING 状态
- 这 0.75s 内的语音帧只进了 pre_roll 缓冲区，未发送给服务端
- 修复：增加 **pre-roll buffer**（预滚动缓冲），VAD 触发时将缓冲区的预热帧一次性发送

工作原理：
```
帧0(sil) 帧1(sil) 帧2(sp1) 帧3(sp2) 帧4(sp3)←VAD触发  帧5(sp4)
  │        │         │        │         │              │
  └──pre-roll 缓冲: [sil,sil,sp1,sp2]────┘flush        │
                                       发送 sp3 ──────→│ 发送 sp4
```

`--pre-roll-sec` 控制缓冲区长度（默认 1.5s），该值应 ≥ VAD 预热时间。

### 问题：GPU 显存无论大小模型都占满

**根因：vLLM 的 `gpu_memory_utilization` 控制 KV Cache 预占比，非模型占用。**

vLLM 内部计算：
```
KV_cache_size = total_vram × gpu_memory_utilization − model_size
```

- 24GB 显卡 @ 0.8: `24×0.8 − 3.4 = 15.8 GB` KV cache
- 24GB 显卡 @ 0.5: `24×0.5 − 3.4 = 8.6 GB` KV cache

0.6B 模型（~1.2GB）和 1.7B 模型（~3.4GB）差 2.2GB，但 KV cache 都是按比例算的，所以显存占用看起来差不多。

**修复：** 默认值从 0.8 降为 0.5，显卡越小越应该调低。
```bash
python asr_server.py --gpu-memory-utilization 0.3 ...  # 8GB 显卡
```

### 问题：无法做到实时识别

**根因：Qwen3-ASR 架构设计 — 每个 chunk 重新编码全部累积音频。**

关键代码 `qwen_asr/inference/qwen3_asr.py:725-751`：
```python
state.audio_accum = np.concatenate([state.audio_accum, chunk])  # 无限增长
inp = {"audio": [state.audio_accum]}  # 每次编码全部音频
```

| 说话时长 | 编码量 | 每 chunk 耗时 |
|----------|--------|--------------|
| 5s | 1s+2s+3s+4s+5s = 15s 量 | ~200ms |
| 15s | 累积 ~120s 量 | ~500ms |
| 30s | 累积 ~465s 量 | ~1s+ → **掉队** |

这是模型的设计取舍（完整上下文 → 最高精度）。Qwen3-ASR 流式提供的是"增量结果"而非"实时编码"。

**缓解措施：**
1. **客户端 VAD 自然分段**：每段语音自动重启 session，编码量重置
2. **并发 send/receive**：客户端现在分离为两个 asyncio task，音频持续发送不阻塞
3. **降低 `gpu_memory_utilization`**：避免显存争抢导致 swap
4. **2-pass 模式**：pass 1 流式快览 + pass 2 离线完整一次（不再每 chunk 重复编码）

**如果你需要真正"逐帧实时"（如直播字幕），建议考虑纯 CTC 模型（如 Whisper、Zipformer），而非 encoder-decoder 架构。**

## 已知限制

1. 流式推理仅支持 vLLM backend（transformers backend 不支持）
2. 流式模式下不支持时间戳输出（forced aligner 不支持）
3. 使用 Uvicorn ASGI 单进程运行，不支持多 GPU
4. VAD 为简单能量检测，对嘈杂环境敏感度需调 `--vad-threshold`
