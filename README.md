# Qwen3-ASR WebSocket 实时语音识别服务

基于 [Qwen3-ASR](https://github.com/Qwen/Qwen3-ASR) 的 WebSocket 实时语音识别服务，支持服务端 VAD 断句、热词唤醒、两遍解码。

## 功能特性

| 特性 | 说明 |
|------|------|
| 实时流式识别 | WebSocket 协议，低延迟 streaming 转写 |
| 两遍解码 (two-pass) | Pass 1 流式预览 + Pass 2 离线精修 |
| 服务端 VAD | Silero VAD 自动断句，客户端只需发送音频 |
| 热词唤醒 | 通过 context 提示词 + 后处理匹配实现唤醒词/指令识别 |
| 多语言 | 支持 52 种语言和方言的识别 |
| 并发安全 | 信号量控制 GPU 并发，ThreadPoolExecutor 异步执行 |

## 架构

```
┌─────────────────────┐     WebSocket      ┌─────────────────────────┐
│   asr_client.py     │ ◄── binary PCM ──► │    asr_server.py        │
│  (客户端 VAD 断句)   │ ◄── JSON 消息 ──► │  (Qwen3-ASR 推理引擎)    │
└─────────────────────┘                    └─────────────────────────┘

┌──────────────────────────┐   WebSocket   ┌─────────────────────────────┐
│ asr_client_vad_server.py │ ◄── binary ──► │   asr_server_vad.py         │
│ (纯音频流，无 VAD)         │ ◄── JSON ───► │ (Silero VAD + Qwen3-ASR)    │
└──────────────────────────┘               └─────────────────────────────┘
```

- **方案一**：客户端做能量阈值 VAD，检测到语音后发 `start` → 发音频 → `finish`
- **方案二**：客户端只持续发音频，服务端用 Silero VAD 神经网络自动断句

## 文件说明

| 文件 | 用途 |
|------|------|
| `asr_server.py` | 服务端（原始协议，客户端 VAD） |
| `asr_client.py` | 客户端（能量阈值 VAD + start/finish 协议） |
| `asr_server_vad.py` | 服务端（Silero VAD 自动断句） |
| `asr_client_vad_server.py` | 客户端（纯音频流，无 VAD） |
| `hotwords.json` | 热词配置（唤醒词 + 指令 + context 提示词） |
| `start_simple_server.sh` | 启动原始服务端 |
| `start_simple_client.sh` | 启动原始客户端 |
| `start_server_vad.sh` | 启动 VAD 服务端 |
| `start_client_vad_server.sh` | 启动 VAD 客户端 |

## 环境要求

### 硬件
- NVIDIA GPU（vLLM 推理）
- 建议显存：Qwen3-ASR-0.6B ≥ 6GB，Qwen3-ASR-1.7B ≥ 12GB

### 软件
- Python ≥ 3.9
- CUDA 驱动 + PyTorch

## 安装

```bash
# 安装基础依赖
pip install qwen-asr[vllm] fastapi uvicorn websockets numpy sounddevice

# 如需 ModelScope 下载模型
pip install modelscope

# Silero VAD（首次启动自动通过 torch.hub 下载到 ./models/）
```

## 快速启动

### 1. 启动服务端

```bash
# 原始服务端 + 热词
bash start_simple_server.sh

# 或 VAD 服务端 + 热词
bash start_server_vad.sh

# 手动命令示例
python asr_server_vad.py \
  --asr-model-path ./models/Qwen3-ASR-0.6B \
  --gpu-memory-utilization 0.85 \
  --max-new-tokens 128 \
  --chunk-size-sec 2.0 \
  --hotwords ./hotwords.json \
  --port 8000
```

### 2. 启动客户端

```bash
# 原始客户端（自带能量 VAD）
bash start_simple_client.sh

# VAD 服务端配套客户端
bash start_client_vad_server.sh

# 手动命令示例
python asr_client_vad_server.py --url ws://localhost:8000/ws/asr --verbose
python asr_client_vad_server.py --url ws://localhost:8000/ws/asr --two-pass
```

### 3. 开始说话

客户端连接后，对着麦克风说话即可看到实时转写结果。

## 服务端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--asr-model-path` | `./models/Qwen3-ASR-0.6B` | 模型路径或 HuggingFace/ModelScope 名称 |
| `--port` | `8000` | 监听端口 |
| `--gpu-memory-utilization` | `0.5` | vLLM GPU 显存利用率 |
| `--max-new-tokens` | `256` | 最大生成 token 数 |
| `--chunk-size-sec` | `1.0` | 流式推理 chunk 时长（秒） |
| `--max-concurrent-requests` | `4` | 最大并发推理数 |
| `--hotwords` | _(空)_ | 热词配置文件路径 |
| `--save-audio-dir` | _(空)_ | 保存音频的目录 |
| `--debug` | off | 开启 DEBUG 日志 |

### VAD 服务端额外参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--vad-threshold` | `0.5` | Silero VAD 语音概率阈值 (0.0–1.0) |
| `--vad-min-silence-ms` | `300` | 最小静音时长后断句 (ms) |
| `--vad-speech-pad-ms` | `100` | 语音片段前后填充 (ms) |
| `--vad-pre-roll-chunks` | `2` | 语音开始前预缓冲 chunk 数 |
| `--max-utterance-sec` | `30` | 单句最大时长，超时强制断句 |

## 客户端参数

| 参数 | 说明 |
|------|------|
| `--url` | 服务端 WebSocket 地址 |
| `--two-pass` | 开启两遍解码模式 |
| `--device` | 指定音频输入设备（索引或名称） |
| `--list-devices` | 列出可用音频设备 |
| `--save-audio` | 保存录音到 WAV 文件 |
| `--verbose` | 详细日志 |

## 热词配置

热词文件 `hotwords.json` 结构：

```json
{
  "wake_words": ["你好小明", "小明同学", "嘿小明"],
  "commands":   ["打开电视", "关闭电视", "开灯", "关灯", ...],
  "context":    "智能家居语音控制"
}
```

- `wake_words` — 唤醒词列表，匹配后显示 `⚑ [WAKE]`
- `commands` — 指令列表，匹配后显示 `⌘ [CMD]`
- `context` — ASR 模型 context 提示词（**越短越好**，5-10 字，如 "智能家居控制"）

**注意**：Qwen3-ASR 没有原生热词 biasing，热词功能通过以下两层实现：
1. `context` 参数作为系统提示词引导模型关注特定领域词汇
2. ASR 输出后做子串匹配，命中唤醒词/指令时在结果中附带 `hotword_match` 字段

## WebSocket 协议

### 方案一（原始协议）

```
Client → Server:  {"type": "start", "mode": "streaming"|"two-pass"}
Server → Client:  {"type": "started", "session_id": "...", "mode": "..."}
Client → Server:  BINARY int16 PCM 16kHz 音频
Client → Server:  {"type": "finish"}
Server → Client:  {"type": "result", "text": "...", "is_partial": false, "pass": 1|2,
                   "hotword_match": {"type": "wake_word"|"command", "word": "..."}}
```

### 方案二（服务端 VAD 协议）

```
Client → Server:  {"type": "set_mode", "mode": "streaming"|"two-pass"}
Server → Client:  {"type": "mode_set", "mode": "..."}
Client → Server:  BINARY int16 PCM 16kHz 音频（持续流）
Server → Client:  {"type": "result", "text": "...", "is_partial": true|false, "pass": 1|2,
                   "hotword_match": {...}}
Server → Client:  {"type": "vad_state", "state": "speaking"|"listening"}
```

## 健康检查

```bash
curl http://localhost:8000/health
```

## 引用

本项目基于 [Qwen3-ASR](https://github.com/Qwen/Qwen3-ASR)。VAD 基于 [Silero VAD](https://github.com/snakers4/silero-vad)。
