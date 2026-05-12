# 2026-05-09 热词（Hotwords）功能 & 服务端 VAD 断句支持

## 概述

本次修改新增两项能力：
1. **服务端 Silero VAD 断句** — 新增 `asr_server_vad.py` / `asr_client_vad_server.py`，VAD 移至服务端执行，客户端只需持续发送音频
2. **热词（Hotwords）功能** — 通过 ASR 的 `context` 参数注入领域提示，唤醒词/指令识别后匹配并高亮显示

---

## 一、新增文件

| 文件 | 说明 |
|------|------|
| `asr_server_vad.py` | 服务端版本，内置 Silero VAD 实时语音检测，自动断句并管理 ASR 会话 |
| `asr_client_vad_server.py` | 对应客户端，去掉了 VAD 逻辑和 start/finish 协议，仅持续采集并发送音频 |
| `hotwords.json` | 热词配置文件，定义唤醒词、指令和自定义 context 提示词 |
| `start_server_vad.sh` | 服务端 VAD 版本的启动脚本 |
| `start_client_vad_server.sh` | 客户端 VAD 服务端版本的启动脚本 |

## 二、修改文件

### 2.1 `asr_server.py`（原始服务端）

- 新增全局变量 `ASR_CONTEXT`、`HOTWORDS_DATA`
- 新增 CLI 参数 `--hotwords` （指向 hotwords.json 路径）
- `_send_result()` 增加 `hotword_match` 可选字段
- `_offline_transcribe()` 的 `context` 参数改为 `ASR_CONTEXT`
- `init_streaming_state()` 调用增加 `context=ASR_CONTEXT`
- finish 分支增加热词匹配并附加到结果消息
- 新增函数：`_load_hotwords()`、`_build_asr_context()`、`_match_hotwords()`
- `main()` 中加载 hotwords 配置

### 2.2 `asr_client.py`（原始客户端）

- `TUI` 类新增 `hotword(hw_type, word)` 方法，显示 `⚑ [WAKE]` 或 `⌘ [CMD]`
- `recv_loop()` 中处理 `hotword_match` 字段
- `_wait_final_results()` 中处理 `hotword_match` 字段

### 2.3 `asr_server_vad.py`（VAD 服务端）

- 同 2.1 的热词修改
- 额外：
  - 使用 `torch.hub.set_dir('./models')` 将 Silero VAD 模型缓存到本地
  - VADEngine 直接调用 Silero 模型推理（不依赖 VADIterator，避免 API 版本兼容问题）
  - 支持 `--vad-threshold`、`--vad-min-silence-ms`、`--vad-speech-pad-ms` 等调参

### 2.4 `asr_client_vad_server.py`（VAD 客户端）

- 同 2.2 的热词显示修改
- 协议简化为：发 `set_mode` → 持续发 binary 音频 → 接收结果

### 2.5 `start_simple_server.sh`

- 添加 `--hotwords ./hotwords.json` 参数

---

## 三、热词功能原理

Qwen3-ASR 模型**没有**原生的热词概率偏置（word biasing），但支持 `context` 参数 —— 以系统提示词形式传入领域描述文本，引导模型关注特定词汇。

```
context = "你是一个智能家居语音控制系统。唤醒词：你好小明。指令：开灯、关灯、打开电视..."
```

- **加载**：服务端启动时读取 `hotwords.json`，构建 `ASR_CONTEXT` 字符串
- **注入**：每次初始化 ASR 会话（`init_streaming_state`）或离线转写（`transcribe`）时传入 `context`
- **匹配**：ASR 返回文本后，子串匹配 `wake_words` / `commands` 列表
- **通知**：匹配结果以 `hotword_match: {"type":"wake_word"|"command","word":"..."}` 附带在 result 消息中
- **显示**：客户端终端显示 `⚑ [WAKE] 你好小明` 或 `⌘ [CMD] 打开电视`

### `hotwords.json` 结构

```json
{
  "wake_words": ["你好小明", "小明同学", "嘿小明"],
  "commands":   ["打开电视", "关闭电视", "开灯", "关灯", ...],
  "context":    "自定义 ASR 提示词（可选，不填则自动生成）"
}
```

---

## 四、服务端 VAD 架构

### 协议对比

| | 原始协议 | VAD 服务端协议 |
|---|---|---|
| 客户端→服务端 | `start` → binary → `finish` | `set_mode` → binary 持续流 |
| VAD 位置 | 客户端（能量阈值） | 服务端（Silero 神经网络） |
| 断句 | 客户端 VAD 检测 | 服务端 VAD 自动断句 |

### Silero VAD 引擎

- 每 32ms（512 采样 @16kHz）一帧送入 Silero 模型获取语音概率
- 状态机：概率 >= 阈值 → 触发 speech；概率 < 阈值-0.15 且持续 >= min_silence_ms → 结束
- 预录音缓冲（pre-roll）：保留最近 N 个 chunk 在 deque 中，speech 触发时一并发送给 ASR，避免丢失句首

---

## 五、命令行参数

```bash
# 原始服务端 + 热词
python asr_server.py --asr-model-path ./models/Qwen3-ASR-0.6B \
  --hotwords ./hotwords.json --port 8000

# VAD 服务端 + 热词
python asr_server_vad.py --asr-model-path ./models/Qwen3-ASR-0.6B \
  --hotwords ./hotwords.json --port 8000 --vad-min-silence-ms 300

# 客户端（任意一个都能连接两个服务端）
python asr_client.py --url ws://localhost:8000/ws/asr
python asr_client_vad_server.py --url ws://localhost:8000/ws/asr
```

---

## 六、依赖

| 包 | 用途 |
|---|---|
| `torch` | Silero VAD 模型推理 |
| `snakers4/silero-vad`（torch.hub） | VAD 模型，首次自动下载到 `./models/` |
| `qwen-asr[vllm]` | ASR 模型 |
| `fastapi` `uvicorn` | WebSocket 服务框架 |
| `websockets` `sounddevice` `numpy` | 客户端音频采集与通信 |

Silero VAD 通过 `torch.hub.set_dir('./models')` 缓存到本地 `./models/` 目录，首次下载后不会重复下载。
