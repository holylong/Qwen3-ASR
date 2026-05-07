# asr_server.py 健壮性修复 — 修改记录

> Date: 2026-05-07
> Project: Qwen3-ASR
> Branch: master

## 背景

**症状：** Android 客户端一连接，所有客户端（包括 Python 客户端）都停止接收识别结果，服务器无报错日志。

**触发场景：**
1. Python 客户端（`asr_client.py`）单独使用时正常
2. Android 客户端连上后，Android 端和已在通话的 Python 端均卡死
3. 服务器无 crash，无 error 日志，表现为沉默（所有消息不再到达客户端）

## 根因分析

### 问题 1：WebSocket 断开后无限错误循环 (crash loop)

**原始代码缺陷：**
```python
# asr_server.py 旧版 while 循环
while True:
    raw = await ws.receive()
    if "bytes" in raw:
        ...
    elif "text" in raw:
        ...
    # ← 没有 else / 没有处理 websocket.disconnect！
```

当客户端断开时，Starlette 的 `ws.receive()` 返回 `{"type": "websocket.disconnect"}`。旧代码不匹配任何分支，直接回到循环顶部再次调用 `ws.receive()`，Starlette 此时抛出：
```
RuntimeError: Cannot call "receive" once a disconnect message has been received.
```

`except Exception` 捕捉后仅记录日志，**循环继续** → 无限错误循环，单协程饿死整个事件循环。

### 问题 2：发送结果前客户端断开

`ws.send_text()` 在连接关闭后调用会抛出 `RuntimeError: Cannot call "send" once a close message has been sent.`。`_send_result`/`_send_error` 中 `except Exception: pass` 吞掉了异常，但 handler 中的直接 `ws.send_text()`（"started" 回复）未受保护。

### 问题 3：并发模型访问死锁（核心根因）

**关键代码：**
```python
asr_model = Qwen3ASRModel.LLM(...)  # 全局单例

# 每个 WebSocket handler 通过线程池并发调用：
await loop.run_in_executor(executor, _streaming_step, ...)
```

vLLM 的同步 `LLM` 类不是线程安全的。多客户端并发调用 `streaming_transcribe` 时，vLLM 内部互斥/队列机制出现死锁或状态冲突，**所有客户端的模型调用全部卡死**。这就解释了为什么：

- Python 客户端单独用 ✅ — 只有一个线程在调模型
- Android 客户端一进来 ❌ — 两个线程并发调模型 → 死锁 → 全部客户端收不到结果

### 问题 4：Android 客户端音频块过小

Android 客户端每块只发 **1280 bytes (320 samples = 20ms)**，而 Python 客户端发 **16000 bytes (4000 samples = 250ms)**。

- 同样 2 秒音频：Python 需 8 块，Android 需 100 块
- 每块都要：收消息 → 抢模型锁 → 调 vLLM → 发结果
- Android 每秒抢锁 50 次，`_model_lock` 竞争激烈

## 修改内容

### 修复 1：处理 WebSocket disconnect 消息 + 断开后安全退出

**文件：** `asr_server.py` 第 236-241 行

```python
# 新增：显式处理 Starlette 的 websocket.disconnect 消息类型
if raw.get("type") == "websocket.disconnect":
    code = raw.get("code", "?")
    reason = raw.get("reason", "")
    logger.info(f"WebSocket disconnect: {client_host}:{client_port}"
                f" code={code} reason={reason!r}")
    break  # 安全退出循环
```

**错误处理重构（第 372-393 行）：**

```python
except WebSocketDisconnect:    # 正常断开
    ...
except WebSocketException:    # FastAPI WebSocket 层异常
    ...
except asyncio.TimeoutError:  # executor 超时（见修复 3）
    ...
except asyncio.CancelledError:# 协程被取消
    ...
except RuntimeError as e:     # 连接已关闭时的 send/receive 错误
    ...
except Exception as e:        # 其他未预期错误
    ...
```

关键改进：`RuntimeError` 和 `asyncio.TimeoutError` 被**独立捕获**，不会导致循环重入。

### 修复 2：统一 WebSocket 发送超时保护

**`_send_result`（第 159-176 行）：**

```python
async def _send_result(ws, language, text, is_partial, pass_num):
    try:
        await asyncio.wait_for(
            ws.send_text(json.dumps({...})),
            timeout=SEND_TIMEOUT,  # 5 秒超时
        )
    except asyncio.TimeoutError:
        logger.warning(f"_send_result: send_text timed out after {SEND_TIMEOUT}s"
                       f" — client not reading?")
    except RuntimeError:
        logger.debug("_send_result: websocket already closed, dropping result")
```

**`_send_error`（第 179-191 行）：** 同逻辑。

**"started" 回复保护（第 333-340 行）：**

```python
try:
    await ws.send_text(json.dumps({"type": "started", ...}))
except (RuntimeError, WebSocketDisconnect) as e:
    logger.info(f"Client {client_host}:{client_port} disconnected"
                f" during session start: {e}")
    break
```

### 修复 3：模型推理超时 + 线程池监控

**新增常量（第 68-69 行）：**

```python
EXECUTOR_TIMEOUT = 30   # 单次推理调用最大秒数
SEND_TIMEOUT = 5        # WebSocket 发送最大秒数
```

**`_run_in_executor` wrapper（第 132-156 行）：**

```python
async def _run_in_executor(tag: str, fn, *args):
    """带超时、计时日志的线程池封装。"""
    t0 = time.time()
    async with _model_lock:                # ← 修复 4：模型锁
        t1 = time.time()
        if (t1 - t0) * 1000 > 100:
            logger.info(f"[{tag}] model lock: waited {(t1-t0)*1000:.0f}ms")
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(executor, fn, *args),
                timeout=EXECUTOR_TIMEOUT,
            )
            elapsed = (time.time() - t1) * 1000
            logger.debug(f"[{tag}] executor: done in {elapsed:.0f}ms"
                         f" (lock_wait={(t1-t0)*1000:.0f}ms)")
            return result
        except asyncio.TimeoutError:
            logger.error(f"[{tag}] executor: TIMEOUT after {EXECUTOR_TIMEOUT}s"
                         f" — model inference hung!")
            raise
```

所有 `loop.run_in_executor(...)` 调用已替换为 `_run_in_executor(...)`。

**`/health` 端点增强（第 82-90 行）：**

```python
@app.get("/health")
async def health():
    pending = executor._work_queue.qsize()
    return {
        "status": "ok",
        "sessions": len(SESSIONS),
        "executor_pending": pending,       # 排队中的任务数
        "executor_max_workers": executor._max_workers,
    }
```

若 `executor_pending` 持续 > 0，说明线程池被卡死。

### 修复 4：模型锁 — 串行化 vLLM 访问（最关键）

**新增（第 71-73 行）：**

```python
# vLLM 同步 LLM 类非线程安全，并发调用会死锁/状态冲突。
_model_lock = asyncio.Lock()
```

**影响范围：**

| 调用点 | 原代码 | 改为 |
|---|---|---|
| `_streaming_step` | `executor.submit(fn)` | `_run_in_executor("streaming_step", fn)` → 自动 `async with _model_lock` |
| `_finish_streaming` | `executor.submit(fn)` | `_run_in_executor("finish_streaming", fn)` |
| `_offline_transcribe` | `executor.submit(fn)` | `_run_in_executor("offline_transcribe", fn)` |
| `init_streaming_state` | 直接在事件循环调用 | `async with _model_lock: state = asr_model.init_...` |

保证**同一时刻仅一个协程**在执行模型调用，消除死锁。

### 修复 5：全面日志增强

| 日志场景 | 级别 | 内容 |
|---|---|---|
| 连接/断开 | INFO | 客户端 IP:Port + 断开码 + 原因 |
| 每条 WebSocket 消息 | DEBUG | type + bytes 大小 + text 截断 |
| 未识别的消息类型 | WARNING | raw type + 所有 keys |
| 无效 JSON | WARNING | 原始文本前 200 字符 |
| 无 session 的二进制数据 | WARNING | 客户端地址 |
| 重复 start 被拒绝 | WARNING | 已有 session_id |
| 未知 msg_type | WARNING | type 值 |
| start 握手时断开 | INFO | 异常信息 |
| 发送结果超时 | WARNING | "client not reading?" |
| 推理超时 | ERROR | "model inference hung!" + 耗时 |
| 模型锁等待 | INFO (>100ms) | 锁等待时长 |
| 异常处理 | INFO/ERROR | 客户端 + session + 异常详情 |
| 断开时 session 清理 | INFO | session_id |

### 修复 6：重复 start 拒绝

```python
if session_id is not None:
    logger.warning(f"Duplicate start rejected from {client_host}:{client_port}"
                   f" (existing session={session_id[:8]})")
    await _send_error(ws, "Session already started; finish current first")
    continue
```

防止客户端误发多次 `start` 导致旧 session 泄漏。

### 修复 7：线程池线程命名

```python
executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="asr_worker")
```

方便在 `py-spy` / gdb 中排查卡死的线程。

## 诊断命令

```bash
# 启动服务并显示详细日志
python asr_server.py --asr-model-path ./models/Qwen3-ASR-1.7B --debug --port 8000

# 查看线程池状态
curl http://localhost:8000/health
# → {"status":"ok","sessions":2,"executor_pending":3,"executor_max_workers":4}
# executor_pending > 0 持续增长 = 线程池被卡死

# 观察 Android 客户端日志特征
# 正常 Python 客户端的音频块: bytes=16000 (4000 samples)
# Android 客户端的音频块:      bytes=1280  (320 samples) ← 太小
```

### 日志解读示例

正常流程：
```
[DEBUG] WS recv: type=websocket.receive, bytes=16000, text=False
[DEBUG] [streaming_step] executor: done in 280ms (lock_wait=0ms)
```

异常信号：
```
[WARNING] Binary data without active session from 127.0.0.1:55070     ← 客户端没发 start
[WARNING] _send_result: send_text timed out after 5s — client not reading?  ← 客户端不收结果
[ERROR]   [streaming_step] executor: TIMEOUT after 30s — model inference hung! ← 模型卡死
[INFO]    [streaming_step] model lock: waited 2350ms                  ← 前面有人占锁很久
```

## Android 客户端建议

确保 Android 端遵循以下规范：

1. **音频块大小：** 累积至至少 4000 samples (0.25s @ 16kHz) 再发送一个 WebSocket binary 消息
2. **协议流程：** start → 持续发音频 → finish，不要 start → finish（空 session）
3. **读取服务端消息：** 必须消费 `_send_result` 发回的 JSON 结果，否则服务器发送缓冲满后 `send_text` 超时

参考 `asr_client.py` 第 38-40 行：
```python
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.25
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)   # = 4000
```

## 文件变更摘要

| 文件 | 变更类型 | 行数变化 |
|---|---|---|
| `asr_server.py` | 修改 | 396 → 501 行 |

无新增文件。
