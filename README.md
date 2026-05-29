# AstrBot Enhance Mode

**Version**: `v0.2.7`  
**Author**: `阿汐`

`astrbot_plugin_astrbot_enhance_mode` 是 AstrBot 的群聊增强插件，提供 React 群聊上下文、主动回复、标签解析、封禁控制、Memory RAG、联网搜索与可视化 WebUI。

## 功能亮点

- 面向群聊场景的 React 上下文增强，支持消息编号、角色标签、发送者 ID、图片与视频占位记录。
- 主动回复支持概率触发与模型判定触发，并可为判定流程指定独立 Provider。
- 被动回复、主动回复和 `model_choice` 判定共用同一套群聊历史上下文，统一由 enhance-mode 显式组装 prompt。
- 支持 `<mention/>`、`<quote/>`、`<refuse/>` 控制标签，把模型输出转换为平台消息组件或主动取消发送。
- 提供 Bot 侧封禁控制、可检索长期记忆、联网搜索工具与记忆管理 WebUI。
- 可消费 `astrbot_plugin_forward_context` 的合并转发解析结果，让转发消息进入主动回复判定和历史上下文。

## 环境要求

- AstrBot 插件运行环境。
- Python 依赖：`fastapi>=0.115.0`、`uvicorn>=0.30.0`，已在 `requirements.txt` 声明。
- 可选增强：如需解析 QQ 合并转发、嵌套转发或 JSON 分享卡片，请同时安装 `astrbot_plugin_forward_context`。

## Update Notes (v0.2.7)

- 平台历史补齐注入时，若历史消息包含合并转发、嵌套转发、JSON/Ark 分享卡片，会调用 forward-context 公共解析 API 展开后再写入上下文。
- 图片历史注入会优先复用 forward-context 的共享图片描述缓存，并把新解析结果写回共享缓存。
- 视频历史会记录为 `[Video]`，模型需要解读时可调用 `enhance_use_video(...)`，由 Gemini 原生 Files API 上传视频后生成描述并写回历史。
- 被动回复、active reply 正式回复和 `model_choice` 判定改为共用统一历史窗口，不再维护 model_choice 专用历史池。
- `active_reply.seed_context_on_auto_create` 已移除；`auto_create_conversation` 仅表示是否自动创建空会话容器。

## Features

### Group Chat Enhancement

- React 模式（群聊上下文增强总开关）
- 群聊历史增强（可注入发送者 ID、角色标签、消息编号，并作为统一历史来源）
- 图片转述（可选，历史先记录 `[Image]`，注入上下文时自动解析并回填历史）
- 视频转述（历史先记录 `[Video]`，需要时通过 `enhance_use_video` 上传到 Gemini Files API 解析并回填历史）
- 支持消费 `astrbot_plugin_forward_context` 写入的合并转发解析文本
- 角色显示（在 system reminder 注入 `admin/member`）

### Active Reply

- `probability` 概率触发
- `model_choice` 模型判定触发（支持人格面具占位符）
- 白名单控制（按 `unified_msg_origin` 或群号）
- 可自动创建缺失的 AstrBot 对话，但不会再向新会话自动 seed 最近群聊上下文
- `model_choice` 判定历史不足时可通过平台适配器查询 QQ 群最新历史进行补齐
- 正式回复阶段由 enhance-mode 显式组装最终 prompt 并注入 `req.prompt`

### Output Tags

- `<mention id="..."/>`：转为平台 At 组件
- `<quote id="..."/>`：转为平台引用组件
- `<refuse/>`：触发拒绝发送，清空结果链

### Ban Control

- 运行时拦截被封禁用户消息
- LLM Tools:
  - `enhance_get_ban_list_status`
  - `enhance_ban_user`
  - `enhance_unban_user`

### Memory RAG

- LLM Tools:
  - `grok_web_search`
  - `enhance_use_image`
  - `enhance_use_video`
  - `enhance_memory_rag_write`
  - `enhance_memory_rag_read`
- Embedding Provider 独立配置（不是聊天模型 Provider）
- 时间显示与时间解析统一使用 AstrBot 全局 `timezone`（默认 `Asia/Shanghai`）
- 按角色、时间、群范围过滤
- 支持 `ignore_group_id=true` 跨群读取

### Memory RAG WebUI

- 独立 HTTP 服务
- 登录认证（支持固定密码或启动时随机密码）
- 统计、筛选、分页、详情、删除
- 管理命令：`/enhance rag-webui`

## Recommended Builtin Settings

为避免能力重叠，建议：

- 关闭内置群聊上下文：`group_icl_enable`
- 关闭内置主动回复：`active_reply.enable`
- 关闭内置引用回复：`reply_with_quote`
- 保持内置识别开启：`identifier`

## Forward Context Integration

如需支持 QQ 合并转发、引用合并转发、嵌套转发和 JSON 分享卡片，请同时安装并启用 `astrbot_plugin_forward_context`。

enhance-mode 不内置转发解析器。当前消息优先消费 forward-context 写入的事件扩展：

```python
event.get_extra("_forward_context_text")
event.get_extra("_forward_context_found")
event.get_extra("_forward_context_ids")
```

平台历史补齐注入时，enhance-mode 会调用 forward-context 的公共 `parse_history_message(event, message)` API；历史里的合并转发、嵌套转发、JSON/Ark 分享卡片会先展开，无法解析时才保留平台适配器提供的普通文本或 `[Forward]`/`[Json]` 占位。

建议 forward-context 使用以下关键配置：

```json
{
  "enable": true,
  "parse_group": true,
  "set_event_extra": true,
  "extra_key": "_forward_context_text",
  "inject_to_llm_request": true,
  "parse_reply_forward": true,
  "parse_direct_forward": true,
  "parse_nested_forward": true
}
```

## Configuration

配置分组（键名与 `_conf_schema.json` 一致）：

- `group_features`
- `group_history_enhancement`
- `active_reply`
- `web_search`
- `memory_rag`
- `memory_rag_webui`
- `global_settings`

### `active_reply`

当前重点配置项：

- `enable`：启用主动回复
- `mode`：`probability` 或 `model_choice`
- `possibility`：概率触发时生效
- `auto_create_conversation`：主动回复触发但当前群没有 AstrBot 对话时，自动创建并切换到新会话
- `unified_context_messages`：被动回复、主动回复与模型判定最多注入多少条统一群聊历史；`0` 表示不注入历史但仍记录历史
- `model_stack_size`：`model_choice` 触发栈长度
- `model_history_messages`：`model_choice` 从统一历史中额外附带的判定历史条数；`0` 表示不附带，也不触发平台历史补足
- `model_choice_provider_id`：判定模型提供商 ID
- `model_choice_prompt`：判定提示词，支持占位符
- `whitelist`：来源/群号白名单

说明：

- `active_reply.seed_context_on_auto_create` 已移除。
- `auto_create_conversation` 现在只负责创建空会话，不再承担 seed 历史的职责。
- 模型判定历史注入窗口统一由 `unified_context_messages` 控制。

### `group_history_enhancement`

图片和视频相关配置：

- `image_caption_provider_id`：图片转述 Provider。
- `image_caption_prompt`：图片转述提示词。
- `video_caption_provider_id`：视频转述 Provider；为空时优先复用 `image_caption_provider_id`，仍为空则使用当前会话 Provider。
- `video_caption_provider_ids`：视频转述 Provider 列表，按顺序尝试；遇到 Gemini 503/429、超时、空结果或坏响应时切换到下一个。非空时优先于单个 `video_caption_provider_id`。
- `video_caption_prompt`：视频转述提示词，默认要求用简体中文简短描述主要画面、动作、可见文字和关键信息。

超时配置位于 `global_settings.timeouts.video_caption_sec`，默认 `120` 秒。

## LLM Tools

### Ban Tools

1. `enhance_get_ban_list_status(user_id="", max_results=20)`
2. `enhance_ban_user(user_id, duration="10m")`
3. `enhance_unban_user(user_id)`

### Other Tools

- `grok_web_search(query)`
- `enhance_use_image(...)`
- `enhance_use_video(message_id, video_index=1, write_to_history=true, prompt="")`
- `enhance_memory_rag_write(...)`
- `enhance_memory_rag_read(...)`

当历史里出现 `[Video]` 或 `[Video: ...]` 且用户要求“解读/分析/看看这个视频”时，模型应调用 `enhance_use_video`。`message_id` 使用历史中的 `#msg...` 编号，`video_index` 从 `1` 开始；`prompt` 应传入用户当前的具体视频问题。工具会把原视频上传到 Gemini 原生 Files API，让 Gemini 直接基于视频回答，而不是只做通用转述；同一个视频的通用描述不会被误用为新问题的答案。成功后返回 JSON，其中 `native_video_result` 是原生视频分析结果，并在 `write_to_history=true` 时把目标 `[Video]`/`[Video: ...]` 替换为 `[Video: 结果]`。

当前视频工具只支持 Gemini / Google GenAI Provider。对 `google_gemini_openai/models/gemini-*` 这类 Provider，会把 OpenAI 兼容地址规范化为 Gemini 原生 `/v1beta` 与 `/upload/v1beta/files`，再上传视频并调用 `generateContent`。如果配置了 `web_search.proxy_url`，Gemini Files API 上传、轮询、生成和删除请求会复用该代理。Gemini `generateContent` 返回 503/429 时会短退避重试；如果配置了 `video_caption_provider_ids`，还会自动切换到下一个 Provider。非 Gemini Provider 会返回明确错误。

## Data Storage

插件数据目录：

```text
data/plugin_data/astrbot_plugin_astrbot_enhance_mode/
```

数据库文件：

- `ban_list.db`
- `memory_rag.db`

## Development

在插件目录执行：

```bash
ruff format .
ruff check .
```
