# enhance_mode 上下文统一改造计划
> 目标：保留 `forward_context` 对其他插件的隐式 prompt 注入能力，只改造 `astrbot_plugin_astrbot_enhance_mode`，消除主动回复链路里的重复上下文注入与不回复问题。

## 一、范围

### 本次修改
- 修改：`Aeolian01/astrbot_plugin_astrbot_enhance_mode`
- 保留：`Aeolian01/astrbot_plugin_forward_context` 的隐式注入能力
- 不修改：`Zhalslar/astrbot_plugin_pokepro`

### 不做的事
- 不让 `forward_context` 退化成纯 extra-only
- 不要求 `pokepro` 主动读取 extra
- 不在本次改造中调整其他插件的 prompt 组装逻辑

---

## 二、现状基线

### 1. forward_context 当前职责
`forward_context` 当前仍然负责：
- 解析复杂消息并写入 `event.extra`
- 可选改写 `event.message_str`
- 在 `on_llm_request` 阶段隐式改写 `req.prompt`
- 可选把最近插件输出追加进 LLM 请求

这部分能力需要保留，作为 `pokepro` 等插件继续工作的基础。

### 2. enhance_mode 当前职责
`enhance_mode` 当前已经：
- 直接依赖 `forward_context` 的公共 API
- 优先尝试读取 `_forward_context_text`
- 自己维护 `session_chats`
- 自己有 active reply / model_choice 两段流程
- 内部存在 `_enhance_active_reply_prompt`、`_enhance_active_reply_seeded_context` 这类交接位

说明问题的根源不在 `forward_context`，而在 `enhance_mode` 自己的上下文组装和 active reply 第二阶段。

---

## 三、改造目标

把 `enhance_mode` 收敛成下面这套规则：

### 规则 A：forward_context 只负责“外部解析增强”
- 对其他插件继续保留隐式 prompt 注入
- 对 `enhance_mode` 来说，它只是“当前消息文本的增强来源”

### 规则 B：enhance_mode 自己只允许一个“主动回复最终 prompt 组装器”
- `model_choice` 和 `active_reply` 共用同一套上下文采集逻辑
- `active_reply` 不再依赖“自动建会话后再 seed 最近群聊”来补上下文
- `active_reply` 的最终 prompt 必须由 `enhance_mode` 显式组装并交接

### 规则 C：消除 enhance_mode 内部二次注入
- 不允许“显式组好的 prompt” + “会话 seed 上下文”同时生效
- 不允许“session_chats 历史” + “active reply auto seed 历史”重复出现

---

## 四、核心设计

## 4.1 当前消息文本来源统一

新增一个统一入口函数，例如：

```python
async def _collect_active_reply_context(self, event, cfg) -> dict[str, Any]:
    ...
```

返回结构建议：

```python
{
    "origin": origin,
    "current_message_text": current_message_text,
    "recent_history_lines": recent_history_lines,
}
```

### current_message_text 规则
优先级固定为：

1. `event.extra["_forward_context_text"]`
2. `session_chats` 中与当前消息 ID 对应的历史行
3. 当前 event 原始 message chain 解析文本
4. `event.message_str`

也就是说：
- `forward_context` 负责把复杂消息解析好
- `enhance_mode` 只消费结果，不重复发明第二套当前消息解析逻辑

---

## 4.2 model_choice 与 active_reply 共用一套上下文采集

新增两个轻量函数：

```python
def _build_model_choice_prompt(...)
def _build_active_reply_prompt(...)
```

但它们的数据源必须都来自：

```python
_collect_active_reply_context(...)
```

这样可以确保：
- 判定阶段看到的“当前消息/历史”
- 正式回复阶段看到的“当前消息/历史”

来自同一个收集器，而不是两套来源拼在一起。

---

## 4.3 active_reply 改成“显式 prompt 交接”，不再靠 seed

### 旧问题
active reply 当前很可能依赖：
- auto create conversation
- seed 最近群聊上下文
- 然后再走 AstrBot pipeline

这会造成：
- 与 `session_chats` 重复
- 与 `_forward_context_text` 解析后的当前消息语义重复
- 在日志上表现为“判定通过，但第二段很快结束或上下文重复”

### 新规则
active reply 统一改成：

1. `enhance_mode` 自己显式组装 final prompt
2. 写入：
   - `_enhance_active_reply_prompt`
3. 在 `enhance_mode` 的 `on_llm_request` 或等效交接点中，把该 prompt 写进 `req.prompt`
4. 调用模型时不再额外 seed 最近群聊

### 结果
- 最终 prompt 只来自 `enhance_mode`
- `forward_context` 不会覆盖它，因为它看到的 prompt 已经不是空占位 prompt
- 其他插件仍然可以继续依赖 `forward_context` 的隐式注入

---

## 五、要删除的配置项与无用状态

### 从 `plugin_config.py` 删除
以下配置项删除：

- `seed_context_on_auto_create`
- 旧模型判定上下文字段

原因：
- 它们只服务于“自动建会话 + 再灌历史”这条旧链路
- 新方案里，active reply 的上下文由 `enhance_mode` 自己显式组装
- 留着只会让配置和实际逻辑不一致

### 保留
- `auto_create_conversation`
  - 仅表示：没有 conversation 时是否创建空会话容器
  - 不再表示：自动灌历史

### 从 `main.py` 删除
如果最终确认不再使用，应删除：
- `_enhance_active_reply_seeded_context`
- 与 seed 行为相关的日志
- 与 “最近群聊自动灌入会话” 相关的辅助函数/分支

### 保留
- `_enhance_active_reply_prompt`
  - 作为 active reply 最终 prompt 的唯一交接位

---

## 六、具体实施步骤

## 第一步：收敛配置
修改：
- `astrbot_plugin_astrbot_enhance_mode/plugin_config.py`
- `astrbot_plugin_astrbot_enhance_mode/_conf_schema.json`
- `astrbot_plugin_astrbot_enhance_mode/README.md`

动作：
- 删除 `seed_context_on_auto_create`
- 删除旧模型判定上下文字段
- 新增 `unified_context_messages`
- 更新说明：active reply 不再通过 auto seed 灌入群聊上下文

---

## 第二步：抽公共上下文收集器
修改：
- `astrbot_plugin_astrbot_enhance_mode/main.py`

新增：
- `_collect_active_reply_context(...)`
- `_build_model_choice_prompt(...)`
- `_build_active_reply_prompt(...)`

要求：
- `model_choice`
- `active_reply`

都只能从 `_collect_active_reply_context()` 取数据。

---

## 第三步：改造 active_reply 交接
修改：
- `astrbot_plugin_astrbot_enhance_mode/main.py`

动作：
- `active_reply` 触发后，不再 seed 最近群聊
- 只生成最终 prompt
- 将最终 prompt 写入 `_enhance_active_reply_prompt`
- 在 `enhance_mode` 的请求交接点中，显式把该 prompt 写入 `req.prompt`

---

## 第四步：删除 seed 旧逻辑
修改：
- `astrbot_plugin_astrbot_enhance_mode/main.py`

动作：
- 删除“自动创建会话并附带最近群聊上下文”的旧分支
- 删除与 seed 相关的状态标记与日志
- 删除无用常量 `_enhance_active_reply_seeded_context`

---

## 第五步：补日志，便于验收
新增日志建议：

```python
logger.info("enhance-mode | context collected | origin=%s current=%r history_count=%s", ...)
logger.info("enhance-mode | model_choice | final prompt len=%s", ...)
logger.info("enhance-mode | active_reply | final prompt len=%s", ...)
logger.info("enhance-mode | active_reply | req.prompt injected")
```

禁止再出现这类模糊日志作为主判断依据：
- “自动创建会话并附带最近群聊上下文”

新日志应该能直接回答：
- 当前消息文本从哪来
- 历史条数是多少
- final prompt 是否成功写进请求

---

## 七、验收标准

### A. forward_context 不受影响
- `pokepro` 这类插件仍可通过 `forward_context` 的隐式 prompt 注入正常读取复杂消息
- `forward_context` 的 `on_llm_request` 仍然存在并可工作

### B. enhance_mode 不再二次灌历史
- active reply 触发后，不再出现“自动 seed 最近群聊”的路径
- 不再依赖 `seed_context_on_auto_create`
- 配置文件中也不再有该项

### C. model_choice 与 active_reply 上下文一致
- 两阶段读取的当前消息文本一致
- 两阶段读取的最近历史窗口一致
- 不再出现“判定看到 A，正式回复看到另一套 B” 的情况

### D. 不回复问题可观测
日志中必须能明确看到：
- `model_choice` 最终 prompt 已生成
- `active_reply` 最终 prompt 已生成
- `req.prompt` 已被显式注入
- 后续是否产出模型结果

---

## 八、推荐提交顺序

### Commit 1
`refactor: remove active reply auto-seed config and schema`

### Commit 2
`refactor: unify enhance-mode context collection for model_choice and active_reply`

### Commit 3
`refactor: inject active reply prompt explicitly instead of seeding conversation`

### Commit 4
`chore: remove obsolete seeded-context state and logs`

---

## 九、回滚策略

如果改造后 active reply 无法触发，可按下面顺序排查：

1. 检查 `_enhance_active_reply_prompt` 是否写入
2. 检查 `req.prompt` 是否被 enhance_mode 显式覆盖
3. 检查 forward_context 是否在后续把 prompt 改回占位文本
4. 如果必要，临时恢复旧的 active reply 调用分支，但不要恢复 seed 配置项

---

## 十、最终原则

### 保持不变
- `forward_context` 继续服务全局插件生态
- `pokepro` 不改

### 真正收敛的点
- 只改 `enhance_mode`
- 只消除 `enhance_mode` 自己的重复上下文来源
- 让 active reply 变成“显式 prompt 交接”，而不是“隐式 seed 会话”
