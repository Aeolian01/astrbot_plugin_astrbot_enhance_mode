# enhance_mode 重构 TODO 清单

## 配置与文档
- [ ] 删除 `plugin_config.py` 中的 `seed_context_on_auto_create`
- [ ] 删除 `plugin_config.py` 中的旧模型判定上下文字段
- [ ] 新增 `plugin_config.py` 中的 `unified_context_messages`
- [ ] 删除 `_conf_schema.json` 中对应配置项
- [ ] 更新 README，移除“自动 seed 最近群聊上下文”说明
- [ ] 更新 README，补充“active reply 改为显式 prompt 注入”说明

## main.py 上下文统一
- [ ] 抽出 `_collect_active_reply_context(...)`
- [ ] 统一当前消息文本来源优先级
  - [ ] `_forward_context_text`
  - [ ] `session_chats` 对应当前 message_id 的历史行
  - [ ] event message chain 解析文本
  - [ ] `event.message_str`
- [ ] 统一 recent history 采集窗口
- [ ] 统一 model_choice extra history 采集窗口
- [ ] 在 context 收集阶段补充清晰日志

## prompt 构造统一
- [ ] 抽出 `_build_model_choice_prompt(...)`
- [ ] 抽出 `_build_active_reply_prompt(...)`
- [ ] `model_choice` 仅从统一 context 构造 prompt
- [ ] `active_reply` 仅从统一 context 构造 prompt

## active reply 显式交接
- [ ] active reply 触发后只生成 final prompt
- [ ] 将 final prompt 写入 `_enhance_active_reply_prompt`
- [ ] 在请求交接点显式注入 `req.prompt`
- [ ] 为 prompt 注入成功增加日志

## 删除 seed 旧链路
- [ ] 删除 `_enhance_active_reply_seeded_context`
- [ ] 删除与 seed 状态相关的写入逻辑
- [ ] 删除与 seed 路径相关的日志
- [ ] 删除“自动创建会话并附带最近群聊上下文”分支
- [ ] 保留 `auto_create_conversation`，仅用于空会话容器创建

## 验收
- [ ] `forward_context` 对其他插件保持兼容
- [ ] `pokepro` 不需要任何改动
- [ ] `model_choice` 与 `active_reply` 读取同一套上下文
- [ ] 不再出现 prompt + seed 双重历史注入
- [ ] 日志能明确看到 prompt 生成与 req.prompt 注入
- [ ] 主动回复“不回复”问题具备可观测性
