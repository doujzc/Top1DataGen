# Top1DataGen

独立、可复用的 Top1 意图路由训练数据生成项目。它只负责计划、生成、独立审计、断点续跑
与产物治理，不包含模型训练或推理代码。

## 能力

- 由候选注册表、LabelDesc、内容轴和边界规则确定性构建生成计划。
- 支持单轮边界样本、同意图多轮和完整有向 IntentChange 组合。
- 生成、盲标、复核、直接性、对话质量、contrast、plan-fidelity 和 observed-axis
  全部使用严格 JSON Schema。
- 双模型一致性、隐私、消息结构、内容轴和最终意图均为 fail-closed 门禁。
- API 阶段级熔断；失败轮次不消耗样本 attempt，恢复后仅复用 request hash 与
  sample attempt 完全一致的 completed raw。
- manifest 固化配置、候选、taxonomy、endpoint 与实现哈希，支持可审计断点续跑。

当前 v2 配置生成 1,500 条：15 类各 100 条，包括 450 条单轮、630 条 IntentChange
和 420 条其它多轮，覆盖 101 个内容轴。

## 安装

```bash
cd /Users/tourxu/Codes/Top1DataGen
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
```

复制凭据模板到 Git 忽略的本地文件，并设置 HTTPS endpoint：

```bash
cp credentials.example credentials
chmod 600 credentials
```

真实凭据格式：

```text
base_url:https://your-gateway.example/v1
api_key:your-key
```

## 使用

先只生成不可变计划，不调用模型：

```bash
top1-generate --plan-only --credentials-file credentials
```

再运行每个内容轴一条的 pilot：

```bash
top1-generate --axis-pilot-per-axis 1 --credentials-file credentials
```

逐条人工审核 pilot 后，移除 scope 参数，在相同输出目录续跑全量：

```bash
top1-generate --credentials-file credentials
```

默认输入：

- `configs/top1_synthesis_v2.json`
- `configs/top1_candidates_v2.json`
- `data_top1/top1_labeldesc_v2.jsonl`

默认输出目录为 `data_top1/generated/top1_controlled_multiturn_v2`。运行中会维护：

- `manifest.json`：不可变输入、endpoint 和实现哈希；
- `plans.jsonl`：确定性生成蓝图；
- `attempts.jsonl`：已提交轮次；
- `accepted_records.jsonl`：完整通过审计的内部记录；
- `rejected.jsonl`：耗尽重试的样本及原因；
- `raw/`：生成与各审计阶段的原始响应；
- `train.jsonl`：可直接交给训练端的最终数据；
- `summary.json`：分布、拒绝原因、token 用量与产物哈希。

`raw/`、缓存、运行目录和真实凭据均被 Git 忽略。只有人工复核、版本化且带 provenance
和 validation summary 的训练数据才应提交。

## 扩展 taxonomy

新增或修改意图时必须同步更新 candidate registry、LabelDesc、内容轴定义与优先级、
轴级现象约束、单轮近邻 contrast 和后端映射。先运行 `--plan-only` 验证覆盖，再运行
axis pilot；不能直接把旧 taxonomy 数据混入新版本。

## 验证

```bash
uv run --no-sync python -m unittest discover -s tests -v
uv run --no-sync python -m compileall -q src scripts tests
```
