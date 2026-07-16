# Team 1166 · KDD Cup 2026 DataAgent Solution

面向 [KDD Cup 2026 DataAgent-Bench](https://dataagent.top) 的数据分析 Agent 方案

Phase 1 定榜18名

## 快速开始

```bash
# 安装依赖
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync

# 配置模型凭据(环境变量优先于 configs/submission.yaml)
export MODEL_NAME=your-model
export MODEL_API_URL=https://your-endpoint/v1
export MODEL_API_KEY=sk-xxx

# 运行基准
uv run dabench run-benchmark --config configs/submission.yaml
```

Docker:

```bash
docker build -t team1166-solution .
docker run --rm \
  -v /path/to/input:/input:ro \
  -v /path/to/output:/output:rw \
  -e MODEL_NAME=your-model \
  -e MODEL_API_URL=https://your-endpoint/v1 \
  -e MODEL_API_KEY=sk-xxx \
  team1166-solution
```

## 目录结构

```
.
├── src/data_agent_baseline/   # 核心实现:ReAct 主循环、子 Agent、工具、数据加载
│   ├── agents/                 # 主循环、planner/verifier、模型适配、prompt
│   ├── tools/                   # 工具实现(文件/SQL/Python/PDF/视频等)
│   ├── benchmark/               # 数据集加载与 schema
│   └── run/                     # 单任务/批量运行入口
├── configs/submission.yaml     # 提交配置(凭据由环境变量注入)
├── Dockerfile
└── pyproject.toml / uv.lock
```

## License

[MIT](LICENSE)
