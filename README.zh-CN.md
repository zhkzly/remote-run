# remote-run

[English](README.md) | **简体中文**

> **本机编写代码与 AI 协作（Claude Code / Codex / Antigravity `agy`），远端任意 Linux 主机透明执行 —— 不限语言、GPU 可选、远端零安装。**

把编辑器、AI Agent 与凭据全部留在本机笔记本；通过 OpenSSH 按需投递唯一一份零外部依赖的 `remote_worker.py`，基于单一清单安全同步，在远端直接以清晰的 `PATH` + `env` 运行 Rust / Go / C++ / Node / Python 等任意工具链。

---

## 核心架构（4 个正交组件 + 统一交互层）

1. **组件 1 · 连接与通道（`remote_run_local/transport.py`）**：OpenSSH ControlMaster 复用、按需投递唯一独立 `remote_worker.py`（仅 stdlib）、帧式 RPC、SSH 255 缺失 `SSH_AUTH_SOCK` 客户端诊断。
2. **组件 2 · 空间与数据流（`remote_run_local/transfer.py`）**：远端 4 级目录隔离（`root -> projects/<project> -> workspaces/<ws> -> runs/<id>`）、无语言黑名单的单一发送清单、未跟踪文件体积预算（默认 `256 MiB`）、归属校验幽灵清理、去后缀化的原子 `pull`。
3. **组件 3 · 执行与环境（`remote_worker.py`）**：纯 `PATH` + 字面量 `env` 分层叠加、严格互斥的 `argv` / `shell`、`tmux` 后台保活 + 流式 `--follow` + 可选 `--session`、UTF-8 码点对齐日志分页、100% 只读通用 `doctor`。
4. **组件 4 · 可选资源切面**：主机不配 `[gpu]` 即纯 CPU 无锁并发；声明 `[gpu]` 时默认仍不占卡（清空 `CUDA_VISIBLE_DEVICES`），通过绑定或 `--gpus 0,1` 按 NVIDIA UUID 排序防死锁加锁并透明回报排队状态。

---

## 三层配置（严格分离所有权，`schema_version = 2`）

| 文件 | 归属与职责 | 是否进 Git / 是否上传 |
| --- | --- | --- |
| `~/.config/remote-run/hosts.toml` | **只描述机器本身**（`ssh`、`port`、`root`、机器全局 `path`/`env`、可选 `[gpu]`），零项目信息 | 否 / 否 |
| `<repo>/.remote-run.toml`（可选） | **团队通用项目策略**（`project`、工作区相对 `path`、通用 `[env]`、`[workspace].shared`、`[sync]`、`[pull]`） | 可提交 / 否 |
| `~/.config/remote-run/bindings/*.toml`（可选） | **本机私有绑定**（`remote-run bind` 写入）：`--scope repo` 由同仓库全部 worktree 自动继承；`--scope checkout` 仅覆盖当前 worktree（含自定义 `workspace`） | 否 / 否（零污染仓库） |

**统一优先级链**（从低到高）：
- `PATH`：`远端 SSH 基准 < hosts.toml < .remote-run.toml < repo 绑定 < checkout 绑定 < 单次 --path`（越具体越靠左）。
- `env`：`SSH 基准 < hosts.toml < .remote-run.toml < repo 绑定 < checkout 绑定 < 可选 session < 单次 --env < 工具保留变量`。

---

## 快速开始

```bash
# 1. 克隆并链接 CLI 与 MCP 启动器
git clone <repository-url> ~/remote-run
mkdir -p ~/.local/bin
ln -sf ~/remote-run/bin/remote-run ~/.local/bin/remote-run
ln -sf ~/remote-run/bin/remote-run-mcp ~/.local/bin/remote-run-mcp

# 2. 探测并注册远端机器（或手写 3 行 hosts.toml）
remote-run init linux-builder --name builder --default

# 3. 可选：为仓库绑定默认主机、远端预装 PATH 与数据目录（所有 worktree 自动继承；有 [gpu] 的主机可加 --gpus 0）
remote-run bind --scope repo --host builder \
  --note "Prebuilt env ready; invoke python/pytest directly" \
  --path /opt/envs/proj/bin \
  --env DATASET_DIR=/srv/datasets/v2

# 4. 查看最终合并配置与只读体检
remote-run config --pretty
remote-run doctor --check-cmd cargo --check-cmd /opt/envs/proj/bin/python --pretty

# 5. 直接执行任意语言命令
remote-run exec --follow -- cargo test
remote-run exec --follow -- python -u -m pytest
```

> **Codex MCP 注意**：Codex 启动 stdio MCP 默认过滤环境变量；如使用 `ssh-agent`，请在 `~/.codex/config.toml` 的 `[mcp_servers.remote-run]` 下增加 `env_vars = ["SSH_AUTH_SOCK"]`。

完整 CLI / MCP / Schema 说明见 [docs/usage.md](docs/usage.md) 与 [skills/remote-run/references/schema-and-setup.md](skills/remote-run/references/schema-and-setup.md)。
