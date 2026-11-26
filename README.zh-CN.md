# remote-run

[English](README.md) | **简体中文**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib)-brightgreen.svg)](remote_run.py)
[![MCP Server](https://img.shields.io/badge/MCP-stdio%20server-purple.svg)](remote_run_mcp.py)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-orange.svg)](skills/remote-run/SKILL.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **本机编写代码与运行 AI Agent（Claude Code / Codex / Antigravity），远程 Linux 透明执行——不限语言、不限 GPU，远端零 AI 配置，几乎零配置即可开跑。**

编辑器、AI Agent 和 MCP 服务留在笔记本。项目文件按同步规则上传，命令在具备先决工具的 Linux 主机上执行。

---

## 为什么做 `remote-run`？

当你在笔记本上用 **Claude Code / Codex / Antigravity (`agy`)** 编写代码，而需要借用台式机、云卡或测试机的算力时：

- **VS Code Remote-SSH**：每换一台机器就要重装一遍 AI CLI、密钥、代理和插件；在共享机/超算/临时云卡上留凭证既麻烦又难管理。
- **Git `push` / `pull` 来回切**：改两行代码就得提交推送再切换终端，commit 记录全是无效 `wip`。
- **手写 `ssh` + `rsync` 脚本**：每次命令都要重走 SSH 握手；本地删改文件在远端留下“幽灵文件”导致 Go/Rust/C++ 编译报错；乱加 `--delete` 又怕误删远端缓存和结果；多任务还会撞显存。

**`remote-run` 用 4 个正交运行时内核 + 1 个多端交互层彻底解耦：**

```text
你的笔记本（编辑器 + Claude Code / Codex / agy + API Key + MCP + 源码真源）
        │
        │  SSH ControlMaster 复用 + 带清单的安全增量同步（对用户近乎无感）
        ▼
任意远程 Linux 主机（无需 sudo、无需安装 AI 工具、无需常驻 Daemon）
  ├─ 连接与环境：支持 user@host / port / ~/ 路径、自动继承远端登录 Shell PATH、命名环境
  ├─ 空间与同步：单目录切分支保留构建缓存，真 worktree 自动隔离，4 类文件严格分治防误删
  ├─ 执行与周期：无状态 tmux 或按需 --session、--follow 实时流式输出、PGID 干净取消
  └─ 可选资源层：纯 CPU 主机无锁并发；GPU 主机自动按卡排序加锁（--device / --no-gpu 灵活切换）
```

---

## 核心场景

1. **全语言透明开发测试**：本地编写 **Rust / Go / C++ / Node / Python**，直接 `remote-run exec --follow -- cargo test` 或 `pytest`。内置 `.remote-run-synced.json` 清单会在你本地删除/重命名源码时自动清理对应远端“幽灵文件”，仅清理仍与登记版本一致的远端副本。远端自产文件和缓存放进 `[sync].preserve`。
2. **切分支不丢缓存**：普通单仓库 `git checkout -b` 默认保留在 `default` 远端工作区（已配置排除或保留的缓存可以继续用）；只有真开了多个 Git worktree 时才按工作树自动隔离，并支持 `[share].project` 共享目录。
3. **3 行配置 / 一键 `init` 上手**：`remote-run init user@1.2.3.4 --port 22345` 自动探测远程工具与显卡并写入 `hosts.toml`；全流程免交互 SSH、远端零 `sudo`。
4. **长时间 GPU 作业。** 作业从开始即运行于远端 `tmux`，超过本地 `--wait`（默认 15 秒）后继续运行。CUDA 作业在同一锁目录按卡号顺序加锁。取消会向已核实的进程组发信号。`--no-gpu` 跳过卡锁并清空 `CUDA_VISIBLE_DEVICES`，供遵守该变量的程序使用。
5. **AI Agent 原生友好**：同时提供 **CLI**、并发 **stdio MCP（`remote_*`）** 和渐进式 **Agent Skill**。支持字段见 [CLI 与 MCP 契约](docs/usage.md#cli-and-mcp-reference)。

---

## 快速上手

### 环境要求

- **本地笔记本**：Linux 或 macOS，Bash、PATH 上的 Python 3.11+、`ssh`、`rsync`。Git 提供项目身份和忽略规则
- **远端 Linux 主机**：Python 3.11+、`bash`、`tmux`、`rsync`（无需远端 `git`，无需 `sudo`），配置免交互 SSH，并预先验证主机公钥写入 `known_hosts`（`ssh -o BatchMode=yes <target> true`）

### 1. 克隆并链接入口命令

```bash
git clone https://github.com/zhkzly/remote-run.git ~/remote-run
mkdir -p ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
ln -sf ~/remote-run/bin/remote-run ~/.local/bin/remote-run
ln -sf ~/remote-run/bin/remote-run-mcp ~/.local/bin/remote-run-mcp
```

### 2. 一键初始化或手写极简 `hosts.toml`

用 `init` 自动探测写入 `~/.config/remote-run/hosts.toml`：

```bash
remote-run init my-server           # 或: remote-run init user@1.2.3.4 --port 22345 --name box
```

也可以直接写 **3 行极简配置**（`root` 默认远端 `~/.local/share/remote-run`，`device` 默认无锁 `cpu`）：

```toml
schema_version = 1

[hosts.server]
ssh = "my-server"                   # ~/.ssh/config 别名或 "user@host"
# port = 22345                      # 可选自定义 SSH 端口
# device = "cuda:0"                 # 有显卡时再开这一行：自动启用显存锁与 CUDA 隔离
```

检查连通性与工具链：

```bash
remote-run doctor --check-cmd cargo --pretty
# 若 root 指向远端已有非空目录且确认归本工具管理：
remote-run doctor --init-root
```

### 3. 接入 Claude Code / Codex / Antigravity (`agy`)

```bash
# 安装 Agent Skill
mkdir -p ~/.agents/skills ~/.claude/skills
ln -sfn ~/remote-run/skills/remote-run ~/.agents/skills/remote-run   # Codex & agy
ln -sfn ~/remote-run/skills/remote-run ~/.claude/skills/remote-run   # Claude Code

# 注册零依赖 stdio MCP Server
claude mcp add -s user remote-run -- ~/.local/bin/remote-run-mcp
codex mcp add remote-run -- ~/.local/bin/remote-run-mcp
agy mcp add remote-run ~/.local/bin/remote-run-mcp
```

### 4. 日常使用示例

```bash
# 增量同步 + 实时流式查看远程执行输出（Rust / Go / C++ / Node / Python 通用）
remote-run exec --follow -- cargo test
remote-run exec --env venv --follow -- uv run pytest

# 单次运行覆盖设备/跳过 GPU 锁（同一主机配置即可搞定多卡或 CPU 测试）
remote-run exec --no-gpu -- pytest -q
remote-run exec --device cuda:0,1 -- torchrun --nproc_per_node=2 train.py

# 不同步本地目录，直接让远端执行下载/环境安装
remote-run exec --no-gpu --no-sync --remote-dir '~/datasets/v1' --shell "hf download ... --local-dir ."

# 按需持久 shell 会话（仅在 exit_code == 0 时保留 cd 和环境变量变更）
remote-run exec --session dev --shell "cd sub && export TAG=v1"
remote-run exec --session dev --no-sync -- make test

# 跟踪长任务、干净取消、拉回日志与产物
remote-run status RUN_ID --follow
remote-run cancel RUN_ID
remote-run pull RUN_ID ./results/RUN_ID
```

---

## 面向共享服务器与 AI Agent 的安全护栏

1. **清理已登记的源码。** 默认同步仅在本地文件缺失且远端普通文件仍与登记版本一致时清理。远端自产文件放到保留路径。显式镜像前使用 `sync --dry-run --delete` 预览。
2. **托管目录。** 根、工作区和作业的标记用于核对归属。控制目录要求可信路径。已有非空根目录或工作区须显式初始化或认领。
3. **私有状态。** SSH socket、运行、会话和锁目录使用私有权限。常见本地凭据文件被排除，检查接口会遮蔽配置中的环境值。命令输出仍可能包含程序主动打印的内容。
4. **清理和拉取。** `prune` 默认只预览，跳过没有归属标记的运行。默认 `pull` 拉取日志、状态和符合条件的产物，自动选取的单个产物默认上限 10 MiB。显式工作区路径的规则见[使用手册](docs/usage.md)。

---

## MCP 工具列表（`remote-run-mcp`）

零依赖 JSON-RPC 2.0 并发 stdio 服务（支持 `2025-06-18` / `2025-03-26` / `2024-11-05` 协议）：

| 工具名 | 用途 | 安全边界 |
|---|---|---|
| `remote_hosts` | 列出主机、设备、命名环境与共享资产 | 只读 |
| `remote_config` | 校验并查看 `hosts.toml` + `.remote-run.toml` 生效配置 | 只读 |
| `remote_doctor` | 检查 SSH、`bash`/`tmux`/`rsync`、`check_cmds`、GPU、Python 导入与 SHA-256 | 可选的可执行探针 |
| `remote_sync` | 安全增量同步（含已上传幽灵文件清理） | 可能替换源码与清理已登记文件，可先用 `dry_run` |
| `remote_exec` | 同步并执行命令/shell（支持 `device`、`no_gpu`、`no_sync`、`wait`） | 独立 PGID + `tmux` 隔离 |
| `remote_status` | 轮询任务状态（含 `tmux_session`）、字节增量日志或列任务 | 只读 |
| `remote_cancel` | 请求取消并报告观察到的状态，保留日志 | `destructiveHint` |
| `remote_pull` | 拉取日志、元数据、产物或指定 `--path` 工作区文件 | 默认跳过大体积与权重产物 |
| `remote_session` | 列出或重置可选的持久 shell 会话 | 显式触发 |
| `remote_prune` | 预览或清理过期终态运行目录（默认只读） | 仅清理白名单终态目录 |

---

## 文档与测试

- **[docs/usage.md](docs/usage.md)** — 完整 CLI 手册、4 类同步内容划分表、多卡防死锁机制与占位符
- **[skills/remote-run/SKILL.md](skills/remote-run/SKILL.md)** 与 **[references/schema-and-setup.md](skills/remote-run/references/schema-and-setup.md)** — Agent Skill 与分级 Schema 规范
- **[examples/hosts.example.toml](examples/hosts.example.toml)** 与 **[examples/project.example.toml](examples/project.example.toml)** — 渐进式配置模板
- **运行单元测试**：`python3 -B -m unittest discover -s tests -v`

## License

[MIT](LICENSE)
