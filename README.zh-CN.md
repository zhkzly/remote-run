# remote-run

[English](README.md) | **简体中文**

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero%20(stdlib)-brightgreen.svg)](remote_run.py)
[![MCP Server](https://img.shields.io/badge/MCP-stdio%20server-purple.svg)](remote_run_mcp.py)
[![Agent Skill](https://img.shields.io/badge/Agent%20Skill-SKILL.md-orange.svg)](skills/remote-run/SKILL.md)
[![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

> **把 Claude Code、Codex、Antigravity、你的 API Key、MCP、Skill 和编辑器全留在自己笔记本；把任意 Linux GPU 主机变成零配置纯算力后端。**

不用在租来的云 GPU 或实验室机器上重装 AI 工具，不用把密钥和代理放到别人的电脑上，也不用为了跑测试/训练而在多个机器间反复 Git commit / push / pull。

---

## 为什么做 `remote-run`？

当你用轻薄本上的 **Claude Code / Codex / Antigravity (`agy`)** 写代码，而算力在台式机、实验室工作站、学校超算或租来的云卡上时，传统做法往往很折腾：

- **VS Code Remote-SSH**：相当于把远端当成本地开发主机。每次换机器或新租一张卡，都要在服务器上重新配置一遍 AI CLI、代理、API Key、MCP 和工作环境；在别人电脑或共享服务器上留配置既麻烦又“烧心”，而且不少集群根本连不上外部 LLM API。
- **Git `push` / `pull` 来回切换**：本地改几行代码就得提交推送，切到服务器拉取运行，发现 bug 再切回来改——commit 记录全是 `wip`，心智切换极累。
- **手写 `ssh` + `rsync` 脚本**：笔记本合盖或断网任务就断、多任务抢显存，稍不注意 `rsync --delete` 还会把远端 `.venv` 或辛苦跑出来的权重文件误删。

**`remote-run` 把“开发环境”和“算力节点”彻底解耦：**

```text
你的笔记本（所有智能与凭证留在这里）
  编辑器 + Claude Code / Codex / agy + API Key + MCP + Skill + 个人记忆
        │
        │  SSH + 增量只增不删同步（对用户几乎无感）
        ▼
任意 Linux 服务器 / 实验室台式机 / 租用云 GPU（零 AI 配置！）
  ├─ 按项目与分支自动隔离远端工作区（移动本地目录/多 worktree 不串目录）
  ├─ tmux 断连保活 + 物理 GPU flock 排队 + 干净释放显存取消
  ├─ 自动保护远端 .venv/、outputs/、checkpoints/、.cache/ 不被覆盖或误删
  └─ 运行完自动拉回日志、指标与小产物（默认跳过大体积 checkpoint）
```

## 方案对比

| 方式 | 远端需要配 AI / 密钥 / 代理吗？ | 借用他人电脑 / 临时租卡即插即用？ | 每次调试都需要 Git 提交同步？ | 笔记本断网/休眠保活？ | 防误删与 GPU 锁 | AI Agent 原生友好（MCP/Skill） |
|---|---|---|---|---|---|---|
| **`remote-run`** | **不需要（远端零 AI 配置）** | **支持（用完即走）** | **不需要（增量直传）** | **支持（`tmux` + `--wait`）** | **内置所有权校验与 `flock`** | **原生支持（CLI + MCP + `SKILL.md`）** |
| **VS Code Remote-SSH** | 需要（换一台配一次） | 隐私与迁移成本高 | 不需要 | 部分支持 | 无 | 需把 Agent 装在远端 |
| **Git push/pull 来回切** | 若远端用 AI 则需要 | 是 | 需要（大量无效提交） | 需手动套 `tmux` | 无 | 需手动编排 |
| **手写 `ssh` + `rsync`** | 不需要 | 是 | 不需要 | 需手动套 `tmux` | 容易误用 `--delete` 删数据 | 非结构化输出，Agent 易出错 |

---

## 典型使用场景

1. **临时租云卡 / 借用实验室机器 / 超算**：你的所有 AI 工具、提示词工程和 API Key 完全留在自己笔记本，用起来不“烧心”；远端只要是个能 SSH 登录的 Linux 用户账号（**无需 `sudo`**）就能直接开跑。
2. **日常写代码 + 跑测试/构建（不限于训练）**：本地改完 Python / Rust / Go / C++ 代码，直接 `remote-run exec -- pytest -q` 或让 AI 自动调用 MCP 工具同步变更并回传堆栈，告别 Git 来回切换。
3. **长时间 GPU 训练与实验**：运行超过 `--wait`（默认 15s）自动转入远端 `tmux` 后台并返回 `run_id`；同卡任务按 `gpu_slot` 自动排队；支持 `status` 增量拉日志（自动折叠 tqdm 刷屏）和 `cancel` 按进程组清显存。
4. **直接在远端下载大数据集 / 初始化环境**：配合 `--no-sync` 与 CPU 通道（`--host cpu_box` 或 `--no-gpu`），让台式机直接执行 `hf download`、`dvc pull`、解压或 `uv sync`，不占用笔记本硬盘与上行带宽，也不阻塞 GPU 队列。
5. **跨端协同**：同一套 `remote-run` 同时暴露 **CLI**、标准化 **Agent Skill** 和 **stdio MCP Server**，可在 **Claude Code**、**Codex**、**Antigravity (`agy`)** 中开箱即用。

---

## 2 分钟快速上手

### 环境要求

- **本地笔记本**：Linux / macOS，已安装 Python 3.11+、`ssh`、`rsync`
- **远端 Linux 主机**：SSH `PATH` 下有 `python3`（3.11+）、`git`、`tmux`、`rsync`，且已配置免密 SSH 别名（`ssh -o BatchMode=yes <alias> true` 可通），**全流程不需要 `sudo`**

### 1. 克隆仓库并安装入口命令

```bash
git clone https://github.com/zhkzly/remote-run.git ~/remote-run
mkdir -p ~/.local/bin
ln -sf ~/remote-run/bin/remote-run ~/.local/bin/remote-run
ln -sf ~/remote-run/bin/remote-run-mcp ~/.local/bin/remote-run-mcp
```

### 2. 配置主机信息（`~/.config/remote-run/hosts.toml`）

```toml
schema_version = 1
default_host = "desktop"

[hosts.desktop]
ssh = "my-gpu-box"             # ~/.ssh/config 中的免密主机别名
root = "/data/user/remote-jobs"
locks_dir = "/data/user/remote-locks"
device = "cuda:0"
gpu_slot = "gpu-0"
default_env = "venv"           # 远端工作区有 .venv 时自动激活
path_prepend = ["/home/user/.local/bin"]

# 可选：与 GPU 主机共用同一个 root，用于测试/下载等不抢 GPU 锁的任务
[hosts.cpu_box]
ssh = "my-gpu-box"
root = "/data/user/remote-jobs"
locks_dir = "/data/user/remote-locks"
device = "cpu"
default_env = "venv"
```

检查远端连通性与环境：

```bash
remote-run doctor --pretty
# 如果指定的 root 是远端已存在且非空的目录，首次需确认归属：
remote-run doctor --init-root
```

### 3. 接入 Claude Code / Codex / Antigravity (`agy`)

```bash
# 安装 Agent Skill（教 Agent 正确使用并遵守安全规则）
mkdir -p ~/.agents/skills ~/.claude/skills
ln -sfn ~/remote-run/skills/remote-run ~/.agents/skills/remote-run   # Codex & agy
ln -sfn ~/remote-run/skills/remote-run ~/.claude/skills/remote-run   # Claude Code

# 注册零依赖 stdio MCP Server（提供原生 remote_* 工具）
claude mcp add -s user remote-run -- ~/.local/bin/remote-run-mcp
codex mcp add remote-run -- ~/.local/bin/remote-run-mcp
agy mcp add remote-run ~/.local/bin/remote-run-mcp
```

### 4. 日常常用命令

```bash
# 自动增量同步代码并在 CPU 通道跑测试（自动使用远端项目 .venv）
remote-run exec --host cpu_box -- pytest -q

# 跑 GPU 训练（超过 15s 自动转后台，GPU 锁排队，失败返回退出码 2 并保留 JSON）
remote-run exec -- python train.py --epochs 20

# 不同步本地文件、直接让远端从 HuggingFace/DVC 拉数据到指定目录
remote-run exec --host cpu_box --no-sync --remote-dir /data/user/datasets/ds --shell "hf download ... --local-dir ."

# 可选开启跨命令持久会话（仅在 exit_code == 0 时保留 cd 与环境变量变化）
remote-run exec --session dev --shell "cd sub && export TAG=v1"
remote-run exec --session dev --no-sync -- python eval.py

# 查看增量日志、安全取消清理显存、拉回日志与小体积结果
remote-run status RUN_ID --wait 30 --print-log
remote-run cancel RUN_ID
remote-run pull RUN_ID ./results/RUN_ID
```

---

## 面向 AI Agent 的代码级安全护栏

把远端执行交给自主 AI Agent 时，安全不能只靠提示词约束，核心规则全部在代码中硬性执行：

1. **默认只增不删**：`sync` 与 `exec` 默认 `rsync` **不带 `--delete`**，绝不会因为本地没存远端的数据/结果就把远端文件删掉；删除需显式 `remote-run sync --dry-run --delete` 预览后手动确认 `sync --delete`。
2. **机器无关的双重归属标记**：绝不默认覆盖或写入陌生非空目录。根目录受 `.remote-run-root.json` 保护（`doctor --init-root`），工作区受 `.remote-run-workspace.json`（含 `project_id` / `root_commit` / `workspace_id`）保护；接入已有非空目录需显式执行一次 `sync --adopt`。
3. **动态拓扑隔离校验**：动态解析远端 `$HOME` 与 `root`，拒绝 `/`、`$HOME` 及祖先路径、控制目录（`runs/`、`locks/`、`sessions/`、`projects/*/shared/`）以及 `pull` 中的软链越界。
4. **远端产物保护**：`.venv/`、`outputs/`、`checkpoints/`、`.cache/`、`[sync].preserve` 与 `[share].project` 默认禁止本地覆盖且免受删除。
5. **白名单清理与分级拉取**：`prune` 默认只读 dry-run，`--apply` 只删除经核验确认为 `remote-run` 生成且 `tmux` / `worker_pid` / `PGID` 已结束的旧作业目录；`pull` 默认跳过权重后缀及 >10 MB 文件，按需传 `--all-artifacts` 才拉取。

---

## MCP Server 工具一览（`remote-run-mcp`）

零外部依赖的 JSON-RPC 2.0 stdio 服务（支持 `2025-06-18` / `2025-03-26` / `2024-11-05` 协议及标准 `annotations`）：

| 工具名 | 用途 | 安全边界 |
|---|---|---|
| `remote_hosts` | 列出主机、设备、Python 环境与共享资产 | 只读 |
| `remote_config` | 查看生效的 `hosts.toml` + `.remote-run.toml` 配置、工作区与忽略规则 | 只读 |
| `remote_doctor` | 检查 SSH、工具链、GPU 列表、Python 包导入、CUDA 与共享数据 SHA-256 | 只读检查 |
| `remote_sync` | 单向增量同步代码（支持 `dry_run: true` 只读预览） | 有意不暴露 `delete` 与 `adopt` |
| `remote_exec` | 增量同步（可配 `no_sync: true`）并在远端执行命令或 shell | 独立 PGID + `tmux` 隔离 |
| `remote_status` | 按字节偏移查询日志与状态，或列出近期任务 | 只读 |
| `remote_cancel` | 安全终止排队或运行中的任务并释放显存，保留已有日志 | 标记为 `destructiveHint` |
| `remote_pull` | 拉取日志、元数据、小体积产物及指定 `paths` | 默认跳过 checkpoint 与超限文件 |
| `remote_session` | 列出或重置可选的持久会话状态 | 显式触发 |
| `remote_prune` | 预览或清理过期终态运行目录（默认 `apply: false` 只读） | 仅清理白名单终态目录 |

---

## 更多文档与示例

- **[docs/usage.md](docs/usage.md)** — 完整 CLI 手册、多卡隔离、`[share].project` 跨 worktree 共享、`[share].host` SHA-256 校验与环境变量占位符
- **[skills/remote-run/SKILL.md](skills/remote-run/SKILL.md)** 与 **[references/schema-and-setup.md](skills/remote-run/references/schema-and-setup.md)** — Agent Skill 规范与双文件 TOML 严格校验说明
- **[examples/hosts.example.toml](examples/hosts.example.toml)** 与 **[examples/project.example.toml](examples/project.example.toml)** — 配置示例模板
- **运行单元测试**：`python3 -B -m unittest discover -s tests -v`

## License

[MIT](LICENSE)
