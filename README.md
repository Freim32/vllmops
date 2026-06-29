<h1 align="center">vllmctl</h1>

<p align="center">
  <em>A tiny control plane for bare-metal <a href="https://github.com/vllm-project/vllm">vLLM</a> servers.</em>
</p>

<p align="center">
  <a href="https://github.com/Freim32/vllmctl/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Freim32/vllmctl/ci.yml?branch=main&label=CI&style=flat-square"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-261230?style=flat-square"></a>
  <a href="https://mypy-lang.org/"><img alt="mypy" src="https://img.shields.io/badge/types-mypy%20strict-blue?style=flat-square"></a>
</p>

<p align="center">
  <img src="docs/img/tui-running.png" alt="vllmctl TUI" width="900">
</p>

---

## Overview

Self-hosted vLLM made simple. Declare each model in its own YAML file, group models into profiles in your project config, then drive their full lifecycle from either the CLI or a live TUI.

- **Git-friendly YAML.** One file per model, profiles in `.vllmctl/config.yaml`. Reviewed in pull requests, reproducible on a fresh machine.
- **Full lifecycle.** `start`, `stop`, `restart`, `status`, `health`, `logs`. Single model or whole profile, in parallel.
- **CLI and TUI, same actions.** Run from the terminal in scripts, or open the TUI for a live view.
- **Live metrics, no stack.** Direct `/metrics` scrape, in-memory ring buffer. No Docker, no Prometheus, no Grafana.
- **Per-project venv.** Each workspace pins its own vLLM via `uv`. No global install required.
- **POSIX, type-checked, tested.** Linux/macOS, mypy strict, 180+ tests.

## Contents

- [Install](#install)
- [Quickstart](#quickstart)
- [Model YAML](#model-yaml)
- [Profiles](#profiles)
- [Commands](#commands)
- [Shell completion](#shell-completion)
- [Contributing](#contributing)
- [License](#license)

## Install

Requires Python 3.10+ on Linux or macOS.

```bash
pipx install git+https://github.com/Freim32/vllmctl
```

Or with `uv`:

```bash
uv tool install git+https://github.com/Freim32/vllmctl
```

## Quickstart

```bash
mkdir my-llms && cd my-llms
vllmctl init
uv sync                    # creates .venv with vLLM installed
vllmctl create-model       # interactive: name, HF model, GPUs, port
vllmctl start qwen3        # blocks on /health by default
vllmctl tui                # live metrics
```

Layout after `init`:

```
my-llms/
├── .vllmctl/config.yaml     # project config
├── configs/models/*.yaml    # one file per model
├── runtime/logs/            # rotated per spawn (.log + .log.prev)
├── runtime/pids/
├── pyproject.toml           # vLLM as a dep, installed via uv sync
└── .env.example
```

## Model YAML

`vllmctl create-model --name qwen3 --model Qwen/Qwen3-8B --gpus 0 --port 8001` writes:

```yaml
name: qwen3
env:
  CUDA_VISIBLE_DEVICES: '0'
  HF_HOME: data/huggingface
  VLLM_LOGGING_LEVEL: INFO
vllm:
  executable: vllm
  subcommand: serve
  model: Qwen/Qwen3-8B
  args:
    --host: 0.0.0.0
    --port: 8001
    --tensor-parallel-size: 1
    --dtype: auto
    --served-model-name: qwen3
    --disable-access-log-for-endpoints: /health,/metrics,/ping
  flags: []
  extra_args: []
metrics:
  path: /metrics
```

`env` supports `${VAR}` interpolation from the shell, `.env`, and the project config (shell wins). Add `HF_TOKEN: ${HF_TOKEN}` for gated models.

## Profiles

Group models for bulk lifecycle. Declare in `.vllmctl/config.yaml`:

```yaml
profiles:
  dev: [qwen3, llama-small]
  prod: [qwen3-prod]
```

Then run lifecycle commands on the whole group. Each member is processed in parallel; already-running members are skipped (idempotent), broken YAMLs don't block the rest, failures are reported per-model:

```bash
vllmctl start --profile dev      # parallel spawn + parallel /health wait
vllmctl stop --profile dev
vllmctl restart --profile dev
vllmctl profile list             # all profiles with running/total counts
vllmctl profile show dev         # members and their state
```

Models not declared in any profile fall into the synthetic `general` group. The TUI sidebar renders the same grouping; selecting a profile node makes `s`/`S`/`r` operate on every member.

## Commands

| Command | Description |
| --- | --- |
| `vllmctl init [PATH]` | Initialize a project workspace |
| `vllmctl create-model` | Scaffold a model YAML |
| `vllmctl validate` | Validate all model YAMLs |
| `vllmctl start <name> \| --profile <p>` | Spawn one model or every model in a profile |
| `vllmctl stop <name> \| --profile <p>` | SIGTERM, then SIGKILL after timeout |
| `vllmctl restart <name> \| --profile <p>` | Stop, then start |
| `vllmctl status [<name>]` | Running / stale / stopped |
| `vllmctl health <name>` | One-shot `/health` probe |
| `vllmctl logs <name> [--tail N] [-f]` | Print or follow a model log |
| `vllmctl command <name>` | Print the underlying vLLM command |
| `vllmctl profile list \| show <p>` | Inspect profiles defined in config |
| `vllmctl tui` | Launch the Textual TUI |
| `vllmctl doctor` | Diagnose local setup (Python, venv, vllm, GPUs, ports, ...) |
| `vllmctl completion <shell>` | Print shell completion script (bash, zsh, fish, powershell) |

Run `vllmctl <command> --help` for full options.

## Shell completion

```bash
# bash
vllmctl completion bash > ~/.local/share/bash-completion/completions/vllmctl

# zsh (ensure `fpath+=~/.zfunc` and `autoload -U compinit && compinit` are in your .zshrc)
vllmctl completion zsh > ~/.zfunc/_vllmctl

# fish
vllmctl completion fish > ~/.config/fish/completions/vllmctl.fish
```

Restart your shell. Alternative: `vllmctl --install-completion` auto-detects the current shell and installs in one step.

## Contributing

Contributions of any size are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and the project checks.

## License

[Apache-2.0](LICENSE)
