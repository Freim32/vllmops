<h1 align="center">vllmops</h1>

<p align="center">
  <em>A tiny control plane for bare-metal <a href="https://github.com/vllm-project/vllm">vLLM</a> servers.</em>
</p>

<p align="center">
  <a href="https://github.com/Freim32/vllmops/actions"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/Freim32/vllmops/ci.yml?branch=main&label=CI&style=flat-square"></a>
  <a href="https://pypi.org/project/vllmops/"><img alt="PyPI" src="https://img.shields.io/pypi/v/vllmops?style=flat-square"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-green?style=flat-square"></a>
  <a href="https://github.com/astral-sh/ruff"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-261230?style=flat-square"></a>
  <a href="https://mypy-lang.org/"><img alt="mypy" src="https://img.shields.io/badge/types-mypy%20strict-blue?style=flat-square"></a>
</p>

<p align="center">
  <img src="docs/img/tui-running.png" alt="vllmops TUI" width="900">
</p>

---

## Overview

Self-hosted vLLM made simple. Declare each model in its own YAML file, group models into profiles in your project config, then drive their full lifecycle from either the CLI or a live TUI.

- **Git-friendly YAML.** One file per model, profiles in `.vllmops/config.yaml`. Reviewed in pull requests, reproducible on a fresh machine.
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
pipx install vllmops
```

Or with `uv`:

```bash
uv tool install vllmops
```

## Quickstart

```bash
mkdir my-llms && cd my-llms
vllmops init
uv sync                    # creates .venv with vLLM installed
vllmops create-model       # interactive: name, HF model, GPUs, port
vllmops start qwen3        # blocks on /health by default
vllmops tui                # live metrics
```

Layout after `init`:

```
my-llms/
├── .vllmops/config.yaml     # project config
├── configs/models/*.yaml    # one file per model
├── runtime/logs/            # rotated per spawn (.log + .log.prev)
├── runtime/pids/
├── pyproject.toml           # vLLM as a dep, installed via uv sync
└── .env.example
```

## Model YAML

`vllmops create-model --name qwen3 --model Qwen/Qwen3-8B --gpus 0 --port 8001` writes:

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

Group models for bulk lifecycle. Declare in `.vllmops/config.yaml`:

```yaml
profiles:
  dev: [qwen3, llama-small]
  prod: [qwen3-prod]
```

Then run lifecycle commands on the whole group. Each member is processed in parallel; already-running members are skipped (idempotent), broken YAMLs don't block the rest, failures are reported per-model:

```bash
vllmops start --profile dev      # parallel spawn + parallel /health wait
vllmops stop --profile dev
vllmops restart --profile dev
vllmops profile list             # all profiles with running/total counts
vllmops profile show dev         # members and their state
```

Models not declared in any profile fall into the synthetic `general` group. The TUI sidebar renders the same grouping; selecting a profile node makes `s`/`S`/`r` operate on every member.

## Commands

| Command | Description |
| --- | --- |
| `vllmops init [PATH]` | Initialize a project workspace |
| `vllmops create-model` | Scaffold a model YAML |
| `vllmops validate` | Validate all model YAMLs |
| `vllmops start <name> \| --profile <p>` | Spawn one model or every model in a profile |
| `vllmops stop <name> \| --profile <p>` | SIGTERM, then SIGKILL after timeout |
| `vllmops restart <name> \| --profile <p>` | Stop, then start |
| `vllmops status [<name>]` | Running / stale / stopped |
| `vllmops health <name>` | One-shot `/health` probe |
| `vllmops logs <name> [--tail N] [-f]` | Print or follow a model log |
| `vllmops command <name>` | Print the underlying vLLM command |
| `vllmops profile list \| show <p>` | Inspect profiles defined in config |
| `vllmops tui` | Launch the Textual TUI |
| `vllmops doctor` | Diagnose local setup (Python, venv, vllm, GPUs, ports, ...) |
| `vllmops completion <shell>` | Print shell completion script (bash, zsh, fish, powershell) |

Run `vllmops <command> --help` for full options.

## Shell completion

```bash
# bash
vllmops completion bash > ~/.local/share/bash-completion/completions/vllmops

# zsh (ensure `fpath+=~/.zfunc` and `autoload -U compinit && compinit` are in your .zshrc)
vllmops completion zsh > ~/.zfunc/_vllmops

# fish
vllmops completion fish > ~/.config/fish/completions/vllmops.fish
```

Restart your shell. Alternative: `vllmops --install-completion` auto-detects the current shell and installs in one step.

## Contributing

Contributions of any size are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for local setup and the project checks.

## License

[Apache-2.0](LICENSE)
