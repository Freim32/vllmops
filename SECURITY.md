# Security Policy

## Supported versions

Only the latest minor release receives security fixes. There is no LTS branch.

## Reporting a vulnerability

If you find a security issue, please do not open a public GitHub issue. Email `matben32@gmail.com` with:

- A description of the issue and the affected version.
- Steps to reproduce, or a minimal proof of concept.
- Any logs or context that help reproduce locally.

You can expect a first reply within a few business days. Fixes will be coordinated privately and disclosed in the release notes once a patched version is published.

## Scope

`vllmops` spawns local processes, reads YAML configuration from the project workspace and scrapes `/metrics` from servers it started. It does not bind network sockets itself, does not authenticate users and does not store credentials beyond what the user puts in `.env` files.

Out of scope for security reports:

- Misuse of vLLM itself (those belong upstream).
- Security of the model weights or the data served through them.
- Permission issues caused by running the CLI as root or with custom umasks.
