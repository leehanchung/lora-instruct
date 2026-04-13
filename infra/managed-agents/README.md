# Managed Agents Platform

Self-hosted managed agents platform with Firecracker microVM sandboxing.
Run heterogeneous agent harnesses (Claude Code, SWE-agent, OpenHands,
Aider, custom) in isolated microVMs with full kernel-level separation.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Control Plane (K8s)                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                 │
│  │ API      │  │ Postgres │  │ Redis    │                 │
│  │ :8000    │  │ sessions │  │ pub/sub  │                 │
│  └────┬─────┘  └──────────┘  └──────────┘                 │
│       │                                                     │
│       │  Hybrid Orchestrator                                │
│       │  (scheduler → select worker → POST /sessions)       │
└───────┼─────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────────┐  ┌───────────────────┐
│  Worker Node 1    │  │  Worker Node 2    │
│  Node Agent :9090 │  │  Node Agent :9090 │
│                   │  │                   │
│  ┌─────────────┐  │  │  ┌─────────────┐  │
│  │ FC microVM  │  │  │  │ FC microVM  │  │
│  │ claude-code │  │  │  │ swe-agent   │  │
│  └─────────────┘  │  │  └─────────────┘  │
│  ┌─────────────┐  │  │  ┌─────────────┐  │
│  │ FC microVM  │  │  │  │ FC microVM  │  │
│  │ openhands   │  │  │  │ aider       │  │
│  └─────────────┘  │  │  └─────────────┘  │
└───────────────────┘  └───────────────────┘
```

Each Firecracker microVM gets its own Linux kernel, filesystem, and
network stack. ~125ms boot, ~5MB memory overhead.

## Quick Start

```bash
# 1. Local dev (postgres + redis)
cd docker && docker compose up -d

# 2. Start the API
cd .. && uvicorn src.api.main:app --port 8000

# 3. Start a worker node agent
ORCHESTRATOR_BACKEND=hybrid \
ORCHESTRATOR_WORKER_ADDRESSES=http://localhost:9090 \
uvicorn src.orchestrator.node_agent:app --port 9090

# 4. Build rootfs for a harness and launch
python -m src.orchestrator.rootfs_builder --harness claude-code
```

## Project Structure

```
managed-agents/
├── docker/                 Dockerfiles + docker-compose (local dev)
├── harnesses/              Per-harness Dockerfiles (→ rootfs source)
├── deploy/                 Helm chart, K8s manifests, validation
├── examples/               Session configs, agent configs
├── src/
│   ├── api/                Control plane REST API (FastAPI)
│   │   └── routes/         /v1/agents, /v1/sessions, /v1/environments, events
│   ├── db/                 SQLAlchemy models (agents, sessions, events)
│   ├── orchestrator/       Core platform
│   │   ├── base.py         Abstract orchestrator + shared types
│   │   ├── scheduler.py    Constraint-based bin-packing scheduler
│   │   ├── harness.py      Harness definitions (rootfs + entrypoint + policy)
│   │   ├── sandbox.py      FirecrackerSandbox (microVM lifecycle)
│   │   ├── node_agent.py   Per-worker FastAPI managing VMs
│   │   ├── rootfs_builder  Dockerfile → ext4 rootfs converter
│   │   ├── hybrid.py       Hybrid orchestrator (K8s + FC workers)
│   │   └── k8s.py          K8s Jobs backend (legacy)
│   └── runtime/            Runs inside the sandbox
│       ├── claude_code.py  Claude Code session builder
│       ├── launcher.py     Multi-session YAML launcher
│       ├── agent_loop.py   Custom Python agent loop
│       └── llm/, tools/    LLM clients + built-in tools
├── pyproject.toml
├── Makefile
└── README.md
```

## API Endpoints

### Control Plane (port 8000)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/agents` | Create agent definition |
| GET | `/v1/agents` | List agents |
| POST | `/v1/environments` | Create environment config |
| POST | `/v1/sessions` | Create session (boots a microVM) |
| GET | `/v1/sessions` | List sessions |
| DELETE | `/v1/sessions/{id}` | Terminate session (kills VM) |
| POST | `/v1/sessions/{id}/events` | Send events (messages, tool results) |
| GET | `/v1/sessions/{id}/stream` | SSE event stream |

### Worker Node Agent (port 9090, internal)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/sessions` | Boot a microVM |
| DELETE | `/sessions/{id}` | Terminate VM |
| GET | `/sessions/{id}/logs` | Stream VM console log |
| GET | `/harnesses` | Available harnesses + rootfs status |
| GET | `/health` | Worker capabilities |

## Built-in Harnesses

| Name | Description | Default Resources |
|------|-------------|-------------------|
| `claude-code` | Claude Code CLI + MCP plugins | 2 vCPU, 2GB RAM |
| `claude-code-discord` | Claude Code + Discord bot | 2 vCPU, 2GB RAM |
| `custom-python` | Custom Python agent loop | 2 vCPU, 2GB RAM |
| `swe-agent` | Princeton's SWE-agent | 2 vCPU, 4GB RAM |
| `openhands` | OpenHands (CodeAct agent) | 4 vCPU, 8GB RAM |
| `aider` | AI pair programming | 1 vCPU, 1GB RAM |

## Deployment

### Production (K8s control plane + bare-metal workers)

```bash
# Deploy control plane
helm upgrade --install managed-agents deploy/helm/managed-agents/ \
  -f deploy/helm/managed-agents/values-production.yaml

# On each worker: install Firecracker, build rootfs, start node agent
# See deploy/ for details.
```

### Multi-session launcher

```bash
python -m src.runtime.launcher --config examples/sessions.yaml
```

## Development

```bash
poetry install
poetry run ruff check .          # Lint
poetry run pytest                # Test
```
