.PHONY: help install lint format test build push deploy clean

# Default target
help:
	@echo "Managed Agents - Make targets:"
	@echo ""
	@echo "Development:"
	@echo "  make install       - Install dependencies"
	@echo "  make lint          - Run linters (ruff, mypy)"
	@echo "  make format        - Format code with black and isort"
	@echo "  make test          - Run tests"
	@echo ""
	@echo "Docker:"
	@echo "  make build-api                 - Build API image"
	@echo "  make build-node-agent          - Build node agent image"
	@echo "  make build-harness-claude-code - Build Claude Code harness"
	@echo "  make build-harness-python      - Build custom Python harness"
	@echo "  make build-harness-swe-agent   - Build SWE-agent harness"
	@echo "  make build-harness-aider       - Build Aider harness"
	@echo "  make build-harnesses           - Build all harnesses"
	@echo "  make build                     - Build core images"
	@echo "  make build-all                 - Build everything"
	@echo ""
	@echo "Local deployment:"
	@echo "  make up            - Start services with docker compose -f docker/docker-compose.yaml"
	@echo "  make down          - Stop services"
	@echo "  make logs          - View logs"
	@echo ""
	@echo "Kubernetes deployment:"
	@echo "  make helm-install  - Install Helm chart"
	@echo "  make helm-upgrade  - Upgrade Helm chart"
	@echo "  make helm-delete   - Delete Helm chart"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Clean up build artifacts"

# ============================================================================
# Development targets
# ============================================================================

install:
	@echo "Installing dependencies..."
	poetry install

lint:
	@echo "Running linters..."
	poetry run ruff check .
	poetry run mypy src --ignore-missing-imports

format:
	@echo "Formatting code..."
	poetry run black .
	poetry run isort .

test:
	@echo "Running tests..."
	poetry run pytest tests/ -v --cov=src

# ============================================================================
# Docker targets
# ============================================================================

build-api:
	@echo "Building API Docker image..."
	docker build -f docker/Dockerfile.api -t managed-agents-api:latest .

build-node-agent:
	@echo "Building node agent Docker image..."
	docker build -f docker/Dockerfile.node-agent -t managed-agents-node-agent:latest .

# --- Harness images (one per agent runtime) ---

build-harness-claude-code:
	@echo "Building Claude Code harness..."
	docker build -f harnesses/claude-code.Dockerfile -t managed-agents-claude-code:latest .

build-harness-python:
	@echo "Building custom Python harness..."
	docker build -f harnesses/custom-python.Dockerfile -t managed-agents-runtime:latest .

build-harness-swe-agent:
	@echo "Building SWE-agent harness..."
	docker build -f harnesses/swe-agent.Dockerfile -t managed-agents-swe-agent:latest .

build-harness-aider:
	@echo "Building Aider harness..."
	docker build -f harnesses/aider.Dockerfile -t managed-agents-aider:latest .

build-harnesses: build-harness-claude-code build-harness-python build-harness-swe-agent build-harness-aider
	@echo "Built all harness images"

build: build-api build-harness-claude-code build-harness-python
	@echo "Built core images (api + default harnesses)"

build-all: build-api build-node-agent build-harnesses
	@echo "Built all images (api + node agent + all harnesses)"

# --- Rootfs images (Firecracker microVM) ---

rootfs-all:
	@echo "Building all rootfs images for Firecracker..."
	python -m src.orchestrator.rootfs_builder --all --dockerfile-dir .

rootfs-claude-code:
	python -m src.orchestrator.rootfs_builder --harness claude-code --dockerfile-dir .

rootfs-list:
	python -m src.orchestrator.rootfs_builder --list

push: build
	@echo "Pushing Docker images to registry..."
	docker push managed-agents-api:latest
	docker push managed-agents-runtime:latest

push-all: build-all
	@echo "Pushing all Docker images including node agent..."
	docker push managed-agents-api:latest
	docker push managed-agents-runtime:latest
	docker push managed-agents-node-agent:latest

# ============================================================================
# Local deployment with docker compose -f docker/docker-compose.yaml
# ============================================================================

up:
	@echo "Starting services with docker compose -f docker/docker-compose.yaml..."
	docker compose -f docker/docker-compose.yaml up -d
	@echo "Services are running. Check logs with: make logs"

down:
	@echo "Stopping services..."
	docker compose -f docker/docker-compose.yaml down

logs:
	docker compose -f docker/docker-compose.yaml logs -f

ps:
	docker compose -f docker/docker-compose.yaml ps

# ============================================================================
# Kubernetes / Helm targets
# ============================================================================

helm-install:
	@echo "Installing Helm chart..."
	helm install managed-agents ./deploy/helm/managed-agents \
		--namespace default \
		--create-namespace \
		-f deploy/helm/managed-agents/values.yaml

helm-upgrade:
	@echo "Upgrading Helm chart..."
	helm upgrade managed-agents ./deploy/helm/managed-agents \
		--namespace default \
		-f deploy/helm/managed-agents/values.yaml

helm-delete:
	@echo "Deleting Helm chart..."
	helm delete managed-agents --namespace default

helm-validate:
	@echo "Validating Helm chart..."
	helm lint ./deploy/helm/managed-agents
	helm template managed-agents ./deploy/helm/managed-agents | kubectl apply -f - --dry-run=client

helm-values:
	@echo "Showing Helm values..."
	helm template managed-agents ./deploy/helm/managed-agents

# ============================================================================
# Kubernetes debugging
# ============================================================================

k8s-status:
	@echo "Cluster status:"
	kubectl get nodes
	@echo ""
	@echo "Managed Agents resources:"
	kubectl get deployments,statefulsets,services,pods -l app.kubernetes.io/name=managed-agents

k8s-logs:
	@echo "API server logs:"
	kubectl logs -l app.kubernetes.io/name=managed-agents,component=api -f

k8s-shell-api:
	@echo "Opening shell in API pod..."
	kubectl exec -it $$(kubectl get pods -l app.kubernetes.io/name=managed-agents,component=api -o jsonpath='{.items[0].metadata.name}') -- /bin/bash

# ============================================================================
# Cleanup targets
# ============================================================================

clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.py[cod]" -delete
	rm -rf build dist *.egg-info .coverage htmlcov .pytest_cache
	docker compose -f docker/docker-compose.yaml down --volumes --remove-orphans
	@echo "Cleanup complete"

clean-images:
	@echo "Removing Docker images..."
	docker rmi managed-agents-api:latest managed-agents-runtime:latest || true
	@echo "Images removed"

distclean: clean clean-images
	@echo "Full cleanup complete"

# ============================================================================
# Development server
# ============================================================================

dev:
	@echo "Starting development server..."
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

db-migrate:
	@echo "Running database migrations..."
	poetry run alembic upgrade head

db-rollback:
	@echo "Rolling back database..."
	poetry run alembic downgrade -1

# ============================================================================
# Documentation targets
# ============================================================================

docs:
	@echo "Building documentation..."
	cd docs && poetry run make html
	@echo "Documentation built in docs/_build/html"
