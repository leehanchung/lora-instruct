.PHONY: help sync sync-dev lint fmt check image restart deploy logs stop shell

# Configurable knobs — override with e.g. `make deploy IMAGE=myimage`.
BOT_NAME   ?= disco
IMAGE      ?= delulu-discord
ENV_FILE   ?= /root/disco.env
MODAL_TOML ?= /root/.modal.toml

help:
	@echo "Targets (VPS deploy):"
	@echo "  make sync          uv sync runtime deps"
	@echo "  make sync-dev      uv sync runtime + dev deps"
	@echo "  make check         ruff check + format --check"
	@echo "  make lint          ruff check --fix"
	@echo "  make fmt           ruff format"
	@echo "  make image         docker build the bot image"
	@echo "  make restart       stop+remove+run the $(BOT_NAME) container"
	@echo "  make deploy        image + restart (rebuild and relaunch the bot)"
	@echo "  make logs          tail bot container logs"
	@echo "  make stop          stop and remove the bot container"
	@echo "  make shell         exec a shell inside the running container"

sync:
	uv sync

sync-dev:
	uv sync --extra dev

check: sync-dev
	uv run ruff check .
	uv run ruff format --check .

lint: sync-dev
	uv run ruff check --fix .

fmt: sync-dev
	uv run ruff format .

image:
	docker build -t $(IMAGE) .

restart:
	-docker stop $(BOT_NAME)
	-docker rm $(BOT_NAME)
	docker run -d \
	  --name $(BOT_NAME) \
	  --restart=unless-stopped \
	  --env-file $(ENV_FILE) \
	  -v $(MODAL_TOML):/root/.modal.toml:ro \
	  $(IMAGE)

deploy: image restart

logs:
	docker logs -f $(BOT_NAME)

stop:
	-docker stop $(BOT_NAME)
	-docker rm $(BOT_NAME)

shell:
	docker exec -it $(BOT_NAME) /bin/sh
