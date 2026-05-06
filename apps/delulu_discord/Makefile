.PHONY: help sync sync-dev lint fmt check test image restart deploy logs stop shell

# Configurable knobs — override with e.g. `make deploy IMAGE=myimage`.
BOT_NAME   ?= disco
IMAGE      ?= delulu-discord
ENV_FILE   ?= /root/disco.env
MODAL_TOML ?= /root/.modal.toml
# Named Docker volume for the bot's persistent state (currently the
# SQLite session DB at /data/sessions.db inside the container).
# Survives `docker rm` so /setrepo bindings + thread sessions stick
# across deploys. Created automatically by Docker on first run.
DATA_VOLUME ?= disco-data

help:
	@echo "Targets (VPS deploy):"
	@echo "  make sync          uv sync runtime deps"
	@echo "  make sync-dev      uv sync runtime + dev deps"
	@echo "  make check         ruff check + format --check"
	@echo "  make test          pytest"
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

test: sync-dev
	uv run pytest

image:
	docker build -t $(IMAGE) .

restart:
	-docker stop $(BOT_NAME)
	-docker rm $(BOT_NAME)
	# --log-driver=journald: ship container stdout/stderr to systemd-journald
	# instead of Docker's default json-file driver. The default writes to a
	# per-container file under /var/lib/docker/containers/<id>/ that is
	# deleted with `docker rm` — every `make deploy` would wipe history,
	# making post-mortems on cancellations or crashes impossible across a
	# rebuild. journald survives container removal, restarts, and host
	# reboots (assuming /var/log/journal/ exists; create it with
	# `sudo mkdir -p /var/log/journal && sudo systemctl restart systemd-journald`
	# if it doesn't).
	docker run -d \
	  --name $(BOT_NAME) \
	  --restart=unless-stopped \
	  --log-driver=journald \
	  --env-file $(ENV_FILE) \
	  -v $(MODAL_TOML):/root/.modal.toml:ro \
	  -v $(DATA_VOLUME):/data \
	  $(IMAGE)

deploy: image restart

# `journalctl -f` tails new lines and also makes historical queries
# possible (`journalctl CONTAINER_NAME=disco --since "1 hour ago"`).
# `docker logs` still works for the current container's lifetime under
# the journald driver, but loses everything on `docker rm`.
logs:
	journalctl -f CONTAINER_NAME=$(BOT_NAME)

stop:
	-docker stop $(BOT_NAME)
	-docker rm $(BOT_NAME)

shell:
	docker exec -it $(BOT_NAME) /bin/sh
