.PHONY: help check deploy-bot deploy-modal deploy-all

APPS := apps/delulu_discord apps/delulu_sandbox_modal

help:
	@echo "Top-level targets:"
	@echo "  make check         run ruff on all apps/"
	@echo "  make deploy-bot    deploy the Discord bot to the VPS"
	@echo "  make deploy-modal  deploy the Modal sandbox app"
	@echo "  make deploy-all    modal-deploy then bot-deploy"
	@echo ""
	@echo "Sub-project targets available via:"
	@echo "  make -C apps/delulu_discord <target>"
	@echo "  make -C apps/delulu_sandbox_modal <target>"

check:
	$(MAKE) -C apps/delulu_discord check
	$(MAKE) -C apps/delulu_sandbox_modal check

deploy-bot:
	$(MAKE) -C apps/delulu_discord deploy

deploy-modal:
	$(MAKE) -C apps/delulu_sandbox_modal modal-deploy

deploy-all: deploy-modal deploy-bot
