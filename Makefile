.PHONY: help build up down logs update backfill stats test lint install-systemd

help:            ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

build:           ## build the image
	docker compose build

up:              ## start the web app (and nginx with the server override)
	docker compose up -d --remove-orphans

down:            ## stop everything
	docker compose down

logs:            ## follow logs
	docker compose logs -f --tail=100

update:          ## daily job: fetch recent days, embed, score
	docker compose run --rm job update

backfill:        ## first load: DAYS=90 by default
	docker compose run --rm job backfill --days $${DAYS:-90}

stats:           ## database statistics
	docker compose run --rm job stats

test:            ## run tests in a throwaway container
	docker run --rm -v $(PWD):/src -w /src python:3.12-slim sh -c \
	  "pip install -q -r requirements-dev.txt && pip install -q --no-deps -e . && pytest"

lint:            ## ruff
	docker run --rm -v $(PWD):/src -w /src python:3.12-slim sh -c "pip install -q ruff==0.17.4 && ruff check src tests"

install-systemd: ## install and enable the service + daily timer
	install -m 644 deploy/systemd/*.service deploy/systemd/*.timer /etc/systemd/system/
	systemctl daemon-reload
	systemctl enable --now cs-paper-radar.service cs-paper-radar-update.timer
