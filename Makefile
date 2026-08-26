SHELL := /bin/bash
COMPOSE ?= docker compose
PYTHON ?= python3
RUN_USER ?= $(shell id -u):$(shell id -g)

.PHONY: help env up down logs test lint demo ingest-b3 ingest-corporate-actions ingest-cvm ingest-cvm-registry ingest-macro refresh-research-data generate-issuer-bridge build-fundamentals tune train backtest diagnose-model-degradation build-regime-candidates horizon-challengers risk-benchmarks qkp-ablation bootstrap-performance decompose-performance native clean

help:
	@printf '%s\n' \
	  'make env                              - create .env + local data directories' \
	  'make up                               - start PostgreSQL, Valkey, API and web UI' \
	  'make demo                             - seed deterministic demo data, then start stack' \
	  'make ingest-b3 YEAR=2025              - download/parse official B3 COTAHIST' \
	  'make ingest-corporate-actions          - B3 actions + adjusted prices; unavailable roots quarantined' \
	  'SKIP_ACTION_SYNC=1 make ingest-corporate-actions - reuse existing actions without HTTP attempts' \
	  'make ingest-cvm YEAR=2025 DOC=ITR     - download CVM ITR/DFP statements' \
	  'make ingest-cvm-registry START_YEAR=2010 END_YEAR=2026 - historical ticker/CVM bridge' \
	  'make ingest-macro START=2011-01-01    - download BCB SGS macro series' \
	  'make refresh-research-data START_YEAR=2011 END_YEAR=2026 - rebuild leakage-safe official-data panel' \
	  'make diagnose-model-degradation        - development-only drift/IC/Top-K/window/target diagnostics' \
	  'make build-regime-candidates           - build PIT regime candidates; does not enable them in production' \
	  'make horizon-challengers               - build development-only 21/42/63-bar target panels' \
	  'make risk-benchmarks                    - sample vs Ledoit-Wolf minvar/inverse-vol/HRP on <=2025' \
	  'make tune TRIALS=25                    - purged time-aware tuning; 2026 never enters selection' \
	  'make train                            - train final artifact with labels known by 2025-12-31' \
	  'make backtest                         - development-only purged walk-forward + 1/N + gates' \
	  'make backtest USE_TRAINED_MODEL=1     - diagnostic 2026 only; cannot change gates' \
	  'make qkp-ablation                      - isolate ML vs QKP vs HRP on development data' \
	  'make bootstrap-performance             - moving-block bootstrap confidence intervals' \
	  'make decompose-performance             - annual/monthly/ticker/sector concentration report' \
	  'make test                             - Python tests + C++ exact-solver tests' \
	  'make native                           - build exact native QKP reference solver' \
	  'make down                             - stop local stack'

env:
	@test -f .env || cp .env.example .env
	@mkdir -p data/{raw,bronze,silver,gold,cache,demo} data/gold/experiments models

demo: env
	$(COMPOSE) up -d postgres valkey
	$(COMPOSE) build api web
	$(COMPOSE) run --rm api python /workspace/scripts/seed_demo.py
	$(COMPOSE) up -d api web
	@echo 'Open http://localhost:3000'

up: env
	$(COMPOSE) up -d --build
	@echo 'Web: http://localhost:3000  API docs: http://localhost:8000/docs'

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=200

ingest-b3: env
	@test -n "$(YEAR)" || (echo 'Usage: make ingest-b3 YEAR=2025' && exit 2)
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/ingest_b3.py --year $(YEAR) $(if $(NO_DOWNLOAD),--no-download,) $(if $(FORCE),--force,)

ingest-corporate-actions: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/ingest_corporate_actions.py $(if $(FORCE),--force,) $(if $(SKIP_ACTION_SYNC),--skip-sync,)

ingest-cvm-registry: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/ingest_cvm_registry.py --start-year $(or $(START_YEAR),2010) --end-year $(or $(END_YEAR),2026) $(if $(FORCE),--force,)

refresh-research-data: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/refresh_research_data.py --start-year $(or $(START_YEAR),2011) --end-year $(or $(END_YEAR),2026) $(if $(FORCE),--force,) $(if $(SKIP_ACTION_SYNC),--skip-action-sync,)

ingest-cvm: env
	@test -n "$(YEAR)" || (echo 'Usage: make ingest-cvm YEAR=2025 DOC=ITR' && exit 2)
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/ingest_cvm.py --year $(YEAR) --document $(or $(DOC),ITR)

ingest-macro: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/ingest_macro.py --start $(or $(START),2011-01-01)

generate-issuer-bridge: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/generate_issuer_bridge.py --output /workspace/$(or $(OUTPUT),data/demo/issuer_bridge.csv) --review-output /workspace/$(or $(REVIEW),data/demo/issuer_bridge_candidates.csv)

build-fundamentals: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/build_fundamentals.py $(if $(BRIDGE),--bridge /workspace/$(BRIDGE),)

diagnose-model-degradation: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/diagnose_model_degradation.py

build-regime-candidates: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/build_regime_candidates.py

horizon-challengers: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/build_horizon_challengers.py

risk-benchmarks: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/run_risk_benchmarks.py

tune: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/tune_model.py --trials $(or $(TRIALS),25)

train: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/train_model.py

backtest: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/run_backtest.py $(if $(filter 1 true yes,$(USE_TRAINED_MODEL)),--use-trained-model,)

qkp-ablation: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/qkp_ablation.py

bootstrap-performance: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/bootstrap_performance.py

decompose-performance: env
	$(COMPOSE) run --rm --build --user $(RUN_USER) api python /workspace/scripts/decompose_performance.py $(or $(STRATEGY),development_research_v2)

test:
	$(COMPOSE) run --rm --build api pytest -q /workspace/packages/portfolio_core/tests /workspace/services/api/tests
	@if command -v cmake >/dev/null 2>&1; then $(MAKE) native && ./native/qkp/build/qkp_tests; fi

lint:
	$(COMPOSE) run --rm --build api ruff check /workspace/packages /workspace/services /workspace/scripts
	npm --prefix apps/web run typecheck

native:
	cmake -S native/qkp -B native/qkp/build -DCMAKE_BUILD_TYPE=Release
	cmake --build native/qkp/build -j

clean:
	rm -rf .venv node_modules apps/web/node_modules apps/web/.next native/qkp/build
