IMAGE ?= qwen38-flash-next-3090:local
PYTHON ?= python3

ifneq (,$(wildcard .env))
include .env
export
endif

.PHONY: validate test manifest build-image preflight serve bench

validate:
	$(PYTHON) scripts/validate_repo.py

test:
	$(PYTHON) -B -m unittest discover -s tests -p 'test_*.py'

manifest:
	$(PYTHON) scripts/update_manifest.py

build-image: validate
	docker build -f docker/Dockerfile -t $(IMAGE) .

preflight:
	./scripts/preflight.sh

serve: preflight
	IMAGE=$(IMAGE) ./scripts/docker_serve.sh

bench:
	$(PYTHON) scripts/bench.py --base-url http://127.0.0.1:$(or $(PORT),8000) --out results/bench-$(shell date +%Y%m%d-%H%M%S).json
