.PHONY: install install-gpu test demo schemas check-access download-public lint

install:
	pip install -e ".[dev]"

install-gpu:
	pip install -e ".[train,constrained,mech,study,dev]"

test:
	pytest

demo:
	python examples/toy_demo.py

schemas:
	python -c "import json; from clinblueprint.edits.types import Edit; from clinblueprint.data.records import CaseRecord; from clinblueprint.data.sct import SCTItem; from clinblueprint.schema.graph import ReasoningGraph; [json.dump({**M.model_json_schema(), '\$$schema': 'https://json-schema.org/draft/2020-12/schema'}, open(f'benchmark/clinedit_faith/schema/{n}.schema.json', 'w'), indent=2) for n, M in [('edit', Edit), ('case', CaseRecord), ('sct_item', SCTItem), ('reasoning_graph', ReasoningGraph)]]"

check-access:
	python scripts/check_access.py

download-public:
	bash scripts/download/download_public.sh

lint:
	ruff check src tests scripts examples
