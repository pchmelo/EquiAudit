# EquiAudit — Dataset Quality & Fairness Evaluation System

An AI-agent pipeline for evaluating datasets on data quality and fairness concerns, with an interactive Streamlit GUI, a headless CLI, and a Python API.

<div align="center">
<img src="https://raw.githubusercontent.com/pchmelo/EquiAudit/master/docs/img/simple_diagram.png" alt="System overview" width="320"/>

*A dataset is fed into EquiAudit, which coordinates AI agents under optional user supervision to produce an audit report and a bias-mitigated dataset.*
</div>

## How It Works

<div align="center">
<img src="https://raw.githubusercontent.com/pchmelo/EquiAudit/master/docs/img/data_preparation_fairness_analysis.png" alt="Agent-tool-data interaction" width="600"/>

*AI agents interact bidirectionally with the analyst, invoke tools to query and compute statistics on the dataset, and synthesise findings into an audit report.*
</div>

<div align="center">
<img src="https://raw.githubusercontent.com/pchmelo/EquiAudit/master/docs/img/mitigation.png" alt="Bias mitigation workflow" width="600"/>

*Each selected mitigation technique produces a transformed dataset variant; tools recompute fairness statistics on every variant; the AI agent compares them against the original statistics and produces a comparative mitigation report.*
</div>

## Demo

A video demonstration of the full GUI workflow is available on YouTube:
[https://youtu.be/_USTmBhzDkI](https://youtu.be/_USTmBhzDkI)

## Documentation

| Guide | Description |
|-------|-------------|
| [USAGE.md](docs/USAGE.md) | Installation, configuration reference, all execution modes (GUI / CLI / Python API), and worked examples |
| [EXTENDING.md](docs/EXTENDING.md) | How to add new Tools, Agent types, LLM backends, and Pipeline stages |

## Installation

**From PyPI (recommended):**
```bash
pip install equiaudit            # core pipeline
pip install "equiaudit[gui]"     # optional: Streamlit GUI
```

**From source:**
```bash
git clone https://github.com/pchmelo/EquiAudit.git
cd EquiAudit
pip install -r requirements.txt          # core pipeline
pip install -r requirements-gui.txt      # optional: Streamlit GUI
```

## Configuration

Copy the example config and edit it for your setup:

```bash
cp examples/config.example.yml examples/config.yml
```

`config.yml` is git-ignored so your API keys and local paths are never committed.

Set your API keys as environment variables or in a `.env` file at the project root:

```bash
GOOGLE_API_KEY=your-google-gemini-api-key-here
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

At least one cloud API key is required unless running a local model via Ollama (`ollama serve`).

`config.example.yml` documents every available option with inline comments. See also [USAGE.md — Configuration File](docs/USAGE.md#configuration-file) for the full reference.

## Running the Application

The recommended entry point is `examples/example_usage.py`. After copying `config.example.yml` to `config.yml`, set `mode: gui` or `mode: quick` in `examples/config.yml`, then run:

```bash
python examples/example_usage.py
```

The script dispatches to the GUI or the headless evaluator based on the config:

```python
# examples/example_usage.py
import os, yaml
from dotenv import load_dotenv
load_dotenv()

CONFIG_PATH  = os.path.join(os.path.dirname(__file__), "config.yml")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "adult-all.csv")

with open(CONFIG_PATH, encoding="utf-8") as f:
    _mode = (yaml.safe_load(f) or {}).get("mode", "quick")

if _mode == "gui":
    from equiaudit.gui import launch
    launch(config_path=CONFIG_PATH, dataset_path=DATASET_PATH)
else:
    from equiaudit.cli import FairnessEvaluator
    evaluator = FairnessEvaluator(config_path=CONFIG_PATH)
    result = evaluator.evaluate(
        data=DATASET_PATH,
        # target="Income",                                          # overrides target_column in config
        # sensitive_columns=["Sex", "Race", "Age"],                # overrides sensitive_attribute_analysis
        # sensitive_pairs=[["Sex", "Race"], ["Age", "Education"]], # overrides pair_evaluation
        # mitigation_techniques=["reweighting", "smote"],          # overrides mitigation_techniques
    )
    if result.success:
        print(f"Report: {result.report_dir}")
    else:
        print(f"Error:  {result.error}")
```

See [USAGE.md — Execution Modes](docs/USAGE.md#execution-modes) for the full CLI flag reference and Python API usage.

## Datasets
Example datasets are provided in `src/equiaudit/data/` for testing. The default example is `adult-all.csv` (UCI Adult Census Income, target column `Income`).

## Reports
Generated reports are saved under `reports/<dataset>_<timestamp>/` and include a Markdown narrative, a PDF, per-group fairness metrics, and raw JSON stage data. A sample report is available in `reports/`.

## GUI Screenshots

![Framework configuration](https://raw.githubusercontent.com/pchmelo/EquiAudit/master/docs/img/new_1.png)
*Framework configuration (dataset selection, model backend, target column, and pipeline options).*

![Sensitive attribute identification](https://raw.githubusercontent.com/pchmelo/EquiAudit/master/docs/img/new_2.png)
*Sensitive attribute identification (agent-generated candidate list pending analyst confirmation).*

![Proxy model fairness results](https://raw.githubusercontent.com/pchmelo/EquiAudit/master/docs/img/new_3.png)
*Proxy model fairness results (per-group metric bar charts from the outcome disparity stage).*

![Previous results browser](https://raw.githubusercontent.com/pchmelo/EquiAudit/master/docs/img/new_4.png)
*Previous results browser (inspect and compare past audit runs).*
