# IC Test Lab Segmentation

A research-friendly framework for developing and evaluating medical image segmentation models. The project provides modular components for nnU-Net style training, flexible inference pipelines, curated data loaders, and lightweight UIs for experimentation and demo deployments.

## Project Structure

```
├── src/
│   └── segmentation_app/
│       ├── data/
│       ├── inference/
│       ├── training/
│       └── ui/
├── tests/
├── pyproject.toml
└── .github/workflows/
```

* `training/`: Experiment orchestration, nnU-Net fine-tuning, and experiment tracking utilities.
* `inference/`: Batch and interactive inference endpoints, including FastAPI and Streamlit adapters.
* `data/`: Dataset registration, preprocessing transforms, and augmentation pipelines.
* `ui/`: Lightweight visual interfaces for demos and annotation workflows.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/ic-test-lab-segmentation.git
   cd ic-test-lab-segmentation
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -U pip
   pip install .[dev]
   ```

4. **Run quality checks**
   ```bash
   ruff check src tests
   black --check src tests
   mypy src
   pytest
   ```

5. **Launch the FastAPI server**
   ```bash
   uvicorn segmentation_app.inference.api:app --reload
   ```

6. **Launch the Streamlit UI**
   ```bash
   streamlit run src/segmentation_app/ui/app.py
   ```

## Development Guidelines

* Place application code under `src/segmentation_app/` and ensure new modules are importable via `pyproject.toml` packaging metadata.
* Add unit tests under `tests/` mirroring the package layout.
* Run the provided lint, format, and type-check commands before submitting a pull request.
* Follow conventional commit messages where possible (e.g., `feat:`, `fix:`, `docs:`).

## Contribution Workflow

1. Fork the repository and create a feature branch.
2. Commit changes with clear messages and include relevant documentation updates.
3. Ensure CI passes locally.
4. Open a pull request describing the motivation, approach, and testing.
5. Request reviews from the maintainers.

## License

This project is released under the MIT License. See `LICENSE` for details.
