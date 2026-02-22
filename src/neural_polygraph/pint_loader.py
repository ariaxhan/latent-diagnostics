"""
PINT Benchmark Loader

Loads the Lakera PINT benchmark dataset for prompt injection detection evaluation.
Format: YAML with text, category, label fields.

Usage:
    loader = PINTBenchmark()
    loader.load_from_yaml("path/to/pint.yaml")
    # or
    loader.load_from_github()  # fetches from lakeraai/pint-benchmark
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Tuple
import json


@dataclass
class PINTSample:
    """A single PINT benchmark sample."""
    text: str
    category: str
    label: bool  # True = injection/jailbreak, False = benign
    idx: int = 0

    @property
    def is_injection(self) -> bool:
        return self.label

    @property
    def is_benign(self) -> bool:
        return not self.label


class PINTBenchmark:
    """
    Loader for PINT (Prompt Injection Test) benchmark.

    Dataset composition (4,314 total):
    - prompt_injection: 5.2%
    - jailbreak: 0.9%
    - hard_negatives: 20.9%
    - chat: 36.5%
    - documents: 36.5%

    Languages: English (3,016) + multilingual (1,298)
    """

    def __init__(self):
        self.samples: List[PINTSample] = []
        self._by_category: Dict[str, List[PINTSample]] = {}

    def load_from_yaml(self, path: Path) -> "PINTBenchmark":
        """Load PINT data from YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PINT dataset not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        self._load_samples(data)
        return self

    def load_from_json(self, path: Path) -> "PINTBenchmark":
        """Load PINT data from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PINT dataset not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        self._load_samples(data)
        return self

    def load_from_github(self, cache_dir: Optional[Path] = None) -> "PINTBenchmark":
        """
        Fetch PINT benchmark from GitHub.

        Uses the lakeraai/pint-benchmark repository.
        """
        import urllib.request
        import yaml

        # The benchmark YAML files are in benchmark/data/
        # We need to fetch the full benchmark, not just example-dataset
        base_url = "https://raw.githubusercontent.com/lakeraai/pint-benchmark/main/benchmark/data/"

        # Try to fetch the main benchmark file
        # Note: The actual benchmark might be private or in a different location
        # For now, we'll use a placeholder that expects the user to download it

        if cache_dir:
            cache_path = Path(cache_dir) / "pint_benchmark.yaml"
            if cache_path.exists():
                return self.load_from_yaml(cache_path)

        raise NotImplementedError(
            "Direct GitHub fetch not implemented. "
            "Please download the PINT benchmark manually from "
            "https://github.com/lakeraai/pint-benchmark and use load_from_yaml()"
        )

    def _load_samples(self, data: List[Dict]) -> None:
        """Parse raw data into PINTSample objects."""
        self.samples = []
        self._by_category = {}

        for idx, item in enumerate(data):
            sample = PINTSample(
                text=item["text"],
                category=item.get("category", "unknown"),
                label=item["label"],
                idx=idx,
            )
            self.samples.append(sample)

            # Index by category
            if sample.category not in self._by_category:
                self._by_category[sample.category] = []
            self._by_category[sample.category].append(sample)

        print(f"✓ Loaded {len(self.samples)} PINT samples")
        self._print_stats()

    def _print_stats(self) -> None:
        """Print dataset statistics."""
        n_injection = sum(1 for s in self.samples if s.is_injection)
        n_benign = len(self.samples) - n_injection

        print(f"  Injection: {n_injection} ({100*n_injection/len(self.samples):.1f}%)")
        print(f"  Benign: {n_benign} ({100*n_benign/len(self.samples):.1f}%)")
        print(f"  Categories: {sorted(self._by_category.keys())}")

    def get_injections(self) -> List[PINTSample]:
        """Get all injection/jailbreak samples."""
        return [s for s in self.samples if s.is_injection]

    def get_benign(self) -> List[PINTSample]:
        """Get all benign samples."""
        return [s for s in self.samples if s.is_benign]

    def get_by_category(self, category: str) -> List[PINTSample]:
        """Get samples by category."""
        return self._by_category.get(category, [])

    def iter_samples(self) -> Iterator[PINTSample]:
        """Iterate over all samples."""
        yield from self.samples

    def iter_with_label(self) -> Iterator[Tuple[str, bool]]:
        """Iterate yielding (text, is_injection) pairs."""
        for sample in self.samples:
            yield sample.text, sample.is_injection

    @property
    def categories(self) -> List[str]:
        """List all categories in the dataset."""
        return sorted(self._by_category.keys())

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        n_inj = sum(1 for s in self.samples if s.is_injection)
        return f"PINTBenchmark(samples={len(self.samples)}, injections={n_inj})"
