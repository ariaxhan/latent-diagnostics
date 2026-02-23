"""
Dataset loaders for mechanistic interpretability research.

Supports diverse text types for SAE attribution analysis:
- Domain signatures (code, scientific, legal, poetry)
- Truthfulness probes (TruthfulQA, HaluEval, FEVER)
- Linguistic structure (CoLA, PAWS, formality)
"""

from dataclasses import dataclass
from typing import Iterator, Optional, Literal
from pathlib import Path
import random

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


@dataclass
class Sample:
    """A text sample for attribution analysis."""
    text: str
    source: str  # dataset name
    domain: str  # category within dataset
    label: Optional[str] = None  # ground truth if available
    metadata: Optional[dict] = None


class DatasetLoader:
    """Unified loader for mechanistic interpretability datasets."""

    AVAILABLE_DATASETS = {
        # Truthfulness
        "truthfulqa": "Truthful vs. false answers",
        "halueval_qa": "Hallucinated vs. faithful QA",
        "halueval_dialogue": "Hallucinated vs. faithful dialogue",
        "halueval_summarization": "Hallucinated vs. faithful summaries",
        "fever": "Fact verification (supported/refuted/nei)",

        # Domain diversity (The Pile subsets)
        "pile_arxiv": "Scientific papers from arXiv",
        "pile_github": "Code from GitHub",
        "pile_pubmed": "Biomedical abstracts",
        "pile_wikipedia": "Wikipedia articles",
        "pile_stackexchange": "StackExchange Q&A",
        "pile_books3": "Book text",
        "pile_freelaw": "Legal documents",

        # Linguistic structure
        "cola": "Grammatical acceptability",
        "paws": "Paraphrase adversaries",
        "snli": "Natural language inference",
        "winogrande": "Commonsense coreference",
        "hellaswag": "Situational commonsense",

        # Domain-specific
        "codesearchnet": "Code-docstring pairs",
        "pubmedqa": "Biomedical QA",
        "legalbench": "Legal reasoning tasks",
        "poem_sentiment": "Poetry with sentiment",

        # Cognitive regimes (orthogonal task families)
        "gsm8k": "Math word problems (symbolic reasoning)",
        "humaneval": "Code synthesis (program generation)",
        "cnn_dailymail": "Summarization (long-context abstraction)",
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        if not HF_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        self.cache_dir = cache_dir

    def load(
        self,
        dataset_name: str,
        n_samples: int = 500,
        split: str = "validation",
        seed: int = 42,
    ) -> list[Sample]:
        """Load samples from a dataset."""
        random.seed(seed)

        loader_map = {
            "truthfulqa": self._load_truthfulqa,
            "halueval_qa": lambda n, s: self._load_halueval(n, s, "qa"),
            "halueval_dialogue": lambda n, s: self._load_halueval(n, s, "dialogue"),
            "halueval_summarization": lambda n, s: self._load_halueval(n, s, "summarization"),
            "fever": self._load_fever,
            "cola": self._load_cola,
            "paws": self._load_paws,
            "snli": self._load_snli,
            "winogrande": self._load_winogrande,
            "hellaswag": self._load_hellaswag,
            "codesearchnet": self._load_codesearchnet,
            "pubmedqa": self._load_pubmedqa,
            "poem_sentiment": self._load_poem_sentiment,
            # Cognitive regimes
            "gsm8k": self._load_gsm8k,
            "humaneval": self._load_humaneval,
            "cnn_dailymail": self._load_cnn_dailymail,
        }

        # Handle Pile subsets
        if dataset_name.startswith("pile_"):
            subset = dataset_name.replace("pile_", "")
            return self._load_pile_subset(n_samples, split, subset)

        if dataset_name not in loader_map:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.AVAILABLE_DATASETS.keys())}")

        return loader_map[dataset_name](n_samples, split)

    def _load_truthfulqa(self, n_samples: int, split: str) -> list[Sample]:
        """Load TruthfulQA - truthful vs. false answers."""
        ds = load_dataset("truthful_qa", "generation", split="validation")
        samples = []

        for item in ds:
            if len(samples) >= n_samples * 2:
                break

            question = item["question"]
            category = item["category"]

            # Best correct answer
            correct = item["best_answer"]
            samples.append(Sample(
                text=f"{question} {correct}",
                source="truthfulqa",
                domain=category,
                label="truthful",
                metadata={"question": question, "answer": correct},
            ))

            # Incorrect answer (common misconception)
            if item["incorrect_answers"]:
                incorrect = item["incorrect_answers"][0]
                samples.append(Sample(
                    text=f"{question} {incorrect}",
                    source="truthfulqa",
                    domain=category,
                    label="false",
                    metadata={"question": question, "answer": incorrect},
                ))

        random.shuffle(samples)
        return samples[:n_samples]

    def _load_halueval(self, n_samples: int, split: str, task: str) -> list[Sample]:
        """Load HaluEval - hallucinated vs. faithful responses."""
        ds = load_dataset("pminervini/HaluEval", task, split="data")
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            # HaluEval format varies by task
            if task == "qa":
                text = f"Q: {item['question']}\nA: {item['answer']}"
                is_hallucinated = item.get("hallucination", "no") == "yes"
            elif task == "dialogue":
                text = f"Context: {item.get('dialogue_history', '')}\nResponse: {item['response']}"
                is_hallucinated = item.get("hallucination", "no") == "yes"
            else:  # summarization
                text = f"Document: {item.get('document', '')[:500]}\nSummary: {item['summary']}"
                is_hallucinated = item.get("hallucination", "no") == "yes"

            samples.append(Sample(
                text=text[:1000],  # Truncate for attribution
                source=f"halueval_{task}",
                domain=task,
                label="hallucinated" if is_hallucinated else "faithful",
            ))

        return samples

    def _load_fever(self, n_samples: int, split: str) -> list[Sample]:
        """Load FEVER - fact verification."""
        ds = load_dataset("fever", "v1.0", split=split)
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            claim = item["claim"]
            label = item["label"]  # SUPPORTS, REFUTES, NOT ENOUGH INFO

            samples.append(Sample(
                text=claim,
                source="fever",
                domain="fact_verification",
                label=label,
            ))

        return samples

    def _load_cola(self, n_samples: int, split: str) -> list[Sample]:
        """Load CoLA - grammatical acceptability."""
        ds = load_dataset("nyu-mll/glue", "cola", split=split)
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            samples.append(Sample(
                text=item["sentence"],
                source="cola",
                domain="grammar",
                label="acceptable" if item["label"] == 1 else "unacceptable",
            ))

        return samples

    def _load_paws(self, n_samples: int, split: str) -> list[Sample]:
        """Load PAWS - paraphrase adversaries."""
        ds = load_dataset("paws", "labeled_final", split=split)
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            # Combine both sentences as input
            text = f"Sentence 1: {item['sentence1']}\nSentence 2: {item['sentence2']}"
            samples.append(Sample(
                text=text,
                source="paws",
                domain="paraphrase",
                label="paraphrase" if item["label"] == 1 else "not_paraphrase",
            ))

        return samples

    def _load_snli(self, n_samples: int, split: str) -> list[Sample]:
        """Load SNLI - natural language inference."""
        ds = load_dataset("snli", split=split)
        label_map = {0: "entailment", 1: "neutral", 2: "contradiction", -1: "unknown"}
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break
            if item["label"] == -1:
                continue

            text = f"Premise: {item['premise']}\nHypothesis: {item['hypothesis']}"
            samples.append(Sample(
                text=text,
                source="snli",
                domain="nli",
                label=label_map[item["label"]],
            ))

        return samples

    def _load_winogrande(self, n_samples: int, split: str) -> list[Sample]:
        """Load WinoGrande - commonsense coreference."""
        ds = load_dataset("winogrande", "winogrande_xl", split=split)
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            # Fill in the blank with correct option
            sentence = item["sentence"]
            correct = item["option1"] if item["answer"] == "1" else item["option2"]
            filled = sentence.replace("_", correct)

            samples.append(Sample(
                text=filled,
                source="winogrande",
                domain="commonsense",
                label="correct_completion",
                metadata={"original": sentence, "options": [item["option1"], item["option2"]]},
            ))

        return samples

    def _load_hellaswag(self, n_samples: int, split: str) -> list[Sample]:
        """Load HellaSwag - situational commonsense."""
        ds = load_dataset("hellaswag", split=split)
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            context = item["ctx"]
            correct_idx = int(item["label"])
            correct_ending = item["endings"][correct_idx]

            samples.append(Sample(
                text=f"{context} {correct_ending}",
                source="hellaswag",
                domain=item.get("activity_label", "situational"),
                label="correct_completion",
            ))

        return samples

    def _load_codesearchnet(self, n_samples: int, split: str) -> list[Sample]:
        """Load CodeSearchNet - code with docstrings."""
        samples = []
        languages = ["python", "java", "javascript", "go", "ruby", "php"]

        per_lang = max(1, n_samples // len(languages))

        for lang in languages:
            try:
                ds = load_dataset("code_search_net", lang, split=split)
                count = 0
                for item in ds:
                    if count >= per_lang:
                        break

                    code = item.get("func_code_string", item.get("code", ""))
                    docstring = item.get("func_documentation_string", item.get("docstring", ""))

                    if code and docstring:
                        samples.append(Sample(
                            text=f"# {docstring[:200]}\n{code[:500]}",
                            source="codesearchnet",
                            domain=f"code_{lang}",
                            label=lang,
                        ))
                        count += 1
            except Exception:
                continue  # Some languages may not be available

        return samples[:n_samples]

    def _load_pubmedqa(self, n_samples: int, split: str) -> list[Sample]:
        """Load PubMedQA - biomedical QA."""
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train")  # Only train available
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            question = item["question"]
            context = " ".join(item["context"]["contexts"])[:500]
            answer = item["final_decision"]  # yes/no/maybe

            samples.append(Sample(
                text=f"Context: {context}\nQuestion: {question}",
                source="pubmedqa",
                domain="biomedical",
                label=answer,
            ))

        return samples

    def _load_poem_sentiment(self, n_samples: int, split: str) -> list[Sample]:
        """Load poem sentiment - poetry with sentiment labels."""
        ds = load_dataset("poem_sentiment", split=split)
        label_map = {0: "negative", 1: "positive", 2: "neutral", 3: "mixed"}
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            samples.append(Sample(
                text=item["verse_text"],
                source="poem_sentiment",
                domain="poetry",
                label=label_map.get(item["label"], "unknown"),
            ))

        return samples

    def _load_pile_subset(self, n_samples: int, split: str, subset: str) -> list[Sample]:
        """Load a subset from The Pile."""
        # Note: Full Pile is very large. This requires streaming or local subset.
        # For now, provide instructions for manual download
        raise NotImplementedError(
            f"Pile subset '{subset}' requires manual download.\n"
            "Use: https://huggingface.co/datasets/EleutherAI/pile\n"
            "Or sample via: datasets.load_dataset('EleutherAI/pile', streaming=True)"
        )

    def _load_gsm8k(self, n_samples: int, split: str) -> list[Sample]:
        """Load GSM8K - grade-school math word problems.

        Truncates to prevent OOM during attribution while preserving
        the math reasoning structure (problem + first solution steps).
        """
        ds = load_dataset("gsm8k", "main", split="test" if split == "validation" else split)
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            question = item["question"][:250]  # Truncate problem
            answer = item["answer"][:200]  # Truncate solution

            samples.append(Sample(
                text=f"Problem: {question}\nSolution: {answer}",
                source="gsm8k",
                domain="math_reasoning",
                label="math_problem",
                metadata={"question": item["question"], "answer": item["answer"]},
            ))

        return samples

    def _load_humaneval(self, n_samples: int, split: str) -> list[Sample]:
        """Load HumanEval - code synthesis problems.

        Truncates to prevent OOM during attribution while preserving
        the code synthesis structure (signature + docstring + partial solution).
        """
        ds = load_dataset("openai_humaneval", split="test")
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            prompt = item["prompt"][:300]  # Truncate prompt
            canonical = item.get("canonical_solution", "")[:200]  # Truncate solution
            task_id = item.get("task_id", "")

            samples.append(Sample(
                text=f"{prompt}\n{canonical}",
                source="humaneval",
                domain="code_synthesis",
                label=task_id,
                metadata={"prompt": item["prompt"], "solution": item.get("canonical_solution", ""), "task_id": task_id},
            ))

        return samples

    def _load_cnn_dailymail(self, n_samples: int, split: str) -> list[Sample]:
        """Load CNN/DailyMail - summarization (long-context abstraction).

        Note: Attribution is memory-intensive. We truncate aggressively
        but still capture the abstraction regime (input → compressed output).
        """
        ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
        samples = []

        for item in ds:
            if len(samples) >= n_samples:
                break

            article = item["article"]
            highlights = item["highlights"]

            # Aggressive truncation for attribution (OOM prevention)
            # Still captures the compression regime: article excerpt → summary
            article_excerpt = article[:300]
            summary_excerpt = highlights[:200]

            samples.append(Sample(
                text=f"Article: {article_excerpt}\n\nSummary: {summary_excerpt}",
                source="cnn_dailymail",
                domain="summarization",
                label="abstractive_summary",
                metadata={"article_length": len(article), "summary_length": len(highlights)},
            ))

        return samples

    def load_multi(
        self,
        dataset_names: list[str],
        n_per_dataset: int = 200,
        split: str = "validation",
        seed: int = 42,
    ) -> list[Sample]:
        """Load samples from multiple datasets."""
        all_samples = []
        for name in dataset_names:
            try:
                samples = self.load(name, n_per_dataset, split, seed)
                all_samples.extend(samples)
                print(f"  Loaded {len(samples)} from {name}")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
        return all_samples


def create_domain_comparison_dataset(n_per_domain: int = 200) -> list[Sample]:
    """
    Create a balanced dataset for domain signature analysis.

    Returns samples from:
    - Code (multiple languages)
    - Scientific (PubMedQA)
    - Legal reasoning
    - Poetry
    - General QA (WinoGrande)
    - Fact verification (FEVER)
    """
    loader = DatasetLoader()

    datasets = [
        "codesearchnet",
        "pubmedqa",
        "poem_sentiment",
        "winogrande",
        "fever",
        "cola",
    ]

    return loader.load_multi(datasets, n_per_domain)


def create_truthfulness_dataset(n_per_source: int = 300) -> list[Sample]:
    """
    Create a dataset for truthfulness/hallucination analysis.

    Returns samples from:
    - TruthfulQA (truthful vs. false)
    - HaluEval (hallucinated vs. faithful)
    - FEVER (supported vs. refuted)
    """
    loader = DatasetLoader()

    datasets = [
        "truthfulqa",
        "halueval_qa",
        "fever",
    ]

    return loader.load_multi(datasets, n_per_source)


def create_linguistic_structure_dataset(n_per_task: int = 250) -> list[Sample]:
    """
    Create a dataset for linguistic structure analysis.

    Returns samples probing:
    - Grammar (CoLA)
    - Paraphrase (PAWS)
    - Inference (SNLI)
    - Commonsense (WinoGrande, HellaSwag)
    """
    loader = DatasetLoader()

    datasets = [
        "cola",
        "paws",
        "snli",
        "winogrande",
        "hellaswag",
    ]

    return loader.load_multi(datasets, n_per_task)
