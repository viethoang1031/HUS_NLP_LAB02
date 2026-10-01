"""LAB 02, sections 14-16 and 20-21: n-gram language models and applications.

Preprocessing matches experiments.ipynb: lowercase Unicode words/numbers,
internal apostrophes, English Punkt sentence splitting, and no boundary tokens.
The regular model has no UNK token; Experiment 2 enables one for held-out data.
N-grams never cross sentence boundaries. A sentence score uses unigram for
the first token, bigram for the second, then the selected context length.
Without an EOS token this scores the token sequence, not its termination.
Context denominators count outgoing n-grams, excluding sentence-final contexts
that have no next token. This keeps observed conditional distributions normalized.

Log probability adds log(p) instead of multiplying many small probabilities.
It avoids product underflow; an unseen MLE event still correctly gives -inf.
sentence_probability() can underflow for long sequences; prefer the log score.

Example from the repository root:
    python WEEK2/ngram_lm.py --n 2 --sentence "the cat eats fish"
    python WEEK2/ngram_lm.py --interactive --n 3 --smoothing laplace

Section 16 can be run with ``--experiment2``. It makes a reproducible split at
document level, learns the vocabulary from training data only, replaces rare
training words and later OOV words with <UNK>, and reports token-weighted
perplexity for train/validation/test.

AI assistance: Codex generated the implementation for sections 14-16; verified
against hand-counted examples, normalization, zero events, log underflow,
perplexity calculations, and a reproducible train/validation/test split.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path


TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)
UNK_TOKEN = "<UNK>"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "c4-train.00000-of-01024-30K.json"


def tokenize(text: str) -> list[str]:
    """Return lowercase tokens without punctuation or stopword removal."""
    return TOKEN_RE.findall(text.lower())


def _sentence_tokenizer():
    # NLTK is used only for sentence splitting, never for language modeling.
    import nltk

    local_data = str(ROOT / "tmp" / "nltk_data")
    if local_data not in nltk.data.path:
        nltk.data.path.insert(0, local_data)
    try:
        return nltk.tokenize.PunktTokenizer("english")
    except LookupError as exc:
        raise LookupError(
            "Missing NLTK punkt_tab. Run the setup cell in experiments.ipynb "
            "or download punkt_tab into tmp/nltk_data."
        ) from exc


def iter_tokenized_sentences(documents: Iterable[str]) -> Iterator[list[str]]:
    """Split documents into nonempty tokenized sentences, as in Experiment 1."""
    splitter = _sentence_tokenizer()
    for text in documents:
        if not isinstance(text, str):
            raise TypeError("Each document must be a string")
        for paragraph in text.splitlines():
            for sentence in splitter.tokenize(paragraph):
                tokens = tokenize(sentence)
                if tokens:
                    yield tokens


def load_documents(path: str | Path) -> Iterator[str]:
    """Stream the text field of a JSON Lines corpus; fail on invalid records."""
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("text"), str):
                raise ValueError(f"Line {line_number}: expected a string text field")
            yield record["text"]


def _validate_order(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n not in (1, 2, 3):
        raise ValueError("n must be 1, 2, or 3")


def _checked_tokens(sentence: Sequence[str]) -> tuple[str, ...]:
    if isinstance(sentence, str):
        raise TypeError("Expected a sequence of tokens, not raw sentence text")
    tokens = tuple(sentence)
    if any(not isinstance(token, str) or not token for token in tokens):
        raise ValueError("Tokens must be nonempty strings")
    return tokens


def build_vocabulary(sentences: Iterable[Sequence[str]]) -> set[str]:
    """Build vocabulary from already tokenized sentences."""
    return {token for sentence in sentences for token in _checked_tokens(sentence)}


def count_ngrams(sentences: Iterable[Sequence[str]], n: int) -> Counter:
    """Count n-grams within each tokenized sentence (no boundary padding)."""
    _validate_order(n)
    counts = Counter()
    for sentence in sentences:
        tokens = _checked_tokens(sentence)
        counts.update(zip(*(tokens[offset:] for offset in range(n))))
    return counts


class NGramLanguageModel:
    """Unigram/bigram/trigram MLE, optionally with Laplace smoothing.

    Short prefixes use the highest available lower order. There is no implicit
    backoff for unseen full contexts. With ``unk_cutoff > 0``, rare training
    words and evaluation OOV words are mapped to <UNK>; otherwise OOVs score 0.
    """

    def __init__(self, n: int, smoothing: str = "mle", unk_cutoff: int = 0):
        _validate_order(n)
        if smoothing not in ("mle", "laplace"):
            raise ValueError("smoothing must be 'mle' or 'laplace'")
        if isinstance(unk_cutoff, bool) or not isinstance(unk_cutoff, int) or unk_cutoff < 0:
            raise ValueError("unk_cutoff must be a nonnegative integer")
        self.n = n
        self.smoothing = smoothing
        self.unk_cutoff = unk_cutoff
        self.vocabulary: set[str] = set()
        self.counts: dict[int, Counter] = {}
        self.context_counts: dict[int, Counter] = {}
        self.total_tokens = 0
        self.num_sentences = 0
        self._fitted = False

    def fit(self, corpus: Iterable[str] | str) -> NGramLanguageModel:
        """Train on raw documents. A single string is treated as one document."""
        documents = [corpus] if isinstance(corpus, str) else corpus
        return self.fit_tokenized(iter_tokenized_sentences(documents))

    def fit_tokenized(self, sentences: Iterable[Sequence[str]]) -> NGramLanguageModel:
        """Train on preprocessed sentences, resetting any previous counts."""
        if self.unk_cutoff:
            stored_sentences = [tuple(_checked_tokens(sentence)) for sentence in sentences]
            token_counts = Counter(token for sentence in stored_sentences for token in sentence)
            known_words = {word for word, count in token_counts.items() if count > self.unk_cutoff}
            sentences = (
                tuple(token if token in known_words else UNK_TOKEN for token in sentence)
                for sentence in stored_sentences
            )
        self._fitted = False
        self.vocabulary = set()
        self.counts = {order: Counter() for order in range(1, self.n + 1)}
        self.context_counts = {order: Counter() for order in range(2, self.n + 1)}
        self.total_tokens = self.num_sentences = 0
        for sentence in sentences:
            tokens = _checked_tokens(sentence)
            if not tokens:
                continue
            self.num_sentences += 1
            self.total_tokens += len(tokens)
            self.vocabulary.update(tokens)
            for order, counter in self.counts.items():
                counter.update(zip(*(tokens[offset:] for offset in range(order))))
        if not self.total_tokens:
            raise ValueError("Cannot train on an empty token corpus")
        if self.unk_cutoff:
            self.vocabulary.add(UNK_TOKEN)
        for order, contexts in self.context_counts.items():
            for ngram, count in self.counts[order].items():
                contexts[ngram[:-1]] += count
        self._fitted = True
        return self

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("Call fit() or fit_tokenized() before scoring")

    def _context(self, context: Sequence[str] | str | None) -> tuple[str, ...]:
        if context is None:
            tokens = ()
        elif isinstance(context, str):
            tokens = tuple(tokenize(context))
        else:
            tokens = _checked_tokens(context)
        if self.unk_cutoff and UNK_TOKEN in self.vocabulary:
            tokens = tuple(token if token in self.vocabulary else UNK_TOKEN for token in tokens)
        return tokens[-(self.n - 1):] if self.n > 1 else ()

    def probability(self, context: Sequence[str] | str | None, word: str) -> float:
        """Return P(word | last n-1 context tokens), or P(word) at empty context.

        A string word is lowercased and must tokenize to exactly one token.
        Token-sequence contexts/sentences are assumed already preprocessed.
        """
        self._require_fitted()
        if word == UNK_TOKEN:
            target = UNK_TOKEN
        else:
            word_tokens = tokenize(word)
            if len(word_tokens) != 1:
                raise ValueError("word must contain exactly one token")
            target = word_tokens[0]
        if target not in self.vocabulary and self.unk_cutoff and UNK_TOKEN in self.vocabulary:
            target = UNK_TOKEN
        if target not in self.vocabulary:
            return 0.0
        history = self._context(context)
        order = len(history) + 1
        count = self.counts[order][history + (target,)]
        denominator = (
            self.total_tokens if order == 1 else self.context_counts[order][history]
        )
        if self.smoothing == "laplace":
            return (count + 1) / (denominator + len(self.vocabulary))
        return count / denominator if denominator else 0.0

    def sentence_log_probability(self, sentence: str | Sequence[str]) -> float:
        """Natural log of the token-sequence score; empty sequence has log(1)=0."""
        self._require_fitted()
        tokens = tuple(tokenize(sentence)) if isinstance(sentence, str) else _checked_tokens(sentence)
        terms = []
        for i, word in enumerate(tokens):
            start = max(0, i - self.n + 1)
            probability = self.probability(tokens[start:i], word)
            if probability == 0:
                return -math.inf
            terms.append(math.log(probability))
        return math.fsum(terms)

    def sentence_probability(self, sentence: str | Sequence[str]) -> float:
        """Return exp(log score); may underflow to zero for long sequences."""
        return math.exp(self.sentence_log_probability(sentence))

    def continuation_log_probability(
        self, context: str | Sequence[str], continuation: str | Sequence[str]
    ) -> float:
        """Score only continuation tokens conditioned on the supplied context.

        The context's own probability is excluded. Each generated candidate
        token extends the history for the next token. No EOS is scored.
        """
        self._require_fitted()
        history = list(tokenize(context)) if isinstance(context, str) else list(_checked_tokens(context))
        words = tokenize(continuation) if isinstance(continuation, str) else _checked_tokens(continuation)
        terms = []
        for word in words:
            probability = self.probability(history, word)
            if probability == 0:
                return -math.inf
            terms.append(math.log(probability))
            history.append(word)
            history = history[-(self.n - 1):] if self.n > 1 else []
        return math.fsum(terms)

    def rank_continuations(
        self, context: str | Sequence[str], candidates: Iterable[str],
        normalize_length: bool = False,
    ) -> list[dict[str, str | int | float]]:
        """Rank candidates by conditional log probability (or mean log P).

        Raw probability tends to prefer shorter candidates. Mean log P is an
        optional length-normalized score, not a sequence probability.
        Equal scores preserve input order.
        """
        self._require_fitted()
        ranked = []
        for candidate in candidates:
            words = tokenize(candidate)
            if not words:
                raise ValueError("Each candidate must contain at least one token")
            log_probability = self.continuation_log_probability(context, words)
            ranked.append({
                "candidate": candidate, "tokens": len(words),
                "log_probability": log_probability,
                "probability": math.exp(log_probability),
                "score": log_probability / len(words) if normalize_length else log_probability,
            })
        return sorted(ranked, key=lambda row: row["score"], reverse=True)

    def perplexity(self, sentences: Iterable[Sequence[str]]) -> float:
        """Return token-weighted corpus perplexity for tokenized sentences.

        The score is ``exp(-sum(log probabilities) / number of tokens)``.
        Empty sentences are ignored. One zero-probability event makes the
        corpus perplexity infinite, as required by unsmoothed MLE (and by an
        OOV event when no <UNK> token is used).
        """
        self._require_fitted()
        total_log_probability = 0.0
        token_count = 0
        for sentence in sentences:
            tokens = _checked_tokens(sentence)
            if not tokens:
                continue
            log_probability = self.sentence_log_probability(tokens)
            if log_probability == -math.inf:
                return math.inf
            total_log_probability += log_probability
            token_count += len(tokens)
        if token_count == 0:
            raise ValueError("Cannot evaluate perplexity on an empty token corpus")
        try:
            return math.exp(-total_log_probability / token_count)
        except OverflowError:
            return math.inf

    def next_word_distribution(self, context: Sequence[str] | str | None) -> dict[str, float]:
        """Return vocabulary probabilities sorted by score, then by token.

        An unseen MLE context has an all-zero mapping (no defined conditional
        distribution). Laplace instead gives a uniform vocabulary distribution.
        """
        self._require_fitted()
        pairs = ((word, self.probability(context, word)) for word in self.vocabulary)
        return dict(sorted(pairs, key=lambda pair: (-pair[1], pair[0])))


def train_unigram(corpus: Iterable[str] | str, smoothing: str = "mle") -> NGramLanguageModel:
    return NGramLanguageModel(1, smoothing).fit(corpus)


def train_bigram(corpus: Iterable[str] | str, smoothing: str = "mle") -> NGramLanguageModel:
    return NGramLanguageModel(2, smoothing).fit(corpus)


def train_trigram(corpus: Iterable[str] | str, smoothing: str = "mle") -> NGramLanguageModel:
    return NGramLanguageModel(3, smoothing).fit(corpus)


def split_documents(
    documents: Iterable[str],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str]]:
    """Shuffle and split documents without leaking sentences across splits."""
    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("Ratios must satisfy 0 < train_ratio < 1 and 0 <= validation_ratio < 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1")
    partitions = list(documents)
    if len(partitions) < 3:
        raise ValueError("At least three documents are required for a three-way split")
    random.Random(seed).shuffle(partitions)
    train_end = min(len(partitions) - 2, max(1, int(len(partitions) * train_ratio)))
    validation_end = max(train_end + 1, int(len(partitions) * (train_ratio + validation_ratio)))
    validation_end = min(validation_end, len(partitions) - 1)
    return partitions[:train_end], partitions[train_end:validation_end], partitions[validation_end:]


def run_smoothing_experiment(
    documents: Iterable[str],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
    seed: int = 42,
    unk_cutoff: int = 1,
) -> list[dict[str, int | float | str]]:
    """Run section 16: bigram/trigram MLE versus Laplace on three splits."""
    document_splits = split_documents(documents, train_ratio, validation_ratio, seed)
    sentence_splits = [list(iter_tokenized_sentences(split)) for split in document_splits]
    if any(not split for split in sentence_splits):
        raise ValueError("Every document split must contain at least one tokenized sentence")

    results: list[dict[str, int | float | str]] = []
    for n, model_name in ((2, "Bigram"), (3, "Trigram")):
        for smoothing in ("mle", "laplace"):
            model = NGramLanguageModel(n, smoothing, unk_cutoff).fit_tokenized(sentence_splits[0])
            results.append(
                {
                    "model": model_name,
                    "smoothing": smoothing.upper() if smoothing == "mle" else "Laplace",
                    "train_ppl": model.perplexity(sentence_splits[0]),
                    "validation_ppl": model.perplexity(sentence_splits[1]),
                    "test_ppl": model.perplexity(sentence_splits[2]),
                    "vocabulary_size": len(model.vocabulary),
                }
            )
    return results


def save_experiment_results(results: list[dict], path: str | Path) -> None:
    """Save the section 16 table as CSV; infinite perplexity is written as inf."""
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "model", "smoothing", "train_ppl", "validation_ppl", "test_ppl", "vocabulary_size"
        ))
        writer.writeheader()
        writer.writerows(results)


def _format_metric(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.6f}"


def interactive_session(
    model: NGramLanguageModel, top_k: int = 10,
    output: str | Path | None = None,
) -> list[dict]:
    """Interactive sections 20-21; optionally save user trials as CSV."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    model._require_fitted()
    trials = []
    contexts_tried = set()
    print("\nPhan 20-21: Du doan tu tiep theo / Xep hang phan tiep noi")
    print("Thu it nhat 5 contexts, vi du: the cat; natural language; machine learning; i like; this is")
    print("Nhap /back de quay lai, /quit de thoat. Ctrl+C/Ctrl+Z cung ket thuc.")

    def ask(prompt: str) -> str:
        answer = input(prompt).strip()
        if answer.lower() == "/quit":
            raise EOFError
        return answer

    try:
        while True:
            print("\n1. Du doan tu tiep theo\n2. Xep hang candidates\n3. Xem lich su\n0. Thoat")
            choice = ask("Chon: ")
            if choice == "0":
                break
            if choice == "1":
                while True:
                    context = ask("Context (rong = quay lai): ")
                    if not context or context == "/back":
                        break
                    if not tokenize(context):
                        print("Context can co it nhat mot tu.")
                        continue
                    history = model._context(context)
                    order = len(history) + 1
                    if order > 1 and not model.context_counts[order][history]:
                        if model.smoothing == "laplace":
                            print("Context chua thay: Laplace cho xac suat bang nhau; thu tu hoa duoc sap theo chu cai.")
                    predictions = [
                        (word, p) for word, p in model.next_word_distribution(context).items()
                        if p > 0 and word != UNK_TOKEN
                    ][:top_k]
                    if not predictions:
                        print("Khong co du doan duong cho context nay; thu Laplace smoothing.")
                    for rank, (word, p) in enumerate(predictions, 1):
                        print(f"{rank}. {word}\t{p:.8g}")
                    actual = ask("Actual next word (tuy chon, Enter de bo qua): ")
                    if actual == "/back":
                        break
                    if actual and len(tokenize(actual)) != 1:
                        print("Actual next word phai la mot token; khong ghi gia tri nay.")
                        actual = ""
                    word, p = predictions[0] if predictions else ("", 0.0)
                    trials.append({"task": "next_word", "context": context,
                                   "prediction": word, "actual": actual,
                                   "probability": p, "log_probability": math.log(p) if p else -math.inf})
                    contexts_tried.add(tuple(tokenize(context)))
                    print(f"Da thu {len(contexts_tried)} contexts khac nhau / muc tieu >= 5.")
            elif choice == "2":
                context = ask("Context (Enter = context rong): ")
                if context == "/back":
                    continue
                candidates = []
                print("Nhap moi candidate tren mot dong; Enter rong de xep hang (it nhat 2).")
                while True:
                    candidate = ask(f"Candidate {len(candidates) + 1}: ")
                    if candidate == "/back":
                        candidates = []
                        break
                    if not candidate:
                        if len(candidates) >= 2:
                            break
                        print("Can it nhat 2 candidates; /back de huy.")
                    elif tokenize(candidate):
                        candidates.append(candidate)
                    else:
                        print("Candidate can co it nhat mot tu.")
                if not candidates:
                    continue
                mode = ask("Xep hang: 1 = log P(A|context), 2 = log P/token [1]: ")
                if mode == "/back":
                    continue
                if mode not in ("", "1", "2"):
                    print("Lua chon khong hop le; quay lai menu.")
                    continue
                normalized = mode == "2"
                print("Score: " + ("log P/token" if normalized else "log P(A|context)"))
                print("Raw P thuong uu tien candidate ngan; P co the underflow ve 0, thu hang dung log.")
                for rank, row in enumerate(model.rank_continuations(context, candidates, normalized), 1):
                    print(f"{rank}. {row['candidate']} | P={row['probability']:.8g} | "
                          f"log P={row['log_probability']:.6f} | score={row['score']:.6f}")
                    trials.append({"task": "ranking", "context": context,
                                   "prediction": row["candidate"], "actual": "",
                                   "probability": row["probability"],
                                   "log_probability": row["log_probability"]})
            elif choice == "3":
                for trial in trials:
                    print(f"{trial['task']} | {trial['context']} -> {trial['prediction']} | "
                          f"actual={trial['actual']} | P={trial['probability']:.8g}")
                print(f"Tong: {len(trials)} dong; {len(contexts_tried)} contexts du doan khac nhau.")
            else:
                print("Vui long chon 0, 1, 2 hoac 3.")
    except (EOFError, KeyboardInterrupt):
        print("\nKet thuc tuong tac.")
    if output is not None and trials:
        with Path(output).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trials[0]))
            writer.writeheader()
            writer.writerows(trials)
        print(f"Da luu lich su: {output}")
    return trials


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--n", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--smoothing", choices=("mle", "laplace"), default="mle")
    parser.add_argument("--sentence", help="Optional single token sequence to score")
    parser.add_argument("--context", help="Optional context for top-10 next words")
    parser.add_argument("--interactive", action="store_true", help="Interactive applications for sections 20-21")
    parser.add_argument("--top-k", type=int, default=10, help="Number of next-word predictions")
    parser.add_argument("--application-output", type=Path, help="Save interactive trials as CSV on exit")
    parser.add_argument("--experiment2", action="store_true", help="Run MLE vs Laplace experiment")
    parser.add_argument("--output", type=Path, help="Save Experiment 2 results as CSV")
    parser.add_argument("--seed", type=int, default=42, help="Document split seed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--max-documents", type=int, help="Optional quick-run document limit")
    parser.add_argument(
        "--unk-cutoff",
        type=int,
        default=1,
        help="Experiment 2 / interactive mode maps training words with count <= cutoff to <UNK>",
    )
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.interactive and args.experiment2:
        parser.error("Choose either --interactive or --experiment2")
    documents = load_documents(args.corpus)
    if args.max_documents is not None:
        if args.max_documents < 3:
            parser.error("--max-documents must be at least 3")
        from itertools import islice

        documents = islice(documents, args.max_documents)
    if args.experiment2:
        results = run_smoothing_experiment(
            documents,
            args.train_ratio,
            args.validation_ratio,
            args.seed,
            args.unk_cutoff,
        )
        if args.output is not None:
            save_experiment_results(results, args.output)
        print("| Model | Smoothing | Train PPL | Valid PPL | Test PPL | Vocabulary |")
        print("|---|---|---:|---:|---:|---:|")
        for row in results:
            print(
                f"| {row['model']} | {row['smoothing']} | "
                f"{_format_metric(row['train_ppl'])} | "
                f"{_format_metric(row['validation_ppl'])} | "
                f"{_format_metric(row['test_ppl'])} | {row['vocabulary_size']:,} |"
            )
        return
    print("Training model...", flush=True)
    model = NGramLanguageModel(
        args.n, args.smoothing, args.unk_cutoff if args.interactive else 0
    ).fit(documents)
    print(f"Order: {model.n}; smoothing: {model.smoothing}")
    print(f"Sentences: {model.num_sentences:,}; tokens: {model.total_tokens:,}")
    print(f"Vocabulary: {len(model.vocabulary):,}")
    for order, counts in model.counts.items():
        print(f"{order}-grams: {sum(counts.values()):,} total; {len(counts):,} unique")
    if args.sentence is not None:
        print("Sentence probability:", model.sentence_probability(args.sentence))
        print("Sentence log probability:", model.sentence_log_probability(args.sentence))
    if args.context is not None:
        for word, probability in list(model.next_word_distribution(args.context).items())[:10]:
            print(f"{word}\t{probability:.8g}")
    if args.interactive:
        interactive_session(model, args.top_k, args.application_output)


if __name__ == "__main__":
    main()
