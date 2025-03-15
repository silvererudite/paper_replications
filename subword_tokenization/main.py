from collections import defaultdict
import re
import random
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt


def preprocess_text(text: str) -> List[str]:
    """Preprocess text into words, preserving important punctuation."""
    # Normalize spacing and convert to lowercase
    text = re.sub(r'\s+', ' ', text.lower()).strip()
    # Split into words while preserving sentence boundaries
    words = text.split()
    return words


def init_vocab(words: List[str]) -> Dict[str, int]:
    """Initialize vocabulary with character-level tokens and their frequencies."""
    vocab_dict = defaultdict(int)
    for word in words:
        # Split word into characters with spaces between
        char_seq = ' '.join(list(word)) + ' </w>'  # </w> marks word boundary
        vocab_dict[char_seq] += 1
    return dict(vocab_dict)


def split_data(words: List[str], train_ratio: float = 0.8) -> Tuple[List[str], List[str]]:
    """Split data into training and test sets."""
    random.shuffle(words)
    split_idx = int(len(words) * train_ratio)
    return words[:split_idx], words[split_idx:]


def evaluate_compression(test_words: List[str], merges: List[Tuple[str, str]]) -> Dict[str, float]:
    """Evaluate BPE compression on test data."""
    # Original character-level representation
    orig_chars = sum(len(word) for word in test_words)
    orig_tokens = sum(len(word.split()) for word in
                      [' '.join(list(word)) + ' </w>' for word in test_words])
    print("merges", merges)
    print("original tokens", orig_tokens)

    # Apply BPE merges
    compressed_words = [apply_bpe(word, merges) for word in test_words]
    print("compressed words", compressed_words)
    compressed_tokens = sum(len(word.split()) for word in compressed_words)

    return {
        'compression_ratio': compressed_tokens / orig_tokens,
        'orig_tokens_per_word': orig_tokens / len(test_words),
        'compressed_tokens_per_word': compressed_tokens / len(test_words)
    }

def get_pair_freq(vocab_dict: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    """Count frequencies of adjacent token pairs."""
    pairs = defaultdict(int)
    for word, freq in vocab_dict.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_pair(pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
    """Merge all occurrences of the given pair in the vocabulary."""
    pattern = re.escape(' '.join(pair))
    replacement = ''.join(pair)
    new_vocab = {}

    for word, freq in vocab.items():
        new_word = re.sub(pattern, replacement, word)
        new_vocab[new_word] = freq

    return new_vocab


def compute_stats(vocab: Dict[str, int]) -> Tuple[int, float]:
    """Compute vocabulary size and average sequence length."""
    vocab_size = len(vocab)
    avg_seq_length = sum(len(word.split()) for word in vocab) / len(vocab)
    return vocab_size, avg_seq_length


def plot_bpe_progress(merge_steps: List[int], sequence_lengths: List[float]):
    """Plot BPE progress metrics."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(merge_steps)), merge_steps,
             marker='o', linestyle='-', label="Vocabulary Size")
    plt.plot(range(len(sequence_lengths)), sequence_lengths,
             marker='s', linestyle='-', label="Avg Token Sequence Length",
             color='red')
    plt.xlabel("Merge Step")
    plt.ylabel("Count")
    plt.title("BPE Progress")
    plt.legend()
    plt.grid(True)
    plt.show()


def learn_and_evaluate_bpe(text: str, num_merges: int = 100) -> None:
    """Learn BPE on training data and evaluate on test data."""
    # Preprocess and split data
    words = preprocess_text(text)
    train_words, test_words = split_data(words)
    print(f"Training on {len(train_words)} words, testing on {len(test_words)} words")

    # Learn BPE merges on training data
    vocab, merges = learn_bpe(train_words, num_merges)

    # Evaluate on test data
    metrics = evaluate_compression(test_words, merges)

    print("\nEvaluation on test data:")
    print(f"Compression ratio: {metrics['compression_ratio']:.3f}")
    print(f"Original tokens per word: {metrics['orig_tokens_per_word']:.2f}")
    print(f"Compressed tokens per word: {metrics['compressed_tokens_per_word']:.2f}")

    # Plot compression progress on test data
    compression_progress = []
    for i in range(0, len(merges), 10):  # Evaluate every 10 merges
        partial_metrics = evaluate_compression(test_words, merges[:i + 1])
        compression_progress.append(partial_metrics['compression_ratio'])

    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(merges), 10), compression_progress)
    plt.xlabel("Number of Merges")
    plt.ylabel("Compression Ratio")
    plt.title("BPE Compression Progress on Test Data")
    plt.grid(True)
    plt.show()
def learn_bpe(words: List[str], num_merges: int = 3) -> Tuple[Dict[str, int], List[Tuple[str, str]]]:
    """Learn BPE merges from text."""
    vocab = init_vocab(words)
    print("Initial vocab size", len(vocab))
    merges = []
    merge_steps = [len(vocab)]
    sequence_lengths = [sum(len(word.split()) for word in vocab) / len(vocab)]

    for i in range(num_merges):
        pairs = get_pair_freq(vocab)
        if not pairs:
            break

        best_pair = max(pairs.items(), key=lambda x: x[1])[0]
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)

        # Track statistics
        vocab_size, avg_length = compute_stats(vocab)
        merge_steps.append(vocab_size)
        sequence_lengths.append(avg_length)

        if i % 10 == 0:  # Print progress every 10 steps
            print(f"Merge {i}: vocab_size={vocab_size}, avg_length={avg_length:.2f}")

    plot_bpe_progress(merge_steps, sequence_lengths)
    return vocab, merges


def apply_bpe(word: str, merges: List[Tuple[str, str]]) -> str:
    """Apply learned BPE merges to a new word."""
    word = ' '.join(list(word)) + ' </w>'
    for pair in merges:
        word = re.sub(re.escape(' '.join(pair)), ''.join(pair), word)
        print("word", word)
    return word


# Example usage
if __name__ == "__main__":
    with open("../test.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # words = preprocess_text(text)
    # vocab, merges = learn_bpe(words, num_merges=100)
    learn_and_evaluate_bpe(text, num_merges=20)

    # print("\nFinal vocabulary size:", len(vocab))
    # print("\nSample of final vocabulary:")
    # for word, freq in list(vocab.items())[:10]:
    #     print(f"{word}: {freq}")