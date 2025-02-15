import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from typing import List, Dict, Tuple, Set
from tqdm import tqdm


# class BPETokenizer:
#     def __init__(self, vocab_size: int):
#         self.vocab_size = vocab_size
#         self.vocab: Set[str] = set()
#         self.merges: List[Tuple[str, str]] = []
#
#     def get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
#         pairs = defaultdict(int)
#         for word, freq in word_freqs.items():
#             symbols = word.split()
#             for i in range(len(symbols) - 1):
#                 pairs[symbols[i], symbols[i + 1]] += freq
#         return pairs
#
#     def merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
#         new_word_freqs = {}
#         bigram = ' '.join(pair)
#         replacement = ''.join(pair)
#
#         for word, freq in word_freqs.items():
#             new_word = word.replace(bigram, replacement)
#             new_word_freqs[new_word] = freq
#
#         return new_word_freqs
#
#     def get_sequence_lengths(self, texts: List[str], steps: List[int]) -> Dict[int, List[int]]:
#         """Track sequence lengths at different merge steps."""
#         sequence_lengths = defaultdict(list)
#
#         # Initialize word frequencies
#         word_freqs = defaultdict(int)
#         for text in texts:
#             words = text.split()
#             for word in words:
#                 word = ' '.join(list(word.lower())) + ' </w>'
#                 word_freqs[word] += 1
#                 sequence_lengths[0].append(len(word.split()))
#
#         # Initialize vocabulary with characters
#         self.vocab = set(char for word in word_freqs.keys() for char in word.split())
#
#         # Track lengths at each requested step
#         current_merges = []
#         for i in tqdm(range(max(steps))):
#             pairs = self.get_stats(word_freqs)
#             if not pairs:
#                 break
#
#             best_pair = max(pairs.items(), key=lambda x: x[1])[0]
#             current_merges.append(best_pair)
#             word_freqs = self.merge_pair(best_pair, word_freqs)
#
#             if i + 1 in steps:
#                 # Calculate sequence lengths at this step
#                 lengths = []
#                 for text in texts:
#                     tokens = self.tokenize_with_merges(text, current_merges)
#                     lengths.append(len(tokens))
#                 sequence_lengths[i + 1] = lengths
#
#         return sequence_lengths
#
#     def tokenize_with_merges(self, text: str, merges: List[Tuple[str, str]]) -> List[str]:
#         """Tokenize text using only the specified merges."""
#         words = text.lower().split()
#         tokens = []
#
#         for word in words:
#             word = ' '.join(list(word)) + ' </w>'
#
#             for old, new in zip(merges, [(''.join(p)) for p in merges]):
#                 bigram = ' '.join(old)
#                 word = word.replace(bigram, new)
#
#             tokens.extend(word.split())
#
#         return tokens
class BPETokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.vocab: Set[str] = set()
        self.merges: List[Tuple[str, str]] = []

    def read_file(self, file_path: str) -> List[str]:
        """Read and preprocess text from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()  # Lowercase all text
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            return sentences

    def get_stats(self, word_freqs: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_pair(self, pair: Tuple[str, str], word_freqs: Dict[str, int]) -> Dict[str, int]:
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq

        return new_word_freqs

    def get_sequence_lengths(self, file_path: str, steps: List[int]) -> Dict[int, List[int]]:
        sequence_lengths = defaultdict(list)

        # Read and process file
        texts = self.read_file(file_path)
        print(f"Processed {len(texts)} sentences from file")

        # Initialize word frequencies
        word_freqs = defaultdict(int)
        for text in texts:
            words = text.split()
            for word in words:
                word = ' '.join(list(word.lower())) + ' </w>'  # Lowercase each word
                word_freqs[word] += 1
                sequence_lengths[0].append(len(word.split()))

        # Initialize vocabulary with characters
        self.vocab = set(char for word in word_freqs.keys() for char in word.split())
        print(f"Initial vocabulary size: {len(self.vocab)}")

        current_merges = []
        for i in tqdm(range(max(steps)), desc="Processing merge steps"):
            pairs = self.get_stats(word_freqs)
            if not pairs:
                break

            # Print top pairs for debugging
            top_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:5]
            print(f"\nTop 5 pairs at step {i}:")
            for pair, freq in top_pairs:
                print(f"Pair: {pair}, Frequency: {freq}")

            best_pair = max(pairs.items(), key=lambda x: x[1])[0]
            current_merges.append(best_pair)
            word_freqs = self.merge_pair(best_pair, word_freqs)

            if i + 1 in steps:
                sample_size = min(1000, len(texts))
                sample_texts = texts[:sample_size]

                lengths = []
                for text in sample_texts:
                    tokens = self.tokenize_with_merges(text, current_merges)
                    lengths.append(len(tokens))
                sequence_lengths[i + 1] = lengths

                print(f"Step {i + 1}: Average sequence length = {np.mean(lengths):.2f}")

        return sequence_lengths

    def tokenize_with_merges(self, text: str, merges: List[Tuple[str, str]]) -> List[str]:
        words = text.lower().split()  # Lowercase input text
        tokens = []

        for word in words:
            word = ' '.join(list(word)) + ' </w>'

            for old, new in zip(merges, [(''.join(p)) for p in merges]):
                bigram = ' '.join(old)
                word = word.replace(bigram, new)

            tokens.extend(word.split())

        return tokens

def visualize_bpe_compression():
    # Sample text
    # texts = [
    #     "hello world",
    #     "hello there",
    #     "hi world",
    #     "hello hello world",
    #     "the quick brown fox jumps over the lazy dog"
    # ]
    file_path = "../alice.txt"  # Change this to the path of your file
    # with open(file_path, "r", encoding="utf-8") as file:
    #     text = file.read().lower()
    # sentences = [s.strip() for s in text.split('.') if s.strip()]
    tokenizer = BPETokenizer(vocab_size=100)
    steps = [0, 5, 10, 20, 30, 50]  # Track these merge steps

    # Get sequence lengths at different steps
    sequence_lengths = tokenizer.get_sequence_lengths(file_path, steps)

    # Prepare data for plotting
    plt.figure(figsize=(12, 6))

    # Box plot for sequence lengths
    data = [sequence_lengths[step] for step in steps]

    plt.boxplot(data, labels=[f'Step {step}' for step in steps])
    plt.xlabel('Merge Steps')
    plt.ylabel('Sequence Length')
    plt.title('BPE Merge Steps vs Sequence Length Distribution')

    # Add mean line
    means = [np.mean(lengths) for lengths in data]
    plt.plot(range(1, len(steps) + 1), means, 'r--', label='Mean Length')

    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add text statistics
    avg_reduction = (means[0] - means[-1]) / means[0] * 100
    plt.text(0.02, 0.98,
             f'Average length reduction: {avg_reduction:.1f}%\n'
             f'Initial mean length: {means[0]:.1f}\n'
             f'Final mean length: {means[-1]:.1f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='top')

    plt.tight_layout()
    plt.show()

    # Print example tokenizations
    # file_path = "../alice.txt"  # Change this to the path of your file
    # with open(file_path, "r", encoding="utf-8") as file:
    #     example_text = file.read()
    #print(f"\nTokenization evolution for '{example_text}':")
    # for step in steps:
    #     tokens = tokenizer.tokenize_with_merges(example_text, tokenizer.merges[:step])
        #print(f"Step {step}: {tokens}")


if __name__ == "__main__":
    visualize_bpe_compression()