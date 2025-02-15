from collections import defaultdict
import re
import matplotlib.pyplot as plt

toy_dataset = ["low", "lower", "newest", "widest"]


def init_vocab(dataset):
    vocab_dict = {" ".join(word) + " .": 1 for word in dataset}
    return vocab_dict


def get_pair_freq(vocab_dict):
    word_pairs = defaultdict(int)
    for word, freq in vocab_dict.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            word_pairs[(symbols[i], symbols[i + 1])] += freq
    return word_pairs


def merge_pair(pair, vocab):
    pattern = re.escape(" ".join(pair))  # Convert ('A', 'B') -> 'A B'
    replacement = "".join(pair)  # Convert ('A', 'B') -> 'AB'
    new_vocab = {}

    for word in vocab:
        new_word = re.sub(pattern, replacement, word)  # Replace the pair in the text
        new_vocab[new_word] = vocab[word]  # Preserve frequencies

    return new_vocab


def plot_bpe_progress(merge_steps):
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(merge_steps)), merge_steps, marker='o', linestyle='-')
    plt.xlabel("Merge Step")
    plt.ylabel("Number of Unique Tokens")
    plt.title("BPE Token Evolution")
    plt.show()


def bpe(words, num_merges=10):
    vocab = init_vocab(words)
    merges = []
    merge_steps = [len(vocab)]

    for _ in range(num_merges):
        pairs = get_pair_freq(vocab)
        #print("pairs", pairs)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        vocab = merge_pair(best_pair, vocab)
        merges.append(best_pair)
        merge_steps.append(len(vocab))
    plot_bpe_progress(merge_steps)
    return vocab, merges


# orig_vocab = init_vocab(toy_dataset)
# pairs = get_pair_freq(orig_vocab)
# print("orig vocab", orig_vocab)
# print("pairs", pairs)
# best_pair = max(pairs, key=pairs.get)  # Get most frequent pair
# modified_vocab = merge_pair(best_pair, orig_vocab)
# print("modified vocab", modified_vocab)
file_path = "../alice.txt"  # Change this to the path of your file
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()
print("text length", len(text))
bpe_vocab, merge_operations = bpe(text, num_merges=20)
#print("Final BPE Vocabulary:", bpe_vocab)
#print("Merge Operations:", merge_operations)
print(len(bpe_vocab))