# Michael Taylor
# Predict the next word of a sentence using a N-Gram Language Model

import nltk
import sys
from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from collections import defaultdict
import copy
import time


def read_blogs(blogs_folder_path):
    """
    Purpose: Parse all training blogs/data so we can make use of it
    :param blogs_folder_path: Path to folder containing all blogs/training data
    :return: All training data combined into one long string of text
    """
    print("\nConverting blog to string...")
    # Start with blank string for output
    output_string = ""
    # Initiate a plaintext corpus reader to read the files. The corpus reader sorts files by ascending UserID for us
    blog_corpus = PlaintextCorpusReader(blogs_folder_path, ".*", encoding="ISO-8859-1")
    file_names = blog_corpus.fileids()
    # Creates dict that will map file ID #'s to the file names, this is to open files in ascending user ID
    file_id_map = {}
    for file_name in file_names:
        split_str = file_name.split(".", maxsplit=1)
        # Grab the ID number at the front of the file
        id_num = int(split_str[0])
        file_id_map[id_num] = file_name
    sorted_ids = sorted(file_id_map.keys())

    for file_id in sorted_ids:
        file_name = file_id_map[file_id]
        file = blog_corpus.open(file_name)
        # in_bounds; True when inside <post>. . .</post>, False when outside the post tags
        in_bounds = False
        for line in file.readlines():
            # For each line of the blog, strip white space, if in bounds, then add to output string
            line = line.strip()
            if "</post>" in line:
                in_bounds = False
            if len(line) > 0 and in_bounds:
                output_string += line
            if "<post>" in line:
                in_bounds = True
    return str(output_string)


def count_words(text):
    """
    Purpose: Insert sentence boundaries into inputted text and fetch all 1-grams, 2-grams, and 3-grams & record their # of occurrences
    :param text: One long string of training text
    :return: tuple({str(unigram) : int(# of occurrences), str(bigram) : int(# of occurrences), str(trigram) : int(# of occurrences)})
             tuple({}, {}, {})

    Output: Prints # of tokens and # of types
    """
    print("\nGenerating unigrams, bigrams, and trigrams frequency dicts...")
    # Create empty maps for all n-grams and insert into a list to contain them
    unigrams, bigrams, trigrams = defaultdict(int), defaultdict(int), defaultdict(int)
    n_grams = [unigrams, bigrams, trigrams]
    # Break the text down into sentences
    sentences = sent_tokenize(text)
    for phrase in sentences:
        # Break each sentence down into words and insert sentence boundaries
        phrase = phrase.lower()
        phrase_tokens = word_tokenize(phrase)
        phrase_tokens.insert(0, "<s>")
        phrase_tokens.insert(len(phrase_tokens), "</s>")
        n_tokens = len(phrase_tokens)
        # Grab n-grams from the sentence and store in corresponding n-gram map within n_grams
        for i in range(n_tokens):
            for n in range(1, 4):
                # If statement to avoid going out of bounds
                if i < n_tokens - n + 1:
                    # Stitch together n-grams by joining word up to word+n with spaces
                    n_gram = " ".join(phrase_tokens[i : i + n])
                    n_grams[n - 1][n_gram] += 1
    # Print and return requested information
    print("Number of tokens:", sum(unigrams.values()))
    print("Number of types:", len(unigrams.keys()))
    return tuple(n_grams)


def print_frequent_n_grams(unigrams, bigrams, trigrams, k):
    """
    Purpose: Display most frequent n-grams from training data

    :param unigrams: dict of unigrams and their frequency in the training data
    :param bigrams: dict of bigrams and their frequency in the training data
    :param trigrams: dict of trigrams and their frequency in the training data
    :param k: Amount of most frequent n-grams to print

    Output: Prints k most frequent unigrams, bigrams, and trigrams
    """
    print("\nDisplaying most common n-grams...")
    print("Most common unigrams:", nltk.FreqDist(unigrams).most_common(k))
    print("Most common bigrams:", nltk.FreqDist(bigrams).most_common(k))
    print("Most common trigrams:", nltk.FreqDist(trigrams).most_common(k))
    pass


def predict(text, unigrams, bigrams, trigrams):
    """
    Purpose: Predict the next word of an inputted sentence based on n-grams from a corpus, n=[1:3]

    :param text: string of words to predict next word from
    :param unigrams: dict of unigrams and their frequency in the training data
    :param bigrams: dict of bigrams and their frequency in the training data
    :param trigrams: dict of trigrams and their frequency in the training data

    Output: Prints suggestions for each n-gram model
    """
    # Created deep copies to ensure Laplace smoothing doesn't accumulate every call of the predict function
    unigrams, bigrams, trigrams = (
        copy.deepcopy(unigrams),
        copy.deepcopy(bigrams),
        copy.deepcopy(trigrams),
    )

    print("\nPredicting next word using n-gram models...")
    print("Input:", text)
    words = text.split()
    last_word = words[-1]  # Last word of sentence (for bigrams)
    last_2words = " ".join(words[-2:])  # Last 2 words of sentence (for trigrams)
    # Make 1-gram predictions
    for unigram in unigrams:
        unigrams[unigram] += 1
    unigram_freqdist = nltk.FreqDist(unigrams)
    suggested = unigram_freqdist.max()
    suggested_prob = unigrams[suggested] / (sum(unigrams.values()))
    print("1-gram suggestion:", suggested, "with a probability of", suggested_prob)
    # Make 2-gram predictions
    possibilities = {}
    for bigram in bigrams.keys():
        bigrams[bigram] += 1  # Add 1 for Laplace smoothing
        # Grab bigrams that start with the last inputted word
        if bigram.split()[0] == last_word:
            possibilities[bigram] = bigrams[bigram]
    # Turn possible 2-grams into a frequency distributions
    possibilities = nltk.FreqDist(possibilities)
    if len(possibilities) != 0:
        suggested = possibilities.max()
        suggested_prob = bigrams[suggested] / (
            unigrams[last_word] + len(unigrams.keys())
        )
        print("2-gram suggestion:", suggested, "with a probability of", suggested_prob)
    else:
        print("No 2-gram suggestions available.")
    # Make 3-gram predictions
    possibilities = {}
    for trigram in trigrams.keys():
        trigrams[trigram] += 1  # Add 1 for Laplace smoothing
        # Grab trigrams that start with the last 2 inputted words
        if " ".join(trigram.split()[:2]) == last_2words:
            possibilities[trigram] = trigrams[trigram]
    # Turn possible 3-grams into a frequency distributions
    possibilities = nltk.FreqDist(possibilities)
    if len(possibilities) != 0:
        # Grab suggestion with the greatest frequency
        suggested = possibilities.max()
        suggested_prob = trigrams[suggested] / (
            bigrams[last_2words] + len(unigrams.keys())
        )
        print("3-gram suggestion:", suggested, "with a probability of", suggested_prob)
    else:
        print("No 3-gram suggestions available.")
    pass


def main():
    blogs_dir_path = sys.argv[1]
    blog_text = read_blogs(blogs_dir_path)
    unigrams, bigrams, trigrams = count_words(blog_text)
    print_frequent_n_grams(unigrams, bigrams, trigrams, 5)
    sample_texts = ["the past few", "pass the salt and", "jump over", "in the"]
    print("Time Elapsed: ", time.time() - time_start)
    for sample in sample_texts:
        predict(sample, unigrams, bigrams, trigrams)


if __name__ == "__main__":
    time_start = time.time()
    main()
    print("Time Elapsed: ", time.time() - time_start)
