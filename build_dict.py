from collections import defaultdict
import operator

word_dict = defaultdict(int)


def read_words_from_file(filename):
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            words = line.split(' ')
            for word in words:
                word_dict[word] += 1


if __name__ == "__main__":
    train_x = "/Users/quqi/nlp_projects/data_out/train_x.txt"
    train_y = "/Users/quqi/nlp_projects/data_out/train_y.txt"
    test_x = "/Users/quqi/nlp_projects/data_out/test_x.txt"

    read_words_from_file(train_x)
    read_words_from_file(train_y)
    read_words_from_file(test_x)

    vocab = "/Users/quqi/nlp_projects/data_out/vocab.txt"
    vocab_freq = "/Users/quqi/nlp_projects/data_out/vocab_freq.txt"
    sorted_dict = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
    with open(vocab, 'w', encoding='utf-8') as f:
        with open(vocab_freq, 'w', encoding='utf-8') as f_freq:
            for idx, (word, freq) in enumerate(sorted_dict):
                word = word.strip()
                f.write("{}\t{}\n".format(word, idx))
                f_freq.write("{}\t{}\n".format(word, freq))

