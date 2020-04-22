import sys

import pandas as pd
import jieba
import re

bad_words = {'|', '[', ']', '语音', '图片'}
with open("/Users/quqi/nlp_projects/data_in/stop_words.txt", mode='r', encoding="utf-8") as f:
    for line in f:
        word = line.strip()
        bad_words.add(word)


def drop_yn_fill_xn(df):
    try:
        df.dropna(subset=['Report'], how='any', inplace=True)
    except KeyError:
        pass
    df.fillna('', inplace=True)
    res_x = df.Question.str.cat(df.Dialogue, sep=' ')
    res_y = []
    try:
        res_y = df.Report
        assert len(res_x) == len(res_y)
    except AttributeError:
        pass
    print("res_x: {}".format(len(res_x)))
    print("res_y: {}".format(len(res_y)))
    return res_x, res_y


def tokenize_df(df):
    for _, series in df.iterrows():
        sentence = series['Question'] + series['Dialogue']
        if not isinstance(sentence, str) or len(sentence) == 0:
            print("ALERT: {}".format(sentence))
            print(series['Qid'])
            sys.exit(0)


def tokenize_series_and_write_to_file(series, filename):
    lines = 0
    with open(filename, 'w', encoding='utf8') as f:
        for sentence in series:
            sentence = re.sub('\s+', '', sentence)
            words = jieba.lcut(sentence)
            words = filter(lambda w: w not in bad_words, words)
            words_str = ' '.join(words)
            f.write(words_str + "\n")
            lines += 1
    print("{}: {}".format(filename, lines))


if __name__ == "__main__":
    test_path = "/Users/quqi/nlp_projects/data_in/AutoMaster_TestSet.csv"
    train_path = "/Users/quqi/nlp_projects/data_in/AutoMaster_TrainSet.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_x, train_y = drop_yn_fill_xn(train_df)
    test_x, _ = drop_yn_fill_xn(test_df)

    tokenize_series_and_write_to_file(train_x, "/Users/quqi/nlp_projects/data_out/train_x.txt")
    tokenize_series_and_write_to_file(train_y, "/Users/quqi/nlp_projects/data_out/train_y.txt")
    tokenize_series_and_write_to_file(test_x, "/Users/quqi/nlp_projects/data_out/test_x.txt")