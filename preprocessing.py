# -*- coding:utf-8 -*-
import csv
import numpy as np
import itertools
import nltk
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

def quicksort(x):
    if x==[]: return []

    smallerSorted = quicksort([a for a in x[1:] if len(a) <= len(x[0])])
    biggerSorted = quicksort([a for a in x[1:] if len(a) > len(x[0])])

    return(smallerSorted+[x[0]]+biggerSorted)   


def getSentenceData(path, vocabulary_size=8000, class_dim=100):
    unknown_token = "UNKNOWN_TOKEN"
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    print("Reading CSV file...")
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print("Parsed %d sentences." % (len(sentences)))

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Filter the sentences having few words (including SENTENCE_START and SENTENCE_END)
    tokenized_sentences = list(filter(lambda x: len(x) > 3, tokenized_sentences))

    # Count the word frequencies
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print("Found %d unique words tokens." % len(word_freq.items()))

    # Get the most common words and build index_to_word and word_to_index vectors
    # 定めた語彙数文だけの単語でvocabを作る
    vocab = word_freq.most_common(vocabulary_size-1)
    '''
    V = word_freq.most_common()
    icount = np.array([x[1] for x in V])
    num_tokens = np.sum(icount)
    '''
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])
    index_to_count = [x[1] for x in vocab]
    
    
    
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print("\nExample sentence: '%s'" % sentences[1])
    print("\nExample sentence after Pre-processing: '%s'\n" % tokenized_sentences[0])

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    # 単語が所属するクラスをクラス分布で表す
    index_to_class_dist = []
    num_tokens = 0
    for i in range(vocabulary_size-1):
        num_tokens += index_to_count[i]
    for i in range(vocabulary_size):
        index_to_class_dist.append([0.0] * class_dim)
        
    class_to_list = []
    for i in range(class_dim):
        class_to_list.append([])
    # 最後のクラスは未知語をもつ
    class_to_list[class_dim - 1].append(vocabulary_size - 1)
    # 未知語はクラスclass_size-1を予測する
    index_to_class_dist[vocabulary_size - 1][class_dim - 1] = 1.0
    
    df = 0.0
    a = 0
    for i in range(vocabulary_size-1):
        df += index_to_count[i] / num_tokens
        if df > 1.0:
            df = 1.0
        if df > (a + 1) / (class_dim - 1):
            index_to_class_dist[i][a] = 1.0
            class_to_list[a].append(i)
            if a < class_dim - 2:
                a += 1
        else:
            index_to_class_dist[i][a] = 1.0
            class_to_list[a].append(i)

    # 作成したクラスターを表示する
    '''
    for i in range(class_dim):
        print("\nclass %d"%i)
        word_list = class_to_list[i]
        for w in word_list:
            print("%d : %s"%(w,index_to_word[w]))
    '''
            
    #文の長さの分布を求める
    '''
    max_len = 0
    min_len = 10000
    num_sents = len(X_train)
    sents_list = []
    for k in range(num_sents):
        v = len(X_train[k])
        if v < 100:
            sents_list.append(v)
        else:
            print(v)
        if max_len < v:
            max_len = v
            print("max length %d"%max_len)
        if min_len > v:
            min_len = v
    print("%d / %d"%(len(sents_list),num_sents))
    plt.hist(sents_list,bins=20)
    plt.show()
    '''
    #文の長さで教師データをソートする
    #X_train = np.asarray(quicksort(X_train))
    #y_train = np.asarray(quicksort(y_train))

    print("X_train shape: " + str(X_train.shape))
    print("y_train shape: " + str(y_train.shape))
    # Print an training data example
    x_example, y_example = X_train[17], y_train[17]
    print("x:\n%s\n%s" % (" ".join([index_to_word[x] for x in x_example]), x_example))
    print("\ny:\n%s\n%s" % (" ".join([index_to_word[x] for x in y_example]), y_example))

    return X_train, y_train

if __name__ == '__main__':
    X_train, y_train = getSentenceData('data/reddit-comments-2015-08.csv')
