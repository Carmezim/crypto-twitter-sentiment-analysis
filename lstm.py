import numpy as np
import argparse
import os
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM
import utils
from keras.preprocessing.sequence import pad_sequences

# Performs classification using LSTM network.

parser = argparse.ArgumentParser(description='files')
parser.add_argument('--train', required=False, help='training data')
parser.add_argument('--test', required=False, help='test data')
parser.add_argument('--freq', required=False, help='frequence distribution file')
parser.add_argument('--bifreq', required=False, help='binomial frequency distribution')
parser.add_argument('--model', required=False, help='model to be used for predictions')
args = parser.parse_args()

print(args.train)
print(os.getcwd())
print(os.path.join(os.getcwd(), args.train))
FREQ_DIST_FILE = str(args.freq)
BI_FREQ_DIST_FILE = str(args.bifreq)
TRAIN_PROCESSED_FILE = str(args.train)
TEST_PROCESSED_FILE = 'dataset/test-processed.csv'
WORD_VECTORS = './dataset/numberbatch-en-17.06.txt'
dim = 300


def get_glove_vectors(vocab):
    print('Looking for pre-trained vectors')
    pretrained_vectors = {}
    found = 0
    with open(WORD_VECTORS, 'r', encoding='utf-8') as glove_file:
        for i, line in enumerate(glove_file):
            utils.write_status(i + 1, 0)
            tokens = line.split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                pretrained_vectors[word] = np.array(vector)
                found += 1
    print('\n')
    print('Found %d words on pre-trained word vectors' % found)
    return pretrained_vectors


def get_feature_vector(tweet):
    words = tweet.split()
    feature_vector = []
    for i in range(len(words) - 1):
        word = words[i]
        if vocab.get(word) is not None:
            feature_vector.append(vocab.get(word))
    if len(words) >= 1:
        if vocab.get(words[-1]) is not None:
            feature_vector.append(vocab.get(words[-1]))
    return feature_vector


def process_tweets(csv_file, test_file=False):
    tweets = []
    labels = []
    print('Generating feature vectors')
    with open(csv_file, 'r', encoding="utf-8") as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            if test_file:
                tweet_id, tweet = line.split(',')
            else:
                tweet_id, sentiment, tweet = line.split(',')
            feature_vector = get_feature_vector(tweet)
            if test_file:
                tweets.append(feature_vector)
            else:
                tweets.append(feature_vector)
                labels.append(int(sentiment))
            utils.write_status(i + 1, total)
    print('\n')
    return tweets, np.array(labels)


if __name__ == '__main__':
    train = args.train is not None
    np.random.seed(2017)
    vocab_size = 80000
    batch_size = 128
    max_length = 60
    filters = 600
    kernel_size = 5 # incremented by 1 from default
    vocab = utils.top_n_words(FREQ_DIST_FILE, vocab_size, shift=1)
    glove_vectors = get_glove_vectors(vocab)
    tweets, labels = process_tweets(TRAIN_PROCESSED_FILE, test_file=False)
    embedding_matrix = np.random.randn(vocab_size + 1, dim) * 0.01
    for word, i in vocab.items():
        glove_vector = glove_vectors.get(word)
        if glove_vector is not None:
            embedding_matrix[i] = glove_vector
    tweets = pad_sequences(tweets, maxlen=max_length, padding='post')
    shuffled_indices = np.random.permutation(tweets.shape[0])
    tweets = tweets[shuffled_indices]
    labels = labels[shuffled_indices]
    print("tweets vector test",tweets[0])
    if train:
        model = Sequential()
        model.add(Embedding(vocab_size + 1, dim, weights=[embedding_matrix],
                            input_length=max_length))
        model.add(Dropout(0.50))
        model.add(LSTM(256))
        model.add(Dense(128))
        model.add(Dropout(0.5))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        filepath = "./models/lstm-{epoch:02d}-{loss:0.3f}-{acc:0.3f}-{val_loss:0.3f}-{val_acc:0.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1,
                                     save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.4, verbose=1,
                                      patience=2, min_lr=0.000001)
        print(model.summary())
        model.fit(tweets, labels, batch_size=128, epochs=30,
                  validation_split=0.1, shuffle=True, callbacks=[checkpoint,
                                                                 reduce_lr])
    else:
        if args.model is not None:
            model = load_model(args.model)
        else:
            print('model arg not defined on --model')
        
        if args.test is not None:
            # gets dataset file from command args to predict on
            file_to_predict = args.train
        else:
            file_to_predict = TEST_PROCESSED_FILE
        print("Evaluating %s dataset" % file_to_predict)
        print(model.summary())
        test_tweets, _ = process_tweets(file_to_predict, test_file=True)
        test_tweets = pad_sequences(test_tweets, maxlen=max_length,
                                    padding='post')
        predictions = model.predict(test_tweets, batch_size=batch_size,
                                    verbose=1)
        results = zip(map(str, range(len(test_tweets))),
                      np.round(predictions[:, 0]).astype(int))
        utils.save_results_to_csv(results, 'analysis-%s' %
                                  file_to_predict)
