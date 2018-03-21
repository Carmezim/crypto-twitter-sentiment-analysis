import numpy as np
import sys
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import LSTM
import utils
from keras.preprocessing.sequence import pad_sequences

# Performs classification using LSTM network.

FREQ_DIST_FILE = './dataset/train-processed-freqdist.pkl'
BI_FREQ_DIST_FILE = './dataset/train-processed-freqdist-bi.pkl'
TRAIN_PROCESSED_FILE = './dataset/train-processed.csv'
TEST_PROCESSED_FILE = './dataset/XLM-tweets-processed.csv'
GLOVE_FILE = './dataset/numberbatch-en-17.06.txt'
dim = 200


def get_glove_vectors(vocab):
    print('Looking for GLOVE vectors')
    glove_vectors = {}
    found = 0
    with open(GLOVE_FILE, 'r', encoding='utf-8') as glove_file:
        for i, line in enumerate(glove_file):
            utils.write_status(i + 1, 0)
            tokens = line.split()
            word = tokens[0]
            if vocab.get(word):
                vector = [float(e) for e in tokens[1:]]
                glove_vectors[word] = np.array(vector)
                found += 1
    print('\n')
    print('Found %d words in GLOVE' % found)
    return glove_vectors


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
    train = len(sys.argv) == 1
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
        filepath = "./models/lstm-{epoch:02d}-{loss:0.3f}-{acc:0.3f}-\
            {val_loss:0.3f}-{val_acc:0.3f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor="loss", verbose=1,
                                     save_best_only=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, verbose=1,
                                      patience=3, min_lr=0.000001)
        print(model.summary())
        model.fit(tweets, labels, batch_size=128, epochs=30,
                  validation_split=0.1, shuffle=True, callbacks=[checkpoint,
                                                                 reduce_lr])
    else:
        model = load_model(sys.argv[1])
        if len(sys.argv) == 3:
            # gets dataset file from command args to predict on
            data_dir = sys.argv[2].split('/')
            file_to_predict = data_dir[-1]
        else:
            file_to_predict = TEST_PROCESSED_FILE
        print("Evaluating %s dataset" % file_to_predict)
        print(model.summary())
        test_tweets, _ = process_tweets("/".join(data_dir), test_file=True)
        test_tweets = pad_sequences(test_tweets, maxlen=max_length,
                                    padding='post')
        predictions = model.predict(test_tweets, batch_size=batch_size,
                                    verbose=1)
        results = zip(map(str, range(len(test_tweets))),
                      np.round(predictions[:, 0]).astype(int))
        utils.save_results_to_csv(results, 'analysis-%s' %
                                  file_to_predict)
