

def predict_sentences(self, sentences):
    '''
    Analyze Some Sentences
    :sentences: list of sentences
    e.g.: sentences = ['this is veeeryyy bad!!', 'I don\'t think he will be happy abt this',
                        'YOU\'re a fool!', 'I\'m sooo happY!!!']
    Sentence: "this is veeeryyy bad!!" , yielded results (pos/neg): 0.04511/0.95489, prediction: neg
    Sentence: "I dont think he will be happy abt this" , yielded results (pos/neg): 0.05929/0.94071, prediction: neg
    Sentence: "YOUre such an incompetent fool!" , yielded results (pos/neg): 0.48503/0.51497, prediction: neg ***
    Sentence: "Im sooo happY!!!" , yielded results (pos/neg): 0.97455/0.02545, prediction: pos
    '''
    BATCH_SIZE = self.hparams['BATCH_SIZE']
    max_word_length = self.hparams['max_word_length']
    pred = self.prediction

    saver = tf.train.Saver()

    with tf.Session() as sess:
        print('Loading model %s...' % SAVE_PATH)
        saver.restore(sess, SAVE_PATH)
        print('Done!')

        # Add placebo value '0,' at the beginning of the sentences to
        # use the make_minibatch() method
        sentences = ['0,' + s for s in sentences]

        with open(TEST_SET, 'r') as f:
            reader = TextReader(file=f, max_word_length=max_word_length)
            reader.load_to_ram(BATCH_SIZE)
            reader.data[:len(sentences)] = sentences
            batch_x, batch_y = reader.make_minibatch(reader.data)

            p = sess.run([pred], feed_dict={self.X: batch_x, self.Y: batch_y})
            for i, s in enumerate(sentences):
                print(
                    'Sentence: %s , yielded results (pos/neg): %.5f/%.5f, prediction: %s' %
                    (s, p[0][i][0], p[0][i][1],
                     'pos' if max(p[0][i]) == p[0][i][0] else 'neg'))
        return p
