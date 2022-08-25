from training_seq2seq import TrainingSeq2Seq

import tensorflow as tf
import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split

import unicodedata
import re
import os
import io


def download_nmt():
    path_to_zip = tf.keras.utils.get_file(
    'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
    extract=True)

    path_to_file = os.path.dirname(path_to_zip)+"/spa-eng/spa.txt"
    return path_to_file

class NMTDataset:
    def __init__(self, problem_type='en-spa'):
        self.problem_type = 'en-spa'
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None


    def unicode_to_ascii(self, s):
        return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    ## Step 1 and Step 2
    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = '<start> ' + w + ' <end>'
        return w

    def create_dataset(self, path, num_examples):
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for faster training (set num_examples = len(lines) to use full data)
        lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
        word_pairs = [[self.preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

        return zip(*word_pairs)

    # Step 3 and Step 4
    def tokenize(self, lang):
        # lang = list of sentences in a language

        # print(len(lang), "example sentence: {}".format(lang[0]))
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<OOV>')
        lang_tokenizer.fit_on_texts(lang)

        ## tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts string (w1, w2, w3, ......, wn)
        ## to a list of correspoding integer ids of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang)

        ## tf.keras.preprocessing.sequence.pad_sequences takes argument a list of integer id sequences
        ## and pads the sequences to match the longest sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')

        return tensor, lang_tokenizer

    def load_dataset(self, path, num_examples=None):
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(path, num_examples)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE):
        file_path = download_nmt()
        input_tensor, target_tensor, self.inp_lang_tokenizer, self.targ_lang_tokenizer = self.load_dataset(file_path, num_examples)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return train_dataset, val_dataset, self.inp_lang_tokenizer, self.targ_lang_tokenizer






def main():
    BUFFER_SIZE = 32000
    BATCH_SIZE = 64
    # Let's limit the #training examples for faster training
    num_examples = 30000

    dataset_creator = NMTDataset('en-spa')
    train_dataset, val_dataset, inp_lang, targ_lang = dataset_creator.call(num_examples, BUFFER_SIZE, BATCH_SIZE)

    vocab_inp_size = len(inp_lang.word_index)+1
    vocab_tar_size = len(targ_lang.word_index)+1
    example_input_batch, example_target_batch = next(iter(train_dataset))
    max_length_input = example_input_batch.shape[1]
    max_length_output = example_target_batch.shape[1]

    embedding_dim = 256
    units = 1024
    steps_per_epoch = num_examples//BATCH_SIZE

    model = TrainingSeq2Seq(vocab_inp_size, vocab_tar_size, embedding_dim, units, BATCH_SIZE, max_length_input, max_length_output)
    model.train(train_dataset, steps_per_epoch, EPOCHS=10)

    def beam_evaluate_sentence(sentence, beam_width=3):
        sentence = dataset_creator.preprocess_sentence(sentence)

        inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
        inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                              maxlen=max_length_input,
                                                              padding='post')
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]
        result = ''

        enc_start_state = [tf.zeros((inference_batch_size, units)), tf.zeros((inference_batch_size,units))]
        enc_out, enc_h, enc_c = model.encoder(inputs, enc_start_state)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([inference_batch_size], targ_lang.word_index['<start>'])
        end_token = targ_lang.word_index['<end>']

        # From official documentation
        # NOTE If you are using the BeamSearchDecoder with a cell wrapped in AttentionWrapper, then you must ensure that:
        # The encoder output has been tiled to beam_width via tfa.seq2seq.tile_batch (NOT tf.tile).
        # The batch_size argument passed to the get_initial_state method of this wrapper is equal to true_batch_size * beam_width.
        # The initial state created with get_initial_state above contains a cell_state value containing properly tiled final state from the encoder.

        enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
        model.decoder.attention_mechanism.setup_memory(enc_out)
        print("beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :", enc_out.shape)

        # set decoder_inital_state which is an AttentionWrapperState considering beam_width
        hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
        decoder_initial_state = model.decoder.rnn_cell.get_initial_state(batch_size=beam_width*inference_batch_size, dtype=tf.float32)
        decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

        # Instantiate BeamSearchDecoder
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(model.decoder.rnn_cell,beam_width=beam_width, output_layer=model.decoder.fc)
        decoder_embedding_matrix = model.decoder.embedding.variables[0]

        # The BeamSearchDecoder object's call() function takes care of everything.
        outputs, final_state, sequence_lengths = decoder_instance(decoder_embedding_matrix, start_tokens=start_tokens, end_token=end_token, initial_state=decoder_initial_state)
        # outputs is tfa.seq2seq.FinalBeamSearchDecoderOutput object.
        # The final beam predictions are stored in outputs.predicted_id
        # outputs.beam_search_decoder_output is a tfa.seq2seq.BeamSearchDecoderOutput object which keep tracks of beam_scores and parent_ids while performing a beam decoding step
        # final_state = tfa.seq2seq.BeamSearchDecoderState object.
        # Sequence Length = [inference_batch_size, beam_width] details the maximum length of the beams that are generated


        # outputs.predicted_id.shape = (inference_batch_size, time_step_outputs, beam_width)
        # outputs.beam_search_decoder_output.scores.shape = (inference_batch_size, time_step_outputs, beam_width)
        # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0,2,1))
        beam_scores = tf.transpose(outputs.beam_search_decoder_output.scores, perm=(0,2,1))

        return final_outputs.numpy(), beam_scores.numpy()

    def beam_translate(sentence):
        result, beam_scores = beam_evaluate_sentence(sentence)
        print(result.shape, beam_scores.shape)
        for beam, score in zip(result, beam_scores):
            print(beam.shape, score.shape)
            output = targ_lang.sequences_to_texts(beam)
            output = [a[:a.index('<end>')] for a in output]
            beam_score = [a.sum() for a in score]
            print('Input: %s' % (sentence))
            for i in range(len(output)):
                print('{} Predicted translation: {}  {}'.format(i+1, output[i], beam_score[i]))

    print(beam_translate(u'hace mucho frio aqui.'))
    print(beam_translate(u'¿todavia estan en casa?'))


if __name__ == "__main__":
    main()