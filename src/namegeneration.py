import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, StringLookup
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from corpus import Corpus
from tokenization import tokenize_to_texts, detokenize_texts, SOS_TOKEN, EOS_TOKEN, OOV_TOKEN, NOTHING_TOKEN


class NamePredictor(tf.keras.Model):
    # recurrent_cell: can be literally any recurrent cell, and you can even hook it up to other things beforehand
    # for recurrent_cell, return_sequences = True and return_state = True
    # you cannot stack recurrent cells (for now)
    def __init__(self, recurrent_cell, vocab_size: int, embedding_dim: int):
        super().__init__(self)
        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.recurrent_cell = recurrent_cell
        self.dense = Dense(vocab_size, activation="linear")  # for softmax

    # states: []; the cell states
    # return_state: bool; return my cell states?
    def call(self, inputs, states=None, return_state=False, training=False):
        embedded_inputs = self.embedding(inputs, training=training)

        if states is None:
            states = self.recurrent_cell.get_initial_state(embedded_inputs)

        main_computation, states = self.recurrent_cell(embedded_inputs,
                                                       initial_state=states,
                                                       training=training)

        output_logits = self.dense(main_computation, training=training)

        if return_state:
            return output_logits, states
        else:
            return output_logits


class NameGenerator(tf.keras.Model):
    def __init__(self, recurrent_cell, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        
        self.recurrent_cell = recurrent_cell
        self.model = None

        self.vocab = []
        self.chars_to_ids = None
        self.ids_to_chars = None

        self.prediction_mask = None

    def preprocess_corpus(self, corpus_text_raw: str, embedding_dim: int, optimizer, metrics: []):
        # Tokenize the text
        corpus_texts = tokenize_to_texts(corpus_text_raw)  # str[]
        corpus_text_full: str = "".join(corpus_texts)

        # Create Vocabulary
        vocab = list(sorted(set(corpus_text_full)))  # list of all unique characters

        # Conversions between characters and indices
        self.chars_to_ids = StringLookup(vocabulary=vocab, output_mode="int",
                                         mask_token=None, oov_token=OOV_TOKEN,
                                         encoding="utf-8")

        self.vocab = self.chars_to_ids.get_vocabulary()  # Doing this so unknown tokens can be set too
        print(f"Vocabulary: {self.vocab}")

        self.ids_to_chars = StringLookup(vocabulary=self.vocab, invert=True,
                                         output_mode="int", mask_token=None, oov_token=OOV_TOKEN,
                                         encoding="utf-8")
        
        # Create and compile a predictor model
        self.model = NamePredictor(self.recurrent_cell, len(self.vocab), embedding_dim)
        self.model.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=metrics)
        
        # Create the training set
        corpus_texts_chars = tf.strings.unicode_split(corpus_texts, input_encoding="UTF-8")  # shape: (m, Tx)
        corpus = Corpus(corpus_texts, corpus_tensor=self.chars_to_ids(corpus_texts_chars))   # shape: (m, Tx)
        
        print(f"Corpus ID Tensor Shape: {corpus.corpus_tensor.shape}")
        #print(f"Corpus ID Tensor: {corpus.corpus_tensor}")
        print(f"Corpus ID tf.Dataset: {corpus.corpus_dataset}")

        # Create a mask for the generator to prevent the OOV_TOKEN from being generated
        skip_ids = self.chars_to_ids([OOV_TOKEN])[:, None]  # skip the unknown values! Don't predict them!!!
        sparse_mask = tf.SparseTensor(
            values=[-float('inf')]*len(skip_ids),  # -inf at each bad index
            indices=skip_ids,
            dense_shape=[len(self.vocab)])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

        return corpus

    # @params
    # shuffle_data: bool
    # batch_size: int; also applies to shuffling if shuffle_data == True, and -1 (default) will do the entire batch
    # the rest of the params are keras model.fit() ones
    def train(self, corpus_text_raw: str, optimizer, metrics=[], embedding_dim=50,
                    shuffle_data=True, batch_size=-1,
                    epochs=1, verbose="auto", callbacks=None, validation_split=0.0, validation_data=None,
                    class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None,
                    validation_steps=None, validation_freq=1):
        # Preprocess the text
        corpus: Corpus = self.preprocess_corpus(corpus_text_raw, embedding_dim, optimizer, metrics)
        dataset = corpus.corpus_dataset
        
        if batch_size == -1:
            batch_size = corpus.size()

        # Shuffle Data
        if shuffle_data:
            dataset = corpus.shuffle_and_batch_dataset(batch_size)
            print(f"Shuffled Dataset Object: {dataset}")
        else:
            dataset = corpus.batch_dataset(batch_size)
            print(f"Batched Dataset Object: {dataset}")
            
        Corpus.print_all(dataset)
        
        # Train the Model
        return self.model.fit(dataset,
                              batch_size=batch_size, epochs=epochs, verbose=verbose, callbacks=callbacks, validation_split=validation_split,
                              validation_data=validation_data, class_weight=class_weight, sample_weight=sample_weight,
                              initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps,
                              validation_freq=validation_freq)

    # inputs: Tensor (tf.constant(str[]); inputs are prev_char but vectorized (to generate in batches)
    # assume each input is an already tokenized string
    @tf.function
    def generate_one_timestep(self, inputs, states=None):
        inputs_chars = tf.strings.unicode_split(inputs, input_encoding="UTF-8")  # strings -> chars
        inputs_ids = self.chars_to_ids(inputs_chars).to_tensor()

        # predict the rest of the sentence
        # predicted_logits_all.shape is [string_batch_size, num_input_chars + 1, vocab_size]
        predicted_logits_all, states = self.model.call(inputs=inputs_ids, states=states, return_state=True)

        # Use last prediction because that's the new predicted character
        predicted_logits = predicted_logits_all[:, -1, :]  # shape is [string_batch_size, vocab_size]
        predicted_logits = predicted_logits / self.temperature

        # Prevent OOV token from being generated by applying prediction mask
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample softmax(predicted_logits) -> token IDs
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        predicted_chars = self.ids_to_chars(predicted_ids)  # char for each batch

        return predicted_chars, states

    # start_inputs: str[]; len(start_inputs) == batch_size
    def generate_raw(self, start_inputs=None, start_at_timestep=1, stop_at_timestep=-1):
        if start_inputs is None:
            start_inputs = [SOS_TOKEN]
        elif type(start_inputs) is int:
            start_inputs = [SOS_TOKEN] * start_inputs

        next_char = tf.constant(start_inputs)
        states = None
        result = [next_char]

        timestep = start_at_timestep
        while timestep != stop_at_timestep:
            #print(f"Generating Time Step {timestep}...")
            next_char, states = self.generate_one_timestep(next_char, states)
            
            result.append(next_char)
            timestep += 1

        return Result(np.array(result), timestep)  # np.array(result: Tensor1x1[][]).shape is (max_Tx, batch_size)

    def generate(self, start_inputs=None, start_at_timestep=1, stop_at_timestep=-1):
        result = self.generate_raw(start_inputs=start_inputs,
                                   start_at_timestep=start_at_timestep, stop_at_timestep=stop_at_timestep)
        
        print("Converting to string representation...")
        
        generated_strings = []
        for batch_chars_tensor in result.value.T:  # join chars for each batch
            generated_string = tf.strings.join(batch_chars_tensor.tolist()).numpy().decode("utf-8")
            
            try:
                generated_string = generated_string[0:generated_string.index(EOS_TOKEN)+1]  # clip to and keep EOS; it will be truncated afterwards during detokenization
            except:  # if eos token not found
                pass
            
            generated_strings.append(generated_string)
        
        generated_strings = detokenize_texts(generated_strings)  # detokenize them all

        return Result(np.array(generated_strings), result.last_timestep)


class Result:
    def __init__(self, value: np.ndarray, last_timestep: int):
        super().__init__()
        self.value = value
        self.last_timestep = last_timestep
