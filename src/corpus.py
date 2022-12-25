import tensorflow as tf

def split_input_target(sequence_i):
    input_chars_ids = sequence_i[:-1]
    target_chars_ids = sequence_i[1:]
    
    print(f"Input: {input_chars_ids}")
    print(f"Target: {target_chars_ids}")
    
    return (input_chars_ids, target_chars_ids)

class Corpus:
    def __init__(self, corpus_texts, corpus_tensor):
        super().__init__()
        self.corpus_texts = corpus_texts
        self.corpus_tensor = corpus_tensor
        self.corpus_dataset = tf.data.Dataset.from_tensor_slices(self.corpus_tensor).map(split_input_target)

    def size(self):
        return len(self.corpus_texts)            
    
    def batch_dataset(self, batch_size: int):
        batched_dataset = self.corpus_dataset.padded_batch(batch_size, drop_remainder=True, padding_values=tf.cast(0, "int64"))  # TODO: change later?
        #batched_dataset = self.corpus_dataset.batch(batch_size, drop_remainder=True)
        #print(list(batched_dataset.as_numpy_iterator()))
        
        return batched_dataset
    
    def shuffle_and_batch_dataset(self, batch_size: int, BUFFER_SIZE=10000):
        shuffled_dataset = (self.corpus_dataset
                            .shuffle(BUFFER_SIZE)
                            .padded_batch(batch_size, drop_remainder=True, padding_values=tf.cast(0, "int64"))  # TODO: change later?
                            .prefetch(tf.data.experimental.AUTOTUNE))
        
        #print(list(shuffled_dataset.as_numpy_iterator()))

        return shuffled_dataset
    
    def print_all(dataset):
        for element in dataset.as_numpy_iterator():
            print(element)