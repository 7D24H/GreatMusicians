import tensorflow as tf
import numpy as np
import random

class Generator(tf.keras.utils.Sequence) :
    def __init__(self, processed_dataset, batch_size, seq_len, val_split = 0, shuffle=True) :
        self.processed_dataset = processed_dataset
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.val_split = val_split
        if(self.val_split != 0):
            self.processed_dataset = random.choices(self.processed_dataset, k = int(self.val_split*len(self.processed_dataset)))
            self.batch_size = len(self.processed_dataset)
        self.on_epoch_end()
    
    def __len__(self) :
        return int(np.ceil(len(self.processed_dataset)/ self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle == True:
            np.random.shuffle(self.processed_dataset)
  
    def __getitem__(self, idx) :
        batch_x = np.empty((0, self.seq_len), float)
        batch_y = np.empty((0, self.seq_len), float)
        for i in range(self.batch_size):
            if(idx*self.batch_size + i == len(self.processed_dataset)-1):
                return batch_x, batch_y
            song = self.processed_dataset[idx*self.batch_size + i]
            start_idx = random.randint(0,len(song) - self.seq_len/2)
            seq = song[start_idx: start_idx + self.seq_len + 1]
            x = seq[:-1]
            y = seq[1:]
            if(len(y) < self.seq_len):
                sub = len(y)
                num_pad = self.seq_len - sub
                # 1 is the padding value
                x = np.append(x, [1]*num_pad, axis = 0)
                y = np.append(y, [1]*num_pad, axis = 0)
            
            batch_x = np.append(batch_x, [x], axis = 0)
            batch_y = np.append(batch_y, [y], axis = 0)
            
        return batch_x, batch_y

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, seq_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
            'maxlen': self.seq_len,
        })
        return config
    
    def positional_encoding(self, position, d_model):
        angle_rads = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :]//2)) / np.float32(d_model))
        angle_rads = np.arange(position)[:, np.newaxis] * angle_rads
        # sine function
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # cosine function
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)
        
    def call(self, x):
        seq_len = tf.shape(x)[-1]
        pos_encoding = self.positional_encoding(10000, self.embed_dim)
        x = self.embedding(x)
        return x + pos_encoding[:,:seq_len,:]

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=4):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.merge = tf.keras.layers.Dense(embed_dim)
        self.query_layer = tf.keras.layers.Dense(embed_dim)
        self.key_layer = tf.keras.layers.Dense(embed_dim)
        self.value_layer = tf.keras.layers.Dense(embed_dim)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config

    @staticmethod
    def attention_mask(n_dest, n_src, dtype):
        i = tf.range(n_dest)[:, None]
        j = tf.range(n_src)
        return tf.cast((i >= j - n_src + n_dest), dtype)

    def atten(self, query, key, value):
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        # scale the score
        scaled_score = tf.matmul(query, key, transpose_b=True) / tf.math.sqrt(dim_key)

        shape = tf.shape(scaled_score)
        dim_dest, dim_src = shape[2], shape[3]
        attention_mask = self.attention_mask(
            dim_dest, dim_src, scaled_score.dtype
        )
        attention_mask = tf.reshape(attention_mask, [1, 1, dim_dest, dim_src])
        scaled_score = scaled_score * attention_mask - 1e4 * (1 - attention_mask)

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def one_head(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.embed_dim // self.num_heads))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.one_head(self.query_layer(inputs), batch_size)
        key = self.one_head(self.key_layer(inputs), batch_size)
        value = self.one_head(self.value_layer(inputs), batch_size)
        attention, _ = self.atten(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        output = self.merge(tf.reshape(attention, (batch_size, -1, self.embed_dim)))
        return output

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, dense_units):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dense_units = dense_units
        
        self.atten = MultiHeadAttention(embed_dim, num_heads)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.forward_layers = tf.keras.Sequential(
            [tf.keras.layers.Dense(units = dense_units, activation="relu"), 
             tf.keras.layers.Dense(units = embed_dim),]
        )
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dense_units': self.dense_units,
        })
        return config

    def call(self, inputs):
        atten = self.dropout1(self.atten(inputs))
        temp = self.layernorm1(inputs + atten)
        return self.layernorm2(self.layernorm1(inputs + atten) + self.dropout2(self.forward_layers(temp)))