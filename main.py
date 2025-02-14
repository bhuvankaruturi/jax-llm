import os
from datasets import load_from_disk
import jax
import jax.numpy as jnp
import optax
from flax import nnx
import matplotlib.pyplot as plt

EMBEDDING_DIM = 512
VOCAB_SIZE = 256
FF_DIM = 1024
LEARNING_RATE = 0.005
MOMENTUM = 0.9
DEFAULT_BATCH_SIZE = 100

def get_training_dataset_size(dataset_size: int) -> int:
    return int(os.environ.get("TRAIN_DATASET_SIZE", dataset_size))

def get_batch_size() -> int:
    return int(os.environ.get("TRAIN_BATCH_SIZE", DEFAULT_BATCH_SIZE))

def truncate_text(text, max_length=1000):
    text['text'] = text['text'][:max_length]
    return text

def min_text_length(ds):
    return min(len(text['text']) for text in ds)

def prepare_dataset():
    my_ds = load_from_disk("oscar-en-10k")
    print(my_ds.shape)
    min_length = min_text_length(my_ds)
    print(f"Min length: {min_length}")
    ds = my_ds.train_test_split(test_size=0.1)
    ds['train'] = ds['train'].map(lambda x: truncate_text(x, max_length=min_length))
    ds['test'] = ds['test'].map(lambda x: truncate_text(x, max_length=min_length))
    print(ds['train'].shape)
    print(ds['test'].shape)
    print(ds['train'][0])
    ds.save_to_disk("oscar-en-10k-truncated")

def load_dataset():
    return load_from_disk("oscar-en-10k-truncated")

def to_ascii_code(text):
    text['token_ids'] = [ord(c) for c in text['text']]
    text['token_ids'] = [0 if c > 127 else c for c in text['token_ids']]
    return text

def from_ascii_to_text(codes):
    return ''.join([chr(c) for c in codes])

def fold_columns(dataset, max_column_length=256):
    num_rows = dataset.shape[0]
    num_cols = dataset.shape[1]
    num_full_cols = num_cols // max_column_length
    truncated_dataset = dataset[:, :num_full_cols * max_column_length]
    folded_dataset = truncated_dataset.reshape(num_rows * num_full_cols, max_column_length)
    return folded_dataset

def save_jax_array(array, filename):
    jnp.save(filename, array)

def load_jax_array(filename):
    return jnp.load(filename)

def to_inputs(dataset):
    zeroes = jnp.zeros(dataset.shape, dtype=jnp.int8)
    zeroes = zeroes.at[:, 1:].set(dataset[:, :-1])
    return zeroes

def to_one_hot(dataset):
    return jax.nn.one_hot(dataset, VOCAB_SIZE, dtype=jnp.int8)

def to_labels_from_one_hot(dataset):
    return jnp.argmax(dataset, axis=-1)

def calculate_loss(model, inputs, outputs):
    logits = model(inputs)
    loss = optax.softmax_cross_entropy(logits, outputs).mean()
    return loss, logits

def get_positional_embeddings(max_len, embedding_dim):
    pe = jnp.zeros((max_len, embedding_dim))
    position = jnp.arange(0, max_len, dtype=jnp.float32)[:, None]
    div_term = jnp.exp(jnp.arange(0, embedding_dim, 2) * (-jnp.log(10000.0) / embedding_dim))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    pe = nnx.Variable(pe[None, :, :])
    return pe

class Model(nnx.Module):
    def __init__(self, input_dim: int, embedding_dim: int, ff_dim: int, output_dim: int, rng=nnx.Rngs(42)):
        self.embedding = nnx.Embed(num_embeddings=input_dim, features=embedding_dim, rngs=rng, dtype=jnp.float32)
        self.positional_encodings = get_positional_embeddings(max_len=512, embedding_dim=embedding_dim)
        self.ff1 = nnx.Linear(in_features=embedding_dim, out_features=ff_dim, rngs=rng)
        self.ff2 = nnx.Linear(in_features=ff_dim, out_features=output_dim, rngs=rng)
    
    def __call__(self, x):
        x = self.embedding(x)
        x = x + self.positional_encodings[:, :x.shape[1]]
        x = nnx.relu(self.ff1(x))
        x = nnx.log_softmax(self.ff2(x))
        return x

@nnx.jit
def train_step(model, optimizer, metrics, inputs, outputs):
    labels = to_labels_from_one_hot(outputs)
    grad_fn = nnx.value_and_grad(calculate_loss, has_aux=True)
    (loss, logits), grads = grad_fn(model, inputs, outputs)
    metrics.update(loss=loss, logits=logits, labels=labels)
    optimizer.update(grads)

@nnx.jit
def eval_step(model, metrics, inputs, outputs):
    labels = to_labels_from_one_hot(outputs)
    loss, logits = calculate_loss(model, inputs, outputs)
    metrics.update(loss=loss, logits=logits, labels=labels)

def setup_jax_dataset():
    raw_ds = load_dataset()['train']
    print(raw_ds.shape)
    ascii_ds = raw_ds.map(to_ascii_code)
    print(ascii_ds)
    ascii_ds = jnp.array(ascii_ds[:]['token_ids'], dtype=jnp.int8)
    print(ascii_ds.shape)
    truncated_ds = fold_columns(ascii_ds, max_column_length=512)
    save_jax_array(truncated_ds, "oscar-en-10k-truncated-ascii.npy")

def main():
    model = Model(VOCAB_SIZE, EMBEDDING_DIM, FF_DIM, VOCAB_SIZE)
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, MOMENTUM))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'),
    )

    ds = load_jax_array("oscar-en-10k-truncated-ascii.npy")
    ds = ds[:get_training_dataset_size(ds.shape[0])]
    inputs = to_inputs(ds)
    outputs = to_one_hot(ds)
    losses = []
    accuracies = []
    BATCH_SIZE = get_batch_size()
    for i in range(0, inputs.shape[0], BATCH_SIZE):
        step = i // BATCH_SIZE
        train_step(model, optimizer, metrics, inputs[i:i+BATCH_SIZE], outputs[i:i+BATCH_SIZE])
        for metric, value in metrics.compute().items():
            if metric == "loss":
                losses.append(value)
            elif metric == "accuracy":
                accuracies.append(value)
        if step % 10 == 0:
            print(f"Step {step}, loss: {losses[-1]:.4f}, accuracy: {accuracies[-1]:.4f}")

    plt.plot(losses)
    plt.plot(accuracies)
    plt.legend(["loss", "accuracy"])
    plt.show()

if __name__ == "__main__":
    main()