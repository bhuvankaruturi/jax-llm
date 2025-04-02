import os
from datasets import load_from_disk
import jax
import jax.numpy as jnp
import optax  # Optimization library for JAX
from flax import nnx  # Neural network library for JAX (NNX module system)
import matplotlib.pyplot as plt

# --- Configuration Constants ---
EMBEDDING_DIM = 512     # Dimension of the token embeddings
VOCAB_SIZE = 256        # Size of the vocabulary (ASCII characters 0-127, plus potential padding/special tokens up to 255)
FF_DIM = 1024           # Dimension of the feed-forward layer in the model
LEARNING_RATE = 0.005   # Learning rate for the optimizer
MOMENTUM = 0.9          # Momentum factor for the AdamW optimizer
DEFAULT_BATCH_SIZE = 100 # Default batch size if not specified by environment variable

# --- Helper Functions ---

def get_training_dataset_size(dataset_size: int) -> int:
    """
    Gets the desired training dataset size from the environment variable 'TRAIN_DATASET_SIZE'.
    If the environment variable is not set, it defaults to the provided dataset_size.

    Args:
        dataset_size: The full size of the dataset, used as a default.

    Returns:
        The desired training dataset size (potentially truncated).
    """
    return int(os.environ.get("TRAIN_DATASET_SIZE", dataset_size))

def get_batch_size() -> int:
    """
    Gets the batch size from the environment variable 'TRAIN_BATCH_SIZE'.
    If the environment variable is not set, it defaults to DEFAULT_BATCH_SIZE.

    Returns:
        The batch size for training.
    """
    return int(os.environ.get("TRAIN_BATCH_SIZE", DEFAULT_BATCH_SIZE))

# --- Dataset Preparation Functions ---

def truncate_text(text, max_length=1000):
    """
    Truncates the 'text' field within a dataset dictionary item to a maximum length.
    This is often used in a .map() operation on a Hugging Face dataset.

    Args:
        text (dict): A dictionary representing a single data sample, expected to have a 'text' key.
        max_length (int): The maximum length to truncate the text to.

    Returns:
        dict: The modified dictionary with the 'text' field truncated.
    """
    text['text'] = text['text'][:max_length]
    return text

def min_text_length(ds):
    """
    Calculates the minimum length of the 'text' field across all samples in a dataset.

    Args:
        ds (datasets.Dataset): A Hugging Face dataset object where each sample is a dict with a 'text' key.

    Returns:
        int: The minimum text length found in the dataset.
    """
    # Iterate through each sample (dictionary) in the dataset and find the minimum length of the 'text' value
    return min(len(text['text']) for text in ds)

def prepare_dataset():
    """
    Loads the 'oscar-en-10k' dataset, finds the minimum text length across all samples,
    splits it into training and testing sets, truncates all samples in both sets
    to this minimum length, and saves the processed dataset to disk.
    """
    # Load the dataset from disk
    my_ds = load_from_disk("oscar-en-10k")
    print(f"Original dataset shape: {my_ds.shape}")

    # Find the shortest text sample to ensure uniform length after truncation
    min_length = min_text_length(my_ds)
    print(f"Min length found: {min_length}")

    # Split the dataset into training (90%) and testing (10%) sets
    ds = my_ds.train_test_split(test_size=0.1)

    # Truncate all text samples in the training set to the minimum length
    ds['train'] = ds['train'].map(lambda x: truncate_text(x, max_length=min_length))
    # Truncate all text samples in the test set to the minimum length
    ds['test'] = ds['test'].map(lambda x: truncate_text(x, max_length=min_length))

    # Print shapes and the first training sample to verify
    print(f"Train dataset shape after truncation: {ds['train'].shape}")
    print(f"Test dataset shape after truncation: {ds['test'].shape}")
    print(f"First training sample after truncation: {ds['train'][0]}")

    # Save the processed (truncated) dataset back to disk
    ds.save_to_disk("oscar-en-10k-truncated")

def load_dataset():
    """
    Loads the pre-processed (truncated) dataset from the specified directory.

    Returns:
        datasets.DatasetDict: The loaded dataset dictionary containing 'train' and 'test' splits.
    """
    return load_from_disk("oscar-en-10k-truncated")

def to_ascii_code(text):
    """
    Converts the 'text' field of a dataset sample into a list of ASCII codes (integers).
    Characters outside the standard ASCII range (0-127) are mapped to 0.
    Adds a new 'token_ids' field to the sample dictionary.

    Args:
        text (dict): A dictionary representing a single data sample, expected to have a 'text' key.

    Returns:
        dict: The modified dictionary with the added 'token_ids' field.
    """
    # Convert each character to its ASCII ordinal value
    text['token_ids'] = [ord(c) for c in text['text']]
    # Map any character code > 127 to 0 (handle potential extended ASCII or Unicode)
    text['token_ids'] = [0 if c > 127 else c for c in text['token_ids']]
    return text

def from_ascii_to_text(codes):
    """
    Converts a sequence of ASCII codes (integers) back into a string.

    Args:
        codes (list or array): A sequence of integer ASCII codes.

    Returns:
        str: The resulting text string.
    """
    # Convert each integer code back to its corresponding character and join them
    return ''.join([chr(c) for c in codes])

def fold_columns(dataset, max_column_length=256):
    """
    Reshapes a 2D JAX array (dataset) into sequences of a fixed maximum length.
    It truncates the dataset to be divisible by max_column_length and then reshapes.

    Example:
        If dataset shape is (10, 1000) and max_column_length is 200:
        1. Truncate columns: (10, 1000) -> (10, 1000) # Already divisible
        2. Calculate num_full_cols: 1000 // 200 = 5
        3. Reshape: (10, 1000) -> (10 * 5, 200) = (50, 200)

    Args:
        dataset (jnp.ndarray): The 2D JAX array to reshape (num_samples, original_sequence_length).
        max_column_length (int): The desired fixed length for the sequences (columns in the output).

    Returns:
        jnp.ndarray: The reshaped JAX array with shape (new_num_samples, max_column_length).
    """
    num_rows = dataset.shape[0]
    num_cols = dataset.shape[1] # Original sequence length

    # Calculate how many full sequences of max_column_length fit into the original sequence length
    num_full_cols = num_cols // max_column_length

    # Truncate the dataset columns to ensure it's perfectly divisible by max_column_length
    truncated_dataset = dataset[:, :num_full_cols * max_column_length]

    # Reshape the dataset: multiply rows by the number of full sequences and set columns to max_column_length
    folded_dataset = truncated_dataset.reshape(num_rows * num_full_cols, max_column_length)
    return folded_dataset

def save_jax_array(array, filename):
    """
    Saves a JAX numpy array to a file using the .npy format.

    Args:
        array (jnp.ndarray): The JAX array to save.
        filename (str): The path and filename to save the array to (e.g., "data.npy").
    """
    jnp.save(filename, array)

def load_jax_array(filename):
    """
    Loads a JAX numpy array from a .npy file.

    Args:
        filename (str): The path and filename of the .npy file to load.

    Returns:
        jnp.ndarray: The loaded JAX array.
    """
    return jnp.load(filename)

# --- Data Transformation for Model Input ---

def to_inputs(dataset):
    """
    Creates input sequences for a language model by shifting the dataset sequences.
    The input at time step `t` should predict the token at time step `t`.
    So, the input sequence is the original sequence shifted right by one position,
    with a 0 (padding or start token) inserted at the beginning.
    Example: [A, B, C, D] -> Input: [0, A, B, C]

    Args:
        dataset (jnp.ndarray): A 2D JAX array where each row is a sequence of token IDs.
                                Shape: (batch_size, sequence_length).

    Returns:
        jnp.ndarray: The input sequences, shifted right with a leading zero.
                     Shape: (batch_size, sequence_length).
    """
    # Create an array of zeros with the same shape and dtype as the dataset
    zeroes = jnp.zeros(dataset.shape, dtype=jnp.int8)
    # Fill the array (starting from the second position) with the dataset's values (excluding the last one)
    # This effectively shifts the sequence one step to the right and pads the beginning with 0.
    zeroes = zeroes.at[:, 1:].set(dataset[:, :-1])
    return zeroes

def to_one_hot(dataset):
    """
    Converts integer token IDs into one-hot encoded vectors.

    Args:
        dataset (jnp.ndarray): A JAX array of integer token IDs. Shape can be (batch, seq_len) or similar.

    Returns:
        jnp.ndarray: The one-hot encoded representation of the dataset.
                     Shape: (*dataset.shape, VOCAB_SIZE).
    """
    # Use jax.nn.one_hot for efficient conversion
    return jax.nn.one_hot(dataset, VOCAB_SIZE, dtype=jnp.int8)

def to_labels_from_one_hot(dataset):
    """
    Converts one-hot encoded vectors back into integer labels (token IDs).
    This finds the index of the '1' in each one-hot vector.

    Args:
        dataset (jnp.ndarray): A JAX array of one-hot encoded vectors.
                               Shape: (*, VOCAB_SIZE).

    Returns:
        jnp.ndarray: The integer labels. Shape: (*).
    """
    # Find the index of the maximum value along the last axis (the vocabulary dimension)
    return jnp.argmax(dataset, axis=-1)

# --- Model Definition and Training Logic ---

def calculate_loss(model, inputs, outputs):
    """
    Calculates the softmax cross-entropy loss between the model's predictions (logits)
    and the true target outputs (one-hot encoded).

    Args:
        model (nnx.Module): The neural network model.
        inputs (jnp.ndarray): The input sequences for the model.
        outputs (jnp.ndarray): The target sequences (one-hot encoded).

    Returns:
        tuple: A tuple containing:
            - jnp.ndarray: The mean loss value (scalar).
            - jnp.ndarray: The raw logits predicted by the model.
    """
    # Get model predictions (logits) for the given inputs
    logits = model(inputs)
    # Calculate the softmax cross-entropy loss between logits and one-hot encoded targets
    # optax.softmax_cross_entropy expects logits and one-hot targets.
    loss = optax.softmax_cross_entropy(logits, outputs).mean() # Average loss across the batch and sequence
    return loss, logits

def get_positional_embeddings(max_len, embedding_dim):
    """
    Generates sinusoidal positional embeddings.

    These embeddings provide information about the position of tokens in a sequence.
    The formula uses sine and cosine functions at different frequencies.

    Args:
        max_len (int): The maximum sequence length for which to generate embeddings.
        embedding_dim (int): The dimension of the embeddings (must match token embedding dim).

    Returns:
        nnx.Variable: A JAX array wrapped in nnx.Variable containing the positional embeddings.
                      Shape: (1, max_len, embedding_dim). Wrapped for potential state management.
    """
    # Initialize a matrix of zeros for positional encodings
    pe = jnp.zeros((max_len, embedding_dim))
    # Create a column vector representing positions (0 to max_len-1)
    position = jnp.arange(0, max_len, dtype=jnp.float32)[:, None]
    # Calculate the division term for the frequencies (decreases exponentially)
    div_term = jnp.exp(jnp.arange(0, embedding_dim, 2) * (-jnp.log(10000.0) / embedding_dim))
    # Apply sine to even indices in the embedding dimension
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    # Apply cosine to odd indices in the embedding dimension
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    # Add a batch dimension (shape becomes (1, max_len, embedding_dim)) and wrap in nnx.Variable
    # Wrapping in Variable is typical in NNX for parameters/state that aren't updated by gradients directly.
    pe = nnx.Variable(pe[None, :, :])
    return pe

class Model(nnx.Module):
    """
    A simple sequence model using token embeddings, positional embeddings,
    and two feed-forward layers.
    """
    def __init__(self, input_dim: int, embedding_dim: int, ff_dim: int, output_dim: int, rng=nnx.Rngs(42)):
        """
        Initializes the model layers.

        Args:
            input_dim (int): The size of the input vocabulary (VOCAB_SIZE).
            embedding_dim (int): The dimension for token and positional embeddings.
            ff_dim (int): The dimension of the hidden feed-forward layer.
            output_dim (int): The size of the output vocabulary (should match input_dim for generation).
            rng (nnx.Rngs): PRNG key generator for layer initializations.
        """
        # Embedding layer to convert token IDs to dense vectors
        self.embedding = nnx.Embed(num_embeddings=input_dim, features=embedding_dim, rngs=rng, dtype=jnp.float32)
        # Precompute positional encodings (treated as fixed parameters/state)
        self.positional_encodings = get_positional_embeddings(max_len=512, embedding_dim=embedding_dim) # max_len could be a parameter
        # First feed-forward layer
        self.ff1 = nnx.Linear(in_features=embedding_dim, out_features=ff_dim, rngs=rng)
        # Second feed-forward layer (output layer)
        self.ff2 = nnx.Linear(in_features=ff_dim, out_features=output_dim, rngs=rng)

    def __call__(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (jnp.ndarray): Input sequences of token IDs. Shape: (batch_size, sequence_length).

        Returns:
            jnp.ndarray: Output logits. Shape: (batch_size, sequence_length, output_dim).
        """
        # 1. Convert token IDs to embeddings
        x = self.embedding(x) # Shape: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        # 2. Add positional embeddings
        # We need to slice positional encodings to match the input sequence length
        # .value accesses the JAX array inside the nnx.Variable
        x = x + self.positional_encodings.value[:, :x.shape[1]]
        # 3. Pass through first feed-forward layer with ReLU activation
        x = nnx.relu(self.ff1(x)) # Shape: (batch, seq_len, ff_dim)
        # 4. Pass through second feed-forward layer and apply log_softmax
        # log_softmax is often used for numerical stability with cross-entropy loss
        x = nnx.log_softmax(self.ff2(x)) # Shape: (batch, seq_len, output_dim)
        return x

@nnx.jit
def train_step(model: Model, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, inputs: jnp.ndarray, outputs: jnp.ndarray):
    """
    Performs a single training step: calculates loss, computes gradients,
    updates model parameters, and updates metrics. JIT-compiled for performance.

    Args:
        model (Model): The model being trained.
        optimizer (nnx.Optimizer): The optimizer managing model parameters.
        metrics (nnx.MultiMetric): Object to track metrics (loss, accuracy).
        inputs (jnp.ndarray): Batch of input sequences.
        outputs (jnp.ndarray): Batch of target sequences (one-hot encoded).
    """
    # Convert one-hot outputs back to integer labels for metric calculation (e.g., accuracy)
    labels = to_labels_from_one_hot(outputs)

    # Create a function that calculates the loss and gradients for the model
    # nnx.value_and_grad computes the function's value (loss, logits) and the gradient wrt the first arg (model)
    grad_fn = nnx.value_and_grad(calculate_loss, has_aux=True) # has_aux=True because calculate_loss returns (loss, logits)

    # Execute the grad_fn to get loss, auxiliary output (logits), and gradients
    (loss, logits), grads = grad_fn(model, inputs, outputs)

    # Update metrics with the results from this step
    metrics.update(loss=loss, logits=logits, labels=labels) # Use computed metrics for update

    # Update the model's parameters using the optimizer and calculated gradients
    optimizer.update(grads)

@nnx.jit
def eval_step(model: Model, metrics: nnx.MultiMetric, inputs: jnp.ndarray, outputs: jnp.ndarray):
    """
    Performs a single evaluation step: calculates loss and updates metrics.
    Does NOT compute gradients or update the model. JIT-compiled for performance.

    Args:
        model (Model): The model being evaluated.
        metrics (nnx.MultiMetric): Object to track metrics (loss, accuracy).
        inputs (jnp.ndarray): Batch of input sequences.
        outputs (jnp.ndarray): Batch of target sequences (one-hot encoded).
    """
    # Convert one-hot outputs back to integer labels for metric calculation
    labels = to_labels_from_one_hot(outputs)

    # Calculate loss and get logits from the model
    loss, logits = calculate_loss(model, inputs, outputs)

    # Update metrics with the results from this step
    metrics.update(loss=loss, logits=logits, labels=labels)

# --- Data Preparation Workflow ---

def setup_jax_dataset():
    """
    Loads the raw (truncated but text-based) dataset, converts text to ASCII codes,
    reshapes the data into fixed-length sequences using fold_columns, and saves
    the final JAX array ready for training.
    """
    # Load the dataset saved by prepare_dataset()
    raw_ds = load_dataset()['train'] # Use the training split
    print(f"Loaded truncated text dataset shape: {raw_ds.shape}")

    # Convert the 'text' field to 'token_ids' (ASCII codes)
    ascii_ds = raw_ds.map(to_ascii_code, remove_columns=['text', 'id']) # Remove original columns after mapping
    print(f"Dataset after ASCII conversion: {ascii_ds}")

    # Convert the 'token_ids' column of the Hugging Face dataset to a single JAX array
    ascii_ds = jnp.array(ascii_ds[:]['token_ids'], dtype=jnp.int8)
    print(f"Shape after converting to JAX array: {ascii_ds.shape}") # Shape: (num_samples, original_seq_length)

    # Reshape the data into fixed-length sequences (e.g., 512)
    # This increases the number of samples and makes sequence length uniform.
    truncated_ds = fold_columns(ascii_ds, max_column_length=512)
    print(f"Shape after folding columns: {truncated_ds.shape}") # Shape: (new_num_samples, 512)

    # Save the final processed JAX array
    save_jax_array(truncated_ds, "oscar-en-10k-truncated-ascii.npy")
    print("Saved final JAX array for training.")


# --- Main Training Execution ---

def main():
    """
    Main function to set up the model, optimizer, data, and run the training loop.
    Plots loss and accuracy after training.
    """
    # --- Setup ---
    # Initialize the model
    model = Model(VOCAB_SIZE, EMBEDDING_DIM, FF_DIM, VOCAB_SIZE)
    # Initialize the optimizer (AdamW) linked to the model's parameters
    optimizer = nnx.Optimizer(model, optax.adamw(LEARNING_RATE, MOMENTUM))
    # Setup metrics tracking (accuracy and average loss)
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(),
        loss=nnx.metrics.Average('loss'), # Track the average of the 'loss' values passed to update()
    )

    # --- Data Loading and Preparation ---
    # Load the preprocessed JAX array dataset
    ds = load_jax_array("oscar-en-10k-truncated-ascii.npy")
    # Optionally truncate the dataset based on environment variable or use full size
    ds = ds[:get_training_dataset_size(ds.shape[0])]
    print(f"Using dataset size: {ds.shape[0]}")

    # Prepare inputs (shifted sequences) and outputs (one-hot encoded targets)
    inputs = to_inputs(ds)
    outputs = to_one_hot(ds)

    # Lists to store metrics history for plotting
    losses = []
    accuracies = []

    # Get batch size from environment or default
    BATCH_SIZE = get_batch_size()
    print(f"Using batch size: {BATCH_SIZE}")

    # --- Training Loop ---
    print("Starting training...")
    # Iterate over the dataset in batches
    for i in range(0, inputs.shape[0], BATCH_SIZE):
        step = i // BATCH_SIZE # Calculate current step number
        # Get the current batch of inputs and outputs
        batch_inputs = inputs[i:min(i + BATCH_SIZE, inputs.shape[0])]
        batch_outputs = outputs[i:min(i + BATCH_SIZE, outputs.shape[0])]

        # Perform a training step
        train_step(model, optimizer, metrics, batch_inputs, batch_outputs)

        # --- Metric Logging (Optional: log every step or periodically) ---
        # Compute the metrics accumulated so far
        current_metrics = metrics.compute()
        # Store the computed metrics
        for metric_name, value in current_metrics.items():
            if metric_name == "loss":
                losses.append(value.item()) # .item() converts JAX scalar to Python float
            elif metric_name == "accuracy":
                accuracies.append(value.item())

        # Reset metrics for the next accumulation period (e.g., next step or next epoch)
        # Important: If you want per-step metrics, reset here. If epoch metrics, reset after epoch.
        # Here we log *cumulative* average loss/accuracy up to the current step if not reset.
        # For per-step logging, reset is needed: metrics.reset()

        # Print progress periodically
        if step % 10 == 0: # Log every 10 steps
            # Print the latest computed metrics
            # Note: If metrics are not reset each step, loss/accuracy represent the average *so far*
            print(f"Step {step}, Avg Loss: {losses[-1]:.4f}, Avg Accuracy: {accuracies[-1]:.4f}")
            # If logging per-step metrics (with reset), compute before reset:
            # step_metrics = metrics.compute()
            # print(f"Step {step}, Loss: {step_metrics['loss']:.4f}, Accuracy: {step_metrics['accuracy']:.4f}")
            # metrics.reset() # Reset here if logging per-step

    print("Training finished.")

    # --- Plotting Results ---
    plt.figure(figsize=(10, 5)) # Create a figure for the plot
    plt.plot(losses, label="Average Loss")      # Plot the loss history
    plt.plot(accuracies, label="Average Accuracy") # Plot the accuracy history
    plt.xlabel("Training Step")
    plt.ylabel("Metric Value")
    plt.title("Training Loss and Accuracy")
    plt.legend() # Show the legend (labels for lines)
    plt.grid(True)
    plt.show() # Display the plot

if __name__ == "__main__":
    # --- Optional: Pre-run data setup if the JAX array doesn't exist ---
    # Check if the final data file exists, if not, run the setup
    if not os.path.exists("oscar-en-10k-truncated-ascii.npy"):
         print("Processed JAX dataset not found. Running setup_jax_dataset()...")
         # Check if the intermediate truncated dataset exists
         if not os.path.exists("oscar-en-10k-truncated"):
             print("Truncated dataset not found. Running prepare_dataset()...")
             prepare_dataset() # Create the truncated text dataset first
             print("prepare_dataset() finished.")
         setup_jax_dataset() # Convert text to ASCII and save as JAX array
         print("setup_jax_dataset() finished.")
    else:
        print("Found existing processed JAX dataset.")

    # Run the main training function
    main()