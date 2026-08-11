import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "../dataset/sms/SMSSpamCollection"

OUTPUT_DIR = "../dataset/sms/processed"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

MAX_VOCAB_SIZE = 5000

MAX_SEQUENCE_LENGTH = 40

TEST_SIZE = 0.2

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 60)

print("SMS SPAM COLLECTION - TEXT PREPROCESSING")

print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "text"]
)


print(
    "Original dataset shape:",
    df.shape
)


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

print("\nChecking missing values...")

print(
    df.isnull().sum()
)


df = df.dropna(
    subset=["text"]
).reset_index(
    drop=True
)


print(
    "\nDataset after cleaning:",
    df.shape
)


# ============================================================
# NORMALIZE TEXT
# ============================================================

df["text"] = (
    df["text"]
    .astype(str)
    .str.lower()
    .str.strip()
)


# ============================================================
# TOKENIZATION
# ============================================================

print("\nTokenizing text...")

tokenized_texts = []

for text in df["text"]:

    tokens = text.split()

    tokenized_texts.append(
        tokens
    )


# ============================================================
# BUILD VOCABULARY
# ============================================================

print("\nBuilding vocabulary...")


counter = Counter()

for tokens in tokenized_texts:

    counter.update(
        tokens
    )


# Special tokens

PAD_TOKEN = "<PAD>"

UNK_TOKEN = "<UNK>"

START_TOKEN = "<START>"

END_TOKEN = "<END>"


vocab = {

    PAD_TOKEN: 0,

    UNK_TOKEN: 1,

    START_TOKEN: 2,

    END_TOKEN: 3
}


# Add most common words

most_common_words = counter.most_common(
    MAX_VOCAB_SIZE - len(vocab)
)


for word, count in most_common_words:

    if word not in vocab:

        vocab[word] = len(vocab)


print(
    "Vocabulary size:",
    len(vocab)
)


# ============================================================
# CREATE REVERSE VOCABULARY
# ============================================================

id_to_word = {
    index: word
    for word, index in vocab.items()
}


# ============================================================
# CONVERT TEXT TO TOKEN IDs
# ============================================================

print("\nConverting text to token IDs...")


sequences = []


for tokens in tokenized_texts:

    sequence = [
        vocab[START_TOKEN]
    ]


    for token in tokens:

        if token in vocab:

            sequence.append(
                vocab[token]
            )

        else:

            sequence.append(
                vocab[UNK_TOKEN]
            )


    sequence.append(
        vocab[END_TOKEN]
    )


    # Limit sequence length

    sequence = sequence[
        :MAX_SEQUENCE_LENGTH
    ]


    # Padding

    while len(sequence) < MAX_SEQUENCE_LENGTH:

        sequence.append(
            vocab[PAD_TOKEN]
        )


    sequences.append(
        sequence
    )


sequences = np.array(
    sequences,
    dtype=np.int64
)


print(
    "Tokenized data shape:",
    sequences.shape
)


# ============================================================
# ENCODE LABELS
# ============================================================

label_mapping = {

    "ham": 0,

    "spam": 1
}


labels = df["label"].map(
    label_mapping
).values


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    sequences,

    labels,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=labels
)


print("\nTraining data:", X_train.shape)

print("Testing data :", X_test.shape)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_train.npy"
    ),
    X_train
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "X_test.npy"
    ),
    X_test
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_train.npy"
    ),
    y_train
)


np.save(
    os.path.join(
        OUTPUT_DIR,
        "y_test.npy"
    ),
    y_test
)


# ============================================================
# SAVE VOCABULARY
# ============================================================

joblib.dump(
    vocab,
    os.path.join(
        OUTPUT_DIR,
        "vocab.pkl"
    )
)


joblib.dump(
    id_to_word,
    os.path.join(
        OUTPUT_DIR,
        "id_to_word.pkl"
    )
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "vocab_size": len(vocab),

    "max_sequence_length": MAX_SEQUENCE_LENGTH,

    "pad_token": PAD_TOKEN,

    "unk_token": UNK_TOKEN,

    "start_token": START_TOKEN,

    "end_token": END_TOKEN,

    "label_mapping": label_mapping,

    "num_samples": len(df)

}


joblib.dump(
    metadata,
    os.path.join(
        OUTPUT_DIR,
        "metadata.pkl"
    )
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)

print("TEXT PREPROCESSING COMPLETED")

print("=" * 60)

print("\nFinal information:")

print(
    "Original records       :",
    len(df)
)

print(
    "Vocabulary size        :",
    len(vocab)
)

print(
    "Maximum sequence length:",
    MAX_SEQUENCE_LENGTH
)

print(
    "Training samples       :",
    len(X_train)
)

print(
    "Testing samples        :",
    len(X_test)
)

print(
    "Training shape         :",
    X_train.shape
)

print(
    "Testing shape          :",
    X_test.shape
)

print("\nFiles created:")

print("X_train.npy")

print("X_test.npy")

print("y_train.npy")

print("y_test.npy")

print("vocab.pkl")

print("id_to_word.pkl")

print("metadata.pkl")

print("\nSaved to:")

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)