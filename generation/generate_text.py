# ============================================================
# generate_text.py
# TEXT VAE + TF-IDF + LOGISTIC REGRESSION
# ============================================================

import os
import sys
import pickle

import torch
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(BASE_DIR)


# ============================================================
# IMPORT TEXT VAE
# ============================================================

from models.text.vae import TextVAE


# ============================================================
# PATHS
# ============================================================

# Trained Text VAE
CHECKPOINT_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "text_vae.pth"
)

# Original SMS dataset
# CHANGE THIS ONLY IF YOUR CSV HAS A DIFFERENT NAME.
DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "sms",
    "SMSSpamCollection"
)

# Vocabulary
VOCAB_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "sms",
    "processed",
    "vocab.pkl"
)

# Output
OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "generation",
    "generated"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

# Your trained checkpoint uses latent dimension 64
LATENT_DIM = 64

MAX_LENGTH = 50

# Number of final synthetic messages
NUM_SAMPLES = 100

# Generate extra candidates because some will be rejected
MAX_ATTEMPTS = NUM_SAMPLES * 15

# Sampling temperature
# 0.7 = safer
# 0.8 = balanced
# 1.0 = more random
TEMPERATURE = 0.8

# Minimum number of words
MIN_TOKENS = 5

# Repetition control
MAX_CONSECUTIVE_REPEAT = 2
MAX_WORD_REPEAT = 3
REPETITION_WINDOW = 5


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TEXT VAE + TF-IDF + LOGISTIC REGRESSION")
print("SYNTHETIC SMS GENERATION")
print("=" * 70)

print("\nDevice:", DEVICE)
print("Latent dimension:", LATENT_DIM)
print("Target synthetic samples:", NUM_SAMPLES)
print("Temperature:", TEMPERATURE)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")


if not os.path.exists(CHECKPOINT_PATH):

    raise FileNotFoundError(
        f"\nText VAE checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


if not os.path.exists(VOCAB_PATH):

    raise FileNotFoundError(
        f"\nVocabulary file not found:\n"
        f"{VOCAB_PATH}"
    )


if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nSMS dataset not found:\n"
        f"{DATASET_PATH}\n\n"
        f"Change DATASET_PATH in this file "
        f"to your actual CSV filename."
    )


print("✓ Text VAE checkpoint found")
print("✓ Vocabulary found")
print("✓ SMS dataset found")

# ============================================================
# LOAD ORIGINAL SMS SPAM COLLECTION
# ============================================================

print("\n" + "=" * 70)
print("LOADING ORIGINAL SMS DATASET")
print("=" * 70)

print("\nDataset being loaded:")
print(DATASET_PATH)

# SMSSpamCollection is TAB separated, NOT comma separated
df = pd.read_csv(
    DATASET_PATH,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="utf-8"
)

print("\nOriginal dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("\nClass distribution:")
print(df["label"].value_counts())
# ============================================================
# IDENTIFY LABEL AND TEXT COLUMNS
# ============================================================

# Your dataset was previously:
#
# ['label', 'text']
#
# But this also supports common names such as:
# v1 / v2

if "label" in df.columns and "text" in df.columns:

    label_column = "label"
    text_column = "text"

elif "v1" in df.columns and "v2" in df.columns:

    label_column = "v1"
    text_column = "v2"

else:

    raise ValueError(
        "\nCould not find label/text columns.\n"
        "Expected either:\n"
        "label, text\n"
        "or:\n"
        "v1, v2"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df = df[
    [label_column, text_column]
].copy()


df.columns = [
    "label",
    "text"
]


df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)


df["text"] = (
    df["text"]
    .astype(str)
    .str.strip()
)


# Remove empty rows

df = df[
    (df["text"] != "")
    &
    (df["label"].isin(["ham", "spam"]))
]


print("\nClean dataset shape:")
print(df.shape)


print("\nClass distribution:")
print(
    df["label"].value_counts()
)


# ============================================================
# TRAIN TF-IDF CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("TRAINING SPAM / HAM CLASSIFIER")
print("=" * 70)


print("\nCreating TF-IDF features...")


vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words=None,
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)


X = vectorizer.fit_transform(
    df["text"]
)


y = df["label"]


print(
    "TF-IDF feature matrix:",
    X.shape
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

print("\nTraining Logistic Regression...")


classifier = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)


classifier.fit(
    X,
    y
)


print("✓ Classifier trained successfully")


# ============================================================
# LOAD VOCABULARY
# ============================================================

print("\n" + "=" * 70)
print("LOADING TEXT VAE VOCABULARY")
print("=" * 70)


with open(
    VOCAB_PATH,
    "rb"
) as f:

    vocab = pickle.load(f)


# ============================================================
# VOCABULARY FORMAT
# ============================================================

if isinstance(vocab, dict):

    if "word_to_idx" in vocab:

        word_to_idx = vocab["word_to_idx"]

    elif "stoi" in vocab:

        word_to_idx = vocab["stoi"]

    else:

        word_to_idx = vocab

else:

    raise ValueError(
        "Unsupported vocabulary format."
    )


# ============================================================
# REVERSE VOCABULARY
# ============================================================

idx_to_word = {
    int(index): word
    for word, index in word_to_idx.items()
}


VOCAB_SIZE = len(
    word_to_idx
)


print(
    "Vocabulary size:",
    VOCAB_SIZE
)


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"


PAD_IDX = word_to_idx.get(
    PAD_TOKEN,
    0
)

START_IDX = word_to_idx.get(
    START_TOKEN,
    2
)

END_IDX = word_to_idx.get(
    END_TOKEN,
    3
)

UNK_IDX = word_to_idx.get(
    UNK_TOKEN,
    1
)


print("\nSpecial tokens:")

print("PAD   :", PAD_IDX)
print("START :", START_IDX)
print("END   :", END_IDX)
print("UNK   :", UNK_IDX)


# ============================================================
# CREATE TEXT VAE
# ============================================================

print("\n" + "=" * 70)
print("CREATING TEXT VAE")
print("=" * 70)


model = TextVAE(
    vocab_size=VOCAB_SIZE,
    embedding_dim=128,
    hidden_dim=256,
    latent_dim=LATENT_DIM
)


model = model.to(
    DEVICE
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading trained Text VAE...")


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model.eval()


print("✓ Text VAE loaded successfully")


# ============================================================
# REPETITION CHECK
# ============================================================

def is_repetitive(tokens):

    if len(tokens) < 2:

        return True


    # --------------------------------------------------------
    # Consecutive words
    # --------------------------------------------------------

    consecutive_count = 1


    for i in range(
        1,
        len(tokens)
    ):

        if tokens[i] == tokens[i - 1]:

            consecutive_count += 1

            if (
                consecutive_count
                > MAX_CONSECUTIVE_REPEAT
            ):

                return True

        else:

            consecutive_count = 1


    # --------------------------------------------------------
    # Word frequency
    # --------------------------------------------------------

    word_counts = {}


    for word in tokens:

        word_counts[word] = (
            word_counts.get(word, 0)
            + 1
        )


        if (
            word_counts[word]
            > MAX_WORD_REPEAT
        ):

            return True


    # --------------------------------------------------------
    # Repeated short phrases
    # --------------------------------------------------------

    if len(tokens) >= 6:

        for pattern_length in [1, 2, 3]:

            for i in range(
                len(tokens)
                - pattern_length * 2
                + 1
            ):

                first = tokens[
                    i:
                    i + pattern_length
                ]

                second = tokens[
                    i + pattern_length:
                    i + pattern_length * 2
                ]


                if first == second:

                    return True


    return False


# ============================================================
# TOKEN SAMPLING
# ============================================================

def sample_token(
    logits,
    previous_tokens
):

    logits = logits / TEMPERATURE


    # Prevent special tokens

    logits[:, PAD_IDX] = -float("inf")

    logits[:, START_IDX] = -float("inf")


    # --------------------------------------------------------
    # Reduce probability of recent tokens
    # --------------------------------------------------------

    recent_tokens = previous_tokens[
        -REPETITION_WINDOW:
    ]


    for token_id in recent_tokens:

        if token_id != END_IDX:

            logits[:, token_id] -= 2.0


    # --------------------------------------------------------
    # Strongly discourage immediate repetition
    # --------------------------------------------------------

    if len(previous_tokens) > 0:

        last_token = previous_tokens[-1]

        logits[:, last_token] -= 4.0


    # --------------------------------------------------------
    # Probability distribution
    # --------------------------------------------------------

    probabilities = torch.softmax(
        logits,
        dim=-1
    )


    # --------------------------------------------------------
    # Random sampling
    # --------------------------------------------------------

    next_token = torch.multinomial(
        probabilities,
        num_samples=1
    )


    return next_token


# ============================================================
# GENERATE ONE SMS
# ============================================================

def generate_text():

    # Random latent representation

    z = torch.randn(
        1,
        LATENT_DIM,
        device=DEVICE
    )


    generated_tokens = []

    generated_ids = []


    # Start token

    current_token = torch.tensor(
        [[START_IDX]],
        dtype=torch.long,
        device=DEVICE
    )


    hidden = None


    with torch.no_grad():

        for step in range(
            MAX_LENGTH
        ):

            # ------------------------------------------------
            # Decoder
            # ------------------------------------------------

            try:

                output, hidden = model.decoder(
                    current_token,
                    z,
                    hidden
                )

            except TypeError:

                try:

                    output = model.decoder(
                        current_token,
                        z
                    )

                except TypeError:

                    output = model.decoder(
                        current_token
                    )


            # ------------------------------------------------
            # Handle tuple output
            # ------------------------------------------------

            if isinstance(
                output,
                tuple
            ):

                output = output[0]


            # ------------------------------------------------
            # Last timestep
            # ------------------------------------------------

            logits = output[
                :,
                -1,
                :
            ]


            # ------------------------------------------------
            # Sample token
            # ------------------------------------------------

            next_token = sample_token(
                logits,
                generated_ids
            )


            token_id = next_token.item()


            # ------------------------------------------------
            # END TOKEN
            # ------------------------------------------------

            if token_id == END_IDX:

                if (
                    len(generated_tokens)
                    >= MIN_TOKENS
                ):

                    break

                else:

                    continue


            # ------------------------------------------------
            # PAD TOKEN
            # ------------------------------------------------

            if token_id == PAD_IDX:

                continue


            # ------------------------------------------------
            # Convert ID to word
            # ------------------------------------------------

            word = idx_to_word.get(
                token_id,
                UNK_TOKEN
            )


            # ------------------------------------------------
            # Ignore UNK
            # ------------------------------------------------

            if word == UNK_TOKEN:

                current_token = next_token

                continue


            # ------------------------------------------------
            # Ignore special tokens
            # ------------------------------------------------

            if word in [
                PAD_TOKEN,
                START_TOKEN,
                END_TOKEN
            ]:

                current_token = next_token

                continue


            # ------------------------------------------------
            # Add word
            # ------------------------------------------------

            generated_tokens.append(
                word
            )


            generated_ids.append(
                token_id
            )


            current_token = next_token


    # ========================================================
    # VALIDATION
    # ========================================================

    if (
        len(generated_tokens)
        < MIN_TOKENS
    ):

        return ""


    if is_repetitive(
        generated_tokens
    ):

        return ""


    text = " ".join(
        generated_tokens
    ).strip()


    if not text:

        return ""


    return text


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

print("\n" + "=" * 70)
print("GENERATING SYNTHETIC SMS")
print("=" * 70)


generated_texts = []

generated_labels = []

seen_texts = set()

attempts = 0

rejected = 0


while (
    len(generated_texts)
    < NUM_SAMPLES
    and attempts < MAX_ATTEMPTS
):

    attempts += 1


    text = generate_text()


    # --------------------------------------------------------
    # Reject empty
    # --------------------------------------------------------

    if not text:

        rejected += 1

        continue


    # --------------------------------------------------------
    # Reject duplicates
    # --------------------------------------------------------

    normalized_text = (
        text
        .lower()
        .strip()
    )


    if normalized_text in seen_texts:

        rejected += 1

        continue


    # --------------------------------------------------------
    # Classify generated SMS
    # --------------------------------------------------------

    text_features = vectorizer.transform(
        [text]
    )


    predicted_label = classifier.predict(
        text_features
    )[0]


    # --------------------------------------------------------
    # Store
    # --------------------------------------------------------

    seen_texts.add(
        normalized_text
    )


    generated_texts.append(
        text
    )


    generated_labels.append(
        predicted_label
    )


    print(
        f"{len(generated_texts):03d} | "
        f"{predicted_label.upper():4s} | "
        f"{text}"
    )


# ============================================================
# GENERATION CHECK
# ============================================================

if len(generated_texts) == 0:

    raise RuntimeError(
        "\nNo valid synthetic messages were generated.\n"
        "The Text VAE may require additional training."
    )


print("\n" + "-" * 70)

print(
    "Valid messages generated:",
    len(generated_texts)
)

print(
    "Generation attempts:",
    attempts
)

print(
    "Rejected messages:",
    rejected
)


# ============================================================
# CREATE FINAL DATAFRAME
# ============================================================

synthetic_df = pd.DataFrame(
    {
        "label": generated_labels,
        "text": generated_texts
    }
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("SYNTHETIC DATA CLASS DISTRIBUTION")
print("=" * 70)


print(
    synthetic_df["label"]
    .value_counts()
)


# ============================================================
# SAVE CSV
# ============================================================

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "generated_sms.csv"
)


synthetic_df.to_csv(
    CSV_PATH,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE TXT
# ============================================================

TXT_PATH = os.path.join(
    OUTPUT_DIR,
    "generated_sms.txt"
)


with open(
    TXT_PATH,
    "w",
    encoding="utf-8"
) as f:

    for _, row in synthetic_df.iterrows():

        f.write(
            f"{row['label']}\t"
            f"{row['text']}\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("SYNTHETIC SMS GENERATION COMPLETED")
print("=" * 70)


print(
    "\nOriginal dataset:",
    len(df)
)


print(
    "Synthetic dataset:",
    len(synthetic_df)
)


print(
    "\nOriginal class distribution:"
)

print(
    df["label"]
    .value_counts()
)


print(
    "\nSynthetic class distribution:"
)

print(
    synthetic_df["label"]
    .value_counts()
)


print(
    "\nCSV saved to:"
)

print(
    CSV_PATH
)


print(
    "\nTXT saved to:"
)

print(
    TXT_PATH
)


print("\nOutput directory:")

print(
    OUTPUT_DIR
)


print("\n" + "=" * 70)