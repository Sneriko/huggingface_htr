from transformers import AutoTokenizer

# Load the tokenizer
#tokenizer = AutoTokenizer.from_pretrained("microsoft/trocr-base-handwritten")

model_path = "Riksarkivet/bert-base-cased-swe-historical"
tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir='/leonardo/home/userexternal/elenas00/projects/huggingface_htr/models/hub')


# Load the dict file
with open("/leonardo/home/userexternal/elenas00/projects/huggingface_htr/data/dict_satrn.txt", "r") as file:
    chars = file.read().splitlines()

# Check if all characters exist in the tokenizer's vocabulary
missing_chars = [char for char in chars if char not in tokenizer.get_vocab()]


if missing_chars:
    print(f"These characters are not in the vocabulary: {missing_chars}")
else:
    print("All characters in the dict file exist in the vocabulary.")