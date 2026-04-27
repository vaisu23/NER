import kagglehub

# Download latest version
path = kagglehub.dataset_download("juliangarratt/conll2003-dataset")

print("Path to dataset files:", path)


def parse_conll(filepath):
    sentences = []
    current_words = []
    current_labels = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()

            # 1. Check for sentence boundaries or document starts
            if not line or line.startswith("-DOCSTART-"):
                if current_words:
                    sentences.append({
                        "sentence": " ".join(current_words),
                        "labels": current_labels
                    })
                    current_words = []
                    current_labels = []
                continue

            # 2. Split the columns (Token, POS, Chunk, NER)
            parts = line.split()
            if len(parts) < 4:
                continue

            word = parts[0]

            ner_tag = parts[3] # This is the B-ORG, B-PER, etc.

            current_words.append(word)
            current_labels.append(ner_tag)

    return sentences


data = parse_conll('path to your  dataset')

# Look at the first processed sentence
for  i in range(10):
  print(data[i])


import json

def format_to_json(sentence_words, labels):
    entities = {"ORG": [], "PER": [], "LOC": [], "MISC": []}

    for word, label in zip(sentence_words, labels):  # ✅ no .split()
        if label.startswith("B-"):
            tag = label.split("-")[1]
            entities[tag].append(word)
        elif label.startswith("I-"):
            tag = label.split("-")[1]
            if entities[tag]:  # safety check
                entities[tag][-1] += " " + word

    output_json = {k: v for k, v in entities.items() if v}

    return {
        "instruction": "Extract the entities and return json",
        "input": " ".join(sentence_words),  # convert back to sentence
        "output": json.dumps(output_json)
    }
dataset=[]
for i in  range(len(data)):
  # print(i)
  temp= format_to_json(data[i]['sentence'].split(),  data[i]['labels'])
  dataset.append(temp)

  import json

file_path = "path to your destined location"

with open(file_path, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")