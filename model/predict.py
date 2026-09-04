import torch

from model.dataset import IntentDataset
from model.vectorizer import BagOfWordsVectorizer
from model.network import IntentClassifier


# Load dataset information
dataset = IntentDataset()
vectorizer = BagOfWordsVectorizer(dataset)

# Load trained model
checkpoint = torch.load(
    "model/intent_model.pth",
    weights_only=True
)

model = IntentClassifier(
    input_size=checkpoint["input_size"],
    hidden_size=checkpoint["hidden_size"],
    output_size=checkpoint["output_size"]
)

model.load_state_dict(checkpoint["model_state"])
model.eval()


def predict(text):
    tokens = dataset.tokenize(text)
    vector = vectorizer.vectorize(tokens)

    with torch.no_grad():
        output = model(vector)
        probabilities = torch.softmax(output, dim=0)

        predicted_index = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_index].item()

    intent = dataset.intents[predicted_index]

    return intent, confidence


tests = [
    "hello there",
    "please remember that I like coding",
    "what do you remember about me",
    "what is 25 plus 17",
    "tell me a joke"
]


for text in tests:
    intent, confidence = predict(text)

    print(
        f"{text} -> {intent} "
        f"({confidence * 100:.2f}%)"
    )