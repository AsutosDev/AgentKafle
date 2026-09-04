import torch
import torch.nn as nn
import torch.optim as optim

from model.dataset import IntentDataset
from model.vectorizer import BagOfWordsVectorizer
from model.network import IntentClassifier


# Load dataset
dataset = IntentDataset()

# Convert text into numerical vectors
vectorizer = BagOfWordsVectorizer(dataset)
X, y = vectorizer.prepare_data(dataset)

# Create model
model = IntentClassifier(
    input_size=len(dataset.words),
    hidden_size=64,
    output_size=len(dataset.intents)
)

# Training configuration
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500

# Train
for epoch in range(epochs):
    predictions = model(X)

    loss = loss_function(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.item():.4f}")


# Check training accuracy
with torch.no_grad():
    predictions = model(X)
    predicted_classes = torch.argmax(predictions, dim=1)

    accuracy = (predicted_classes == y).float().mean()

print("\nTraining complete.")
print(f"Training accuracy: {accuracy.item() * 100:.2f}%")


# Save trained model
torch.save(
    {
        "model_state": model.state_dict(),
        "words": dataset.words,
        "intents": dataset.intents,
        "input_size": len(dataset.words),
        "hidden_size": 64,
        "output_size": len(dataset.intents),
    },
    "model/intent_model.pth"
)

print("Model saved to model/intent_model.pth")