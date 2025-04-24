import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report

from ..models.backbone.fer_mobile_vit import FERMobileViTDAN


def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Classification report
    target_names = ['Negative', 'Neutral', 'Positive']
    report = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # Convert to DataFrame for better visualization
    df_report = pd.DataFrame(report).transpose()

    # Plot metrics
    plt.figure(figsize=(10, 6))
    df_report.iloc[:-3][['precision', 'recall', 'f1-score']].plot(kind='bar')
    plt.title('Classification Metrics by Class')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(r'C:\Users\GIGABYTE\PycharmProjects\FER\results\img\classification_metrics.png')
    plt.show()

    return df_report


def predict_image(model, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        prob, predicted = torch.max(probabilities, 1)

    # Map to emotion label
    emotions = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    predicted_emotion = emotions[predicted.item()]
    confidence = prob.item() * 100

    # Probabilities for each class
    probs = {emotions[i]: float(p) * 100 for i, p in enumerate(probabilities[0])}

    # Display result
    plt.figure(figsize=(8, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.title(f'Predicted: {predicted_emotion} ({confidence:.2f}%)')
    plt.axis('off')

    # Plot probability bars
    plt.subplot(2, 1, 2)
    bars = plt.bar(probs.keys(), probs.values(), color=['red', 'gray', 'green'])
    plt.ylabel('Probability (%)')
    plt.ylim(0, 100)

    # Add values on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(r'C:\Users\GIGABYTE\PycharmProjects\FER\results\img\prediction_results.csv')
    plt.show()

    return predicted_emotion, probs


# Example usage
def main():
    # Load trained model
    model = FERMobileViTDAN(num_classes=3)
    checkpoint = torch.load('../results/models/fer_mobilevit_dan_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on validation set
    df_metrics = evaluate_model(model, valid_loader)

    # Predict on single image
    image_path = r'/DATASET/AffectNet/test/1/image0000007.jpg'  # Replace with actual path
    emotion, probabilities = predict_image(model, image_path)
    print(f"Predicted emotion: {emotion}")
    print(f"Probabilities: {probabilities}")


if __name__ == "__main__":
    main()