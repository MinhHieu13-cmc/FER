import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .backbone.fer_mobile_vit import FERMobileViTDAN  # Sửa lại
from data.dataloader import train_transform, valid_transform  # Sửa lại


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def forward(self, x):
        # Average predictions from all models
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


def k_fold_cross_validation(df, model_class, k=5, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup KFold
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    # Results tracking
    fold_results = []
    best_models = []

    # Training parameters
    criterion = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
        print(f"Fold {fold + 1}/{k}")

        # Split data
        train_data = df.iloc[train_idx].reset_index(drop=True)
        val_data = df.iloc[val_idx].reset_index(drop=True)

        # Create datasets and dataloaders
        train_dataset = FERDataset(train_data, transform=train_transform)
        val_dataset = FERDataset(val_data, transform=valid_transform)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

        # Initialize model
        model = model_class(num_classes=3).to(device)

        # Optimizer
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=0.0001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=0.1, patience=3, verbose=True)

        # Training
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_model_state = None

        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_preds = []
            train_labels = []

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                train_loss += loss.item() * inputs.size(0)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            train_loss = train_loss / len(train_loader.dataset)
            train_acc = accuracy_score(train_labels, train_preds)
            train_f1 = f1_score(train_labels, train_preds, average='weighted')

            # Validate
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    val_loss += loss.item() * inputs.size(0)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = accuracy_score(val_labels, val_preds)
            val_f1 = f1_score(val_labels, val_preds, average='weighted')

            # Update scheduler
            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{epochs} - "
                  f"Train: Loss {train_loss:.4f}, Acc {train_acc:.4f}, F1 {train_f1:.4f} | "
                  f"Val: Loss {val_loss:.4f}, Acc {val_acc:.4f}, F1 {val_f1:.4f}")

            # Save best model
            if val_f1 > best_val_f1:
                best_val_acc = val_acc
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()

        # Save results and best model for this fold
        fold_results.append({
            'fold': fold + 1,
            'val_acc': best_val_acc,
            'val_f1': best_val_f1
        })

        # Create a new model with the best weights
        best_model = model_class(num_classes=3).to(device)
        best_model.load_state_dict(best_model_state)
        best_models.append(best_model)

        # Save fold model
        torch.save(best_model_state, f'best_model_fold_{fold + 1}.pth')

    # Create and save ensemble model
    ensemble = EnsembleModel(best_models)

    # Evaluate ensemble on validation data
    ensemble_preds = []
    ensemble_labels = []

    # Create a complete validation dataset from all folds
    full_dataset = FERDataset(df, transform=valid_transform)
    full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False, num_workers=4)

    ensemble.eval()
    with torch.no_grad():
        for inputs, labels in full_loader:
            inputs = inputs.to(device)

            outputs = ensemble(inputs)
            _, preds = torch.max(outputs, 1)

            ensemble_preds.extend(preds.cpu().numpy())
            ensemble_labels.extend(labels.numpy())

    ensemble_acc = accuracy_score(ensemble_labels, ensemble_preds)
    ensemble_f1 = f1_score(ensemble_labels, ensemble_preds, average='weighted')

    print(f"\nEnsemble Results - Accuracy: {ensemble_acc:.4f}, F1 Score: {ensemble_f1:.4f}")

    # Summarize fold results
    df_results = pd.DataFrame(fold_results)
    print("\nFold Results:")
    print(df_results)

    # Plot fold results
    plt.figure(figsize=(10, 6))

    plt.subplot(1, 2, 2)
    sns.barplot(x='fold', y='val_f1', data=df_results)
    plt.axhline(y=ensemble_f1, color='r', linestyle='--', label=f'Ensemble: {ensemble_f1:.4f}')
    plt.title('Validation F1 Score by Fold')
    plt.legend()

    plt.tight_layout()
    plt.savefig('cross_validation_results.png')
    plt.show()

    # Return ensemble model and results
    return ensemble, df_results


# Custom FERDataset for CSV input
class FERDataset(Dataset):
    def __init__(self, data_df, transform=None):
        """
        Args:
            data_df: Pandas DataFrame with columns for image path and emotion
            transform: Optional transform to be applied on a sample
        """
        self.data = data_df
        self.transform = transform
        self.class_map = {'Negative': 0, 'Neutral': 1, 'Positive': 2}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.data.iloc[idx, 0]
        emotion = self.data.iloc[idx, 1]

        # Load image
        image = Image.open(img_path).convert('RGB')

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Convert label to numeric
        label = self.class_map[emotion]

        return image, label


def predict_with_ensemble(ensemble_model, image_path):
    """
    Make prediction with ensemble model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model = ensemble_model.to(device)
    ensemble_model.eval()

    # Load and transform image
    transform = transforms.Compose([
        transforms.Resize((260, 260)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Get predictions from individual models and ensemble
    individual_predictions = []
    individual_confidences = []

    # Get ensemble prediction
    with torch.no_grad():
        # Get predictions from each model
        for i, model in enumerate(ensemble_model.models):
            model.eval()
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)
            individual_predictions.append(pred.item())
            individual_confidences.append(conf.item())

        # Get ensemble prediction
        ensemble_outputs = ensemble_model(image_tensor)
        ensemble_probs = torch.nn.functional.softmax(ensemble_outputs, dim=1)
        ensemble_conf, ensemble_pred = torch.max(ensemble_probs, 1)

    # Map predictions to emotion labels
    emotions = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

    # Ensemble prediction results
    ensemble_emotion = emotions[ensemble_pred.item()]
    ensemble_confidence = ensemble_conf.item() * 100

    # Individual model predictions
    model_results = []
    for i in range(len(ensemble_model.models)):
        model_results.append({
            'model': f'Model {i + 1}',
            'prediction': emotions[individual_predictions[i]],
            'confidence': individual_confidences[i] * 100
        })

    # Get probabilities for each class from ensemble
    class_probs = {emotions[i]: float(p) * 100 for i, p in enumerate(ensemble_probs[0])}

    # Visualization
    plt.figure(figsize=(15, 10))

    # Display image with prediction
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title(f'Ensemble Prediction: {ensemble_emotion} ({ensemble_confidence:.2f}%)')
    plt.axis('off')

    # Display individual model predictions
    plt.subplot(2, 2, 2)
    model_names = [f'Model {i + 1}' for i in range(len(ensemble_model.models))]
    model_confs = [model_results[i]['confidence'] for i in range(len(model_results))]
    model_colors = ['blue' if model_results[i]['prediction'] == ensemble_emotion else 'red'
                    for i in range(len(model_results))]

    bars = plt.bar(model_names, model_confs, color=model_colors)
    plt.title('Individual Model Confidences')
    plt.ylabel('Confidence (%)')
    plt.ylim(0, 100)

    # Add values and predictions on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f"{model_results[i]['prediction']}\n{height:.1f}%",
                 ha='center', va='bottom')

    # Display ensemble probabilities for each class
    plt.subplot(2, 2, 3)
    class_bars = plt.bar(class_probs.keys(), class_probs.values(),
                         color=['red', 'gray', 'green'])
    plt.title('Ensemble Class Probabilities')
    plt.ylabel('Probability (%)')
    plt.ylim(0, 100)

    # Add values on top of bars
    for bar in class_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'{height:.1f}%', ha='center', va='bottom')

    # Voting visualization
    plt.subplot(2, 2, 4)
    vote_counts = {}
    for emotion in emotions.values():
        vote_counts[emotion] = sum(1 for result in model_results if result['prediction'] == emotion)

    vote_bars = plt.bar(vote_counts.keys(), vote_counts.values(),
                        color=['red', 'gray', 'green'])
    plt.title('Model Votes')
    plt.ylabel('Number of Votes')
    plt.ylim(0, len(ensemble_model.models) + 0.5)

    # Add values on top of bars
    for bar in vote_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('ensemble_prediction.png')
    plt.show()

    return {
        'ensemble_prediction': ensemble_emotion,
        'ensemble_confidence': ensemble_confidence,
        'class_probabilities': class_probs,
        'model_results': model_results
    }


def run_cross_validation_and_ensemble():
    # Load all data into a single DataFrame
    df = pd.read_csv(r'/DATASET/Data/training.csv')

    # Run cross-validation and get ensemble model
    ensemble_model, cv_results = k_fold_cross_validation(df, FERMobileViTDAN, k=5, epochs=20)

    # Save ensemble model
    torch.save({
        'models': [model.state_dict() for model in ensemble_model.models],
        'cv_results': cv_results
    }, 'ensemble_model.pth')

    # Evaluate on validation set
    val_df = pd.read_csv(r'/DATASET/Valid/valid.csv')
    val_dataset = FERDataset(val_df, transform=valid_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensemble_model = ensemble_model.to(device)
    ensemble_model.eval()

    val_preds = []
    val_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)

            outputs = ensemble_model(inputs)
            _, preds = torch.max(outputs, 1)

            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.numpy())

    # Calculate metrics
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f"\nEnsemble Validation Results:")
    print(f"Accuracy: {val_acc:.4f}")
    print(f"F1 Score: {val_f1:.4f}")

    # Generate classification report
    from sklearn.metrics import classification_report
    target_names = ['Negative', 'Neutral', 'Positive']
    report = classification_report(val_labels, val_preds, target_names=target_names)
    print("\nClassification Report:")
    print(report)

    # Plot confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(val_labels, val_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Ensemble Model Confusion Matrix')
    plt.savefig('ensemble_confusion_matrix.png')
    plt.show()

    return ensemble_model


# Example usage
if __name__ == "__main__":
    # Run cross-validation and create ensemble
    ensemble_model = run_cross_validation_and_ensemble()

    # Predict on a test image
    test_image_path = r'/DATASET/AffectNet/test/1/image0000007.jpg'  # Replace with actual path
    prediction_results = predict_with_ensemble(ensemble_model, test_image_path)

    print(f"\nPrediction Results:")
    print(
        f"Ensemble Prediction: {prediction_results['ensemble_prediction']} with {prediction_results['ensemble_confidence']:.2f}% confidence")
    print("\nClass Probabilities:")
    for emotion, prob in prediction_results['class_probabilities'].items():
        print(f"{emotion}: {prob:.2f}%")

    print("\nIndividual Model Results:")
    for model_result in prediction_results['model_results']:
        print(f"{model_result['model']}: {model_result['prediction']} ({model_result['confidence']:.2f}%)")