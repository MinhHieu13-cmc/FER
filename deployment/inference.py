import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import pandas as pd
import os
import argparse

from ..models.backbone.fer_mobile_vit import FERMobileViTDAN
from ..models.backbone.fer_iformer_dan import create_fer_iformer_dan


class EnsembleModel(nn.Module):
    """Ensemble model that combines predictions from multiple models"""

    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # Average predictions from all models
        outputs = [model(x) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)


class FERDeployment:
    def __init__(self, model_path, model_type='mobilevit', ensemble=False, device=None, num_heads=2):
        """
        Initialize FER deployment

        Args:
            model_path: Path to saved model weights
            model_type: 'mobilevit' or 'iformer'
            ensemble: Whether this is an ensemble model
            device: torch.device or None (will use cuda if available when None)
            num_heads: Number of heads for iFormer-DAN model
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.ensemble = ensemble
        self.model_type = model_type
        self.emotions = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

        # Load model based on type
        if not ensemble:
            # Single model
            if model_type == 'mobilevit':
                self.model = FERMobileViTDAN(num_classes=3)
            elif model_type == 'iformer':
                self.model = create_fer_iformer_dan(num_classes=3, num_heads=num_heads)
            else:
                raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")

            # Load weights
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)

            self.model = self.model.to(self.device)

        else:  # ensemble model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.models = []

            for i, state_dict in enumerate(checkpoint['models']):
                if model_type == 'mobilevit':
                    model = FERMobileViTDAN(num_classes=3)
                elif model_type == 'iformer':
                    model = create_fer_iformer_dan(num_classes=3, num_heads=num_heads)
                else:
                    raise ValueError(f"Loại mô hình không được hỗ trợ: {model_type}")

                model.load_state_dict(state_dict)
                model = model.to(self.device)
                self.models.append(model)

            self.model = EnsembleModel(self.models)

        # Set model to evaluation mode
        self.model.eval()

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path, visualize=True):
        """
        Predict emotion from image

        Args:
            image_path: Path to image file
            visualize: Whether to show visualization

        Returns:
            Dict with prediction results
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Get prediction
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probabilities, 1)

            # Get class probabilities
            class_probs = {self.emotions[i]: float(p) * 100 for i, p in enumerate(probabilities[0])}

            # Get prediction result
            predicted_emotion = self.emotions[pred.item()]
            confidence = conf.item() * 100

        result = {
            'predicted_emotion': predicted_emotion,
            'confidence': confidence,
            'class_probabilities': class_probs
        }

        if visualize:
            self._visualize_prediction(image, result)

        return result

    def batch_predict(self, csv_file, output_csv=None):
        """
        Run batch prediction on images from CSV file

        Args:
            csv_file: Path to CSV file with image paths
            output_csv: Path to save results CSV

        Returns:
            DataFrame with predictions
        """
        # Load data
        df = pd.read_csv(csv_file)
        print(f"Đã đọc file CSV với {len(df)} hàng")

        # In ra một vài hàng đầu tiên để kiểm tra
        print("Dữ liệu CSV:")
        print(df.head())

        # Create results columns
        results = []

        # Process each image
        for idx, row in df.iterrows():
            image_path = row[0]  # Assuming first column is image path

            # Skip if file doesn't exist
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            # Get prediction
            try:
                result = self.predict_image(image_path, visualize=False)

                # Add to results
                result_row = {
                    'image_path': image_path,
                    'predicted_emotion': result['predicted_emotion'],
                    'confidence': result['confidence']
                }

                # Add actual emotion if available
                if len(df.columns) > 1:
                    # Chuyển đổi actual_emotion từ số sang chuỗi nếu cần
                    actual_label = row[1]  # Giá trị nhãn thực tế (có thể là 0, 1, 2)

                    if isinstance(actual_label, (int, float)) or (
                            isinstance(actual_label, str) and actual_label.isdigit()):
                        # Nếu là số hoặc chuỗi số, chuyển đổi sang tên cảm xúc
                        actual_label = int(float(actual_label))  # Chuyển về số nguyên
                        if actual_label in self.emotions:
                            actual_emotion = self.emotions[actual_label]
                        else:
                            print(f"Warning: Invalid actual_emotion value: {actual_label}")
                            actual_emotion = str(actual_label)  # Giữ nguyên nếu không ánh xạ được
                    else:
                        # Nếu đã là chuỗi không phải số, giữ nguyên
                        actual_emotion = actual_label

                    result_row['actual_emotion'] = actual_emotion

                results.append(result_row)

                # Print progress
                if (idx + 1) % 10 == 0:
                    print(f"Processed {idx + 1}/{len(df)} images")

            except Exception as e:
                print(f"Error processing {image_path}: {str(e)}")

        # Create results DataFrame
        results_df = pd.DataFrame(results)

        # Save if output_csv provided
        if output_csv:
            results_df.to_csv(output_csv, index=False)
            print(f"Results saved to {output_csv}")

        # Calculate accuracy if actual emotion available
        if 'actual_emotion' in results_df.columns:
            accuracy = (results_df['predicted_emotion'] == results_df['actual_emotion']).mean() * 100
            print(f"Batch accuracy: {accuracy:.2f}%")

        return results_df

    def _visualize_prediction(self, image, result):
        """Visualize prediction results"""
        plt.figure(figsize=(10, 6))

        # Display image with prediction
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f"Predicted: {result['predicted_emotion']}\nConfidence: {result['confidence']:.2f}%")
        plt.axis('off')

        # Display probabilities
        plt.subplot(1, 2, 2)
        emotions = list(result['class_probabilities'].keys())
        probs = list(result['class_probabilities'].values())
        colors = ['red', 'gray', 'green']

        bars = plt.bar(emotions, probs, color=colors)
        plt.title('Class Probabilities')
        plt.ylabel('Probability (%)')
        plt.ylim(0, 100)

        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 1,
                     f'{height:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description='FER Model Deployment')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--model-type', type=str, default='mobilevit',
                        choices=['mobilevit', 'iformer'], help='Model architecture type')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble model')
    parser.add_argument('--num-heads', type=int, default=2, help='Number of heads for iFormer-DAN')
    parser.add_argument('--input', type=str, help='Path to input image or CSV file')
    parser.add_argument('--output', type=str, help='Path to output CSV file for batch prediction')
    parser.add_argument('--batch', action='store_true', help='Run batch prediction on CSV file')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')

    args = parser.parse_args()

    # Initialize deployment
    deployment = FERDeployment(
        model_path=args.model,
        model_type=args.model_type,
        ensemble=args.ensemble,
        num_heads=args.num_heads
    )

    # Run prediction
    if args.batch:
        if not args.input:
            parser.error("--batch requires --input CSV file")

        deployment.batch_predict(args.input, args.output)
    else:
        if not args.input:
            parser.error("--input image file is required")

        result = deployment.predict_image(args.input, visualize=not args.no_viz)
        print(f"Predicted emotion: {result['predicted_emotion']} ({result['confidence']:.2f}%)")
        print("Class probabilities:")
        for emotion, prob in result['class_probabilities'].items():
            print(f"  {emotion}: {prob:.2f}%")


if __name__ == "__main__":
    main()