# src/ml/model_evaluator.py
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Evaluate and visualize model performance"""
    
    def __init__(self, model, test_data):
        self.model = model
        self.test_data = test_data
        
    def plot_roc_curve(self, y_true, y_pred, antibiotic):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {antibiotic}')
        plt.legend()
        plt.savefig(f'models/evaluation/{antibiotic}_roc.png')
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, antibiotic):
        """Plot confusion matrix"""
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
        
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {antibiotic}')
        plt.savefig(f'models/evaluation/{antibiotic}_cm.png')
        plt.close()
    
    def generate_evaluation_report(self, antibiotic):
        """Generate complete evaluation report"""
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(self.test_data['features'])
        
        y_pred = predictions.cpu().numpy()
        y_true = self.test_data['labels']
        
        # Generate plots
        self.plot_roc_curve(y_true, y_pred, antibiotic)
        self.plot_confusion_matrix(y_true, y_pred, antibiotic)
        
        print(f"Evaluation report saved for {antibiotic}")