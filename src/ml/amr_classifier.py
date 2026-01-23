# src/ml/amr_classifier.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib

class AMRDataset(Dataset):
    """PyTorch Dataset for AMR prediction"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class AMRClassifier(nn.Module):
    """Neural network for AMR prediction"""
    
    def __init__(self, input_dim=370, hidden_dims=[512, 256, 128], dropout=0.3):
        super(AMRClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class AMRModelTrainer:
    """Train AMR prediction models"""
    
    def __init__(self, feature_extractor, device='cuda'):
        self.feature_extractor = feature_extractor
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.models = {}  # One model per antibiotic
        
    def prepare_dataset(self, data_csv='data/processed/training_data.csv'):
        """Prepare features from genome sequences"""
        df = pd.read_csv(data_csv)
        
        print("Extracting features from genomes...")
        features_list = []
        labels_list = []
        antibiotics_list = []
        
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"Processing {idx}/{len(df)}")
            
            try:
                # Extract features
                genome_path = row['genome_path']
                feature_dict = self.feature_extractor.extract_features(genome_path)
                
                features_list.append(feature_dict['features'])
                labels_list.append(row['resistance'])
                antibiotics_list.append(row['antibiotic'])
                
            except Exception as e:
                print(f"Error processing {row['sample_id']}: {e}")
                continue
        
        # Save processed features
        processed_data = {
            'features': np.array(features_list),
            'labels': np.array(labels_list),
            'antibiotics': antibiotics_list
        }
        
        joblib.dump(processed_data, 'data/processed/extracted_features.pkl')
        print(f"Saved {len(features_list)} processed samples")
        
        return processed_data
    
    def train_model_for_antibiotic(self, antibiotic: str, X, y, epochs=50, batch_size=32):
        """Train a model for specific antibiotic"""
        print(f"\n{'='*60}")
        print(f"Training model for {antibiotic}")
        print(f"{'='*60}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Resistance ratio - Train: {y_train.mean():.2f}, Test: {y_test.mean():.2f}")
        
        # Create datasets
        train_dataset = AMRDataset(X_train, y_train)
        test_dataset = AMRDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        model = AMRClassifier(input_dim=X.shape[1]).to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_auc = 0
        for epoch in range(epochs):
            # Train
            model.train()
            train_loss = 0
            for features, labels in train_loader:
                features = features.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Evaluate
            model.eval()
            test_predictions = []
            test_labels = []
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features = features.to(self.device)
                    outputs = model(features)
                    test_predictions.extend(outputs.cpu().numpy())
                    test_labels.extend(labels.numpy())
            
            # Calculate metrics
            test_predictions = np.array(test_predictions)
            test_labels = np.array(test_labels)
            test_auc = roc_auc_score(test_labels, test_predictions)
            
            scheduler.step(train_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f}, AUC: {test_auc:.4f}")
            
            # Save best model
            if test_auc > best_auc:
                best_auc = test_auc
                torch.save(model.state_dict(), f'models/checkpoints/{antibiotic}_best.pth')
        
        # Final evaluation
        print(f"\nFinal Results for {antibiotic}:")
        print(f"Best AUC: {best_auc:.4f}")
        
        # Binary predictions
        binary_preds = (test_predictions > 0.5).astype(int).flatten()
        print("\nClassification Report:")
        print(classification_report(test_labels, binary_preds, 
                                    target_names=['Susceptible', 'Resistant']))
        
        self.models[antibiotic] = model
        return model, best_auc
    
    def train_all_antibiotics(self):
        """Train models for all antibiotics"""
        # Load processed features
        data = joblib.load('data/processed/extracted_features.pkl')
        
        features = data['features']
        labels = data['labels']
        antibiotics = data['antibiotics']
        
        # Get unique antibiotics
        unique_antibiotics = list(set(antibiotics))
        
        results = {}
        for antibiotic in unique_antibiotics:
            # Filter data for this antibiotic
            mask = [ab == antibiotic for ab in antibiotics]
            X_ab = features[mask]
            y_ab = labels[mask]
            
            # Check if we have enough samples
            if len(X_ab) < 50:
                print(f"Skipping {antibiotic} - insufficient data ({len(X_ab)} samples)")
                continue
            
            # Train model
            model, auc = self.train_model_for_antibiotic(antibiotic, X_ab, y_ab)
            results[antibiotic] = auc
        
        # Save results summary
        results_df = pd.DataFrame.from_dict(results, orient='index', columns=['AUC'])
        results_df.to_csv('models/training_results.csv')
        print("\n" + "="*60)
        print("Training Complete! Results:")
        print(results_df)
        
        return results


# Training script
if __name__ == "__main__":
    from feature_extractor import CombinedFeatureExtractor
    
    # Initialize
    feature_extractor = CombinedFeatureExtractor()
    trainer = AMRModelTrainer(feature_extractor)
    
    # Step 1: Extract features (only need to do once)
    # trainer.prepare_dataset()
    
    # Step 2: Train models
    results = trainer.train_all_antibiotics()