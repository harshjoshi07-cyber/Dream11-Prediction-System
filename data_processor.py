import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            'strike_rate', 'economy', 'recent_form', 'consistency_score',
            '4s', '6s', '50s', '100s', 'wicket', 'catch', 'stump', 'run_out',
            'is_batsman', 'is_bowler', 'is_allrounder'
        ]
        
    def load_data(self):
        """Load and preprocess the IPL data"""
        self.df = pd.read_csv(self.data_path)
        
    def engineer_features(self):
        """Create new features from existing data"""
        # Calculate strike rate for batsmen
        self.df['strike_rate'] = self.df.apply(
            lambda x: x['run_scored'] / x['ball_faced'] if x['ball_faced'] > 0 else 0, axis=1
        )
        
        # Calculate economy rate for bowlers
        self.df['economy'] = self.df.apply(
            lambda x: x['run_given'] / (x['ball_delivered'] / 6) if x['ball_delivered'] > 0 else 0, axis=1
        )
        
        # Calculate recent form (last 5 matches)
        self.df['recent_form'] = self.df.groupby('player_id')['dream11_score'].transform(
            lambda x: x.rolling(5, min_periods=1).mean()
        )
        
        # Calculate consistency metrics
        self.df['consistency_score'] = self.df.groupby('player_id')['dream11_score'].transform(
            lambda x: x.rolling(10, min_periods=1).std()
        )
        
        # Create role-based features
        self.df['is_batsman'] = (self.df['ball_faced'] > 0).astype(int)
        self.df['is_bowler'] = (self.df['ball_delivered'] > 0).astype(int)
        self.df['is_allrounder'] = ((self.df['ball_faced'] > 0) & (self.df['ball_delivered'] > 0)).astype(int)
        
        # Fill missing values with 0
        for col in self.feature_columns:
            if col not in self.df.columns:
                self.df[col] = 0
            self.df[col] = self.df[col].fillna(0)
        
    def prepare_training_data(self):
        """Prepare features for model training"""
        X = self.df[self.feature_columns]
        y = self.df['dream11_score']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
    def train_model(self):
        """Train XGBoost model"""
        X_train, X_test, y_train, y_test = self.prepare_training_data()
        
        self.model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return {
            'r2_score': r2,
            'rmse': rmse
        }
        
    def predict_player_score(self, player_data):
        """Predict Dream11 score for a player"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in player_data:
                player_data[col] = 0
                
        # Prepare player data
        player_features = player_data[self.feature_columns].values.reshape(1, -1)
        player_features = self.scaler.transform(player_features)
        return self.model.predict(player_features)[0] 