from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()

class User(db.Model):
    """
    Stores credentials, role, and presence timestamps of registered clients.
    """
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), default='user')  # 'user' or 'admin'
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    api_token = db.Column(db.String(255), unique=True, nullable=True, index=True)
    
    # Relationships
    uploads = db.relationship('ImageUpload', backref='uploader', lazy=True, cascade="all, delete-orphan")
    feedbacks = db.relationship('Feedback', backref='user', lazy=True, cascade="all, delete-orphan")
    
    def generate_api_token(self):
        import secrets
        self.api_token = f"pat_{secrets.token_hex(24)}"
        return self.api_token
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
        
    def get_last_seen_str(self):
        """
        Returns a human-friendly string representing relative time since last action.
        """
        if not self.last_seen:
            return "Never"
        now = datetime.now(timezone.utc)
        dt = self.last_seen
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
            
        diff = now - dt
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "Active now"
        minutes = seconds / 60
        if minutes < 60:
            return f"{int(minutes)}m ago"
        hours = minutes / 60
        if hours < 24:
            return f"{int(hours)}h ago"
        days = hours / 24
        return f"{int(days)}d ago"
        
    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat(),
            'last_active_str': self.get_last_seen_str(),
            'api_token': self.api_token
        }

class ImageUpload(db.Model):
    """
    Stores uploaded image metadata, extracted visual features,
    initial predictions, and cumulative feedback statistics.
    """
    __tablename__ = 'image_uploads'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(512), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Feature scores (normalized 0.0 to 1.0)
    brightness = db.Column(db.Float, nullable=False)
    contrast = db.Column(db.Float, nullable=False)
    saturation = db.Column(db.Float, nullable=False)
    warmth = db.Column(db.Float, nullable=False)
    edge_density = db.Column(db.Float, nullable=False)
    color_entropy = db.Column(db.Float, nullable=False)
    edge_contrast = db.Column(db.Float, nullable=False)
    color_temp = db.Column(db.Float, nullable=False)
    
    # Prediction scores
    initial_prediction = db.Column(db.Float, nullable=False)
    
    # User ratings aggregation
    average_rating = db.Column(db.Float, default=0.0)
    num_ratings = db.Column(db.Integer, default=0)

    # Relationships
    feedbacks = db.relationship('Feedback', backref='image', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'uploader_email': self.uploader.email if self.uploader else 'Unknown',
            'filename': self.filename,
            'filepath': self.filepath,
            'uploaded_at': self.uploaded_at.isoformat(),
            'features': {
                'brightness': self.brightness,
                'contrast': self.contrast,
                'saturation': self.saturation,
                'warmth': self.warmth,
                'edge_density': self.edge_density,
                'color_entropy': self.color_entropy,
                'edge_contrast': self.edge_contrast,
                'color_temp': self.color_temp
            },
            'initial_prediction': self.initial_prediction,
            'average_rating': self.average_rating,
            'num_ratings': self.num_ratings
        }

class Feedback(db.Model):
    """
    Stores individual user Ground Truth feedback ratings for images.
    """
    __tablename__ = 'feedback'
    
    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('image_uploads.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    rating = db.Column(db.Float, nullable=False)  # User rating (0.0 to 100.0)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    user_ip = db.Column(db.String(45), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'image_id': self.image_id,
            'user_id': self.user_id,
            'user_email': self.user.email if self.user else 'Unknown',
            'rating': self.rating,
            'created_at': self.created_at.isoformat()
        }

class ModelState(db.Model):
    """
    Stores the model's active weights and bias.
    This enables retrieving the current state and tracking updates.
    """
    __tablename__ = 'model_state'
    
    id = db.Column(db.Integer, primary_key=True)
    weights_json = db.Column(db.Text, nullable=False)  # JSON representation of the weight list
    bias = db.Column(db.Float, nullable=False)
    mse = db.Column(db.Float, default=0.0)
    training_samples = db.Column(db.Integer, default=0)
    ridge_lambda = db.Column(db.Float, default=0.5)
    feedback_weight = db.Column(db.Float, default=5.0)
    updated_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    def to_dict(self):
        import json
        return {
            'weights': json.loads(self.weights_json),
            'bias': self.bias,
            'mse': self.mse,
            'training_samples': self.training_samples,
            'ridge_lambda': self.ridge_lambda,
            'feedback_weight': self.feedback_weight,
            'updated_at': self.updated_at.isoformat()
        }
