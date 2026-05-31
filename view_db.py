import os
from app import app
from database import db, User, ImageUpload, Feedback, ModelState

def inspect_database():
    print("=" * 60)
    print("                PERCEPAI DATABASE INSPECTOR")
    print("=" * 60)
    
    db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'percepai.db')
    print(f"Database File: {db_path}")
    print(f"File Exists: {os.path.exists(db_path)}")
    if os.path.exists(db_path):
        print(f"File Size: {os.path.getsize(db_path)} bytes")
        
    print("-" * 60)
    with app.app_context():
        # 1. Inspect Users Table
        print("[*] Registered Users:")
        users = User.query.all()
        if not users:
            print("  No users found in database.")
        for u in users:
            print(f"  - ID: {u.id} | Email: {u.email} | Role: {u.role} | Created: {u.created_at} | Last Active: {u.last_seen}")
            
        print("-" * 60)
        # 2. Inspect ModelState Table
        print("[*] Active Model State:")
        state = ModelState.query.first()
        if state:
            print(f"  - Samples: {state.training_samples} | MSE: {round(state.mse, 4)}")
            print(f"  - Regularization (Lambda): {state.ridge_lambda} | Feedback Weight: {state.feedback_weight}")
        else:
            print("  No model state seeded yet.")
            
        print("-" * 60)
        # 3. Inspect Uploads and Feedbacks
        print(f"[*] Total Uploaded Images: {ImageUpload.query.count()}")
        print(f"[*] Total Ratings/Feedback: {Feedback.query.count()}")
    print("=" * 60)

if __name__ == '__main__':
    inspect_database()
