import os
import json
from app import app, db, ModelState
from database import ImageUpload, User
from model import ContinuousLearner

def initialize_application():
    """
    Initializes database tables, creates uploads folder,
    and seeds default model weights if not already initialized.
    """
    print("[*] Initializing PercepAI application components...")
    
    # Ensure uploads directory exists
    uploads_dir = os.path.join(os.path.dirname(__file__), 'uploads')
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)
        print(f"[+] Created uploads directory: {uploads_dir}")
        
    with app.app_context():
        # Check for schema mismatch (e.g. missing columns from updates) and recreate if needed
        try:
            # Check if User table exists and ImageUpload has user_id
            db.session.query(ImageUpload).filter(ImageUpload.user_id == 1).first()
        except Exception:
            print("[!] Database schema mismatch or missing columns. Recreating database...")
            db.drop_all()
            
        db.create_all()
        print("[+] SQLite database tables initialized successfully.")
        
        # Seed ModelState if empty
        state = ModelState.query.first()
        if not state:
            default_learner = ContinuousLearner()
            weights = default_learner.weights.tolist()
            bias = default_learner.bias
            
            # Initial baseline seeds
            seeds = ContinuousLearner.generate_baseline_seeds()
            # Fit default learner on the 20 seeds to get baseline MSE
            weights, bias, mse = default_learner.retrain(seeds)
            
            state = ModelState(
                weights_json=json.dumps(weights),
                bias=bias,
                mse=mse,
                training_samples=len(seeds)
            )
            db.session.add(state)
            db.session.commit()
            print(f"[+] Seeded baseline model state. Weights: {weights}, Bias: {bias}, MSE: {round(mse, 4)}")
        # Seed default Admin user if not exists or enforce credentials
        admin_email = 'admin@percep.ai'
        admin_user = User.query.filter_by(email=admin_email).first()
        if not admin_user:
            admin_user = User(email=admin_email, role='admin')
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            db.session.commit()
            print(f"[+] Seeded default Admin user: {admin_email} / admin123")
        else:
            admin_user.set_password('admin123')
            admin_user.role = 'admin'
            db.session.commit()
            print(f"[+] Reset/Ensured Admin user credentials: {admin_email} / admin123")

if __name__ == '__main__':
    initialize_application()
    print("[*] Launching PercepAI local development server...")
    app.run(debug=True, host='127.0.0.1', port=5000)
