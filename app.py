import os
import json
import random
from datetime import datetime, timezone
from functools import wraps
from flask import Flask, request, jsonify, render_template, send_from_directory, session
from werkzeug.utils import secure_filename
from database import db, User, ImageUpload, Feedback, ModelState
from model import extract_image_features, ContinuousLearner

app = Flask(__name__)

# Configure session encryption key
app.secret_key = 'percepai_saas_production_key_xyz_123'

# Configure local SQLite database
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(BASE_DIR, "percepai.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure Uploads
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure uploads directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize database with Flask app
db.init_app(app)

# SMTP Config and Email utilities
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')

def get_smtp_config():
    """Loads SMTP configuration from config.json, falling back to environment or empty defaults."""
    config = {
        'smtp_host': os.environ.get('SMTP_HOST', 'smtp.gmail.com'),
        'smtp_port': int(os.environ.get('SMTP_PORT', 587)),
        'smtp_email': os.environ.get('SMTP_EMAIL', ''),
        'smtp_password': os.environ.get('SMTP_PASSWORD', ''),
        'smtp_use_tls': True
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)
                config['smtp_host'] = data.get('smtp_host', config['smtp_host'])
                config['smtp_port'] = int(data.get('smtp_port', config['smtp_port']))
                config['smtp_email'] = data.get('smtp_email', config['smtp_email'])
                config['smtp_password'] = data.get('smtp_password', config['smtp_password'])
                config['smtp_use_tls'] = data.get('smtp_use_tls', True)
        except Exception:
            pass
    return config

def save_smtp_config(config_data):
    """Saves SMTP configuration to config.json."""
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config_data, f, indent=4)
        return True
    except Exception:
        return False

def send_otp_email(target_email, otp_code):
    """
    Sends the 6-digit OTP code to the target email using authenticated SMTP.
    """
    config = get_smtp_config()
    smtp_host = config['smtp_host']
    smtp_port = config['smtp_port']
    smtp_email = config['smtp_email']
    smtp_password = config['smtp_password']
    
    print(f"\n[+] Attempting to send OTP email to {target_email} via SMTP ({smtp_host}:{smtp_port})...")
    
    if not smtp_email or not smtp_password:
        print(f"[!] SMTP credentials not configured. Printing code to console as fallback.")
        print(f"[EMAIL VERIFICATION MOCK OTP] Sent to: {target_email} | OTP Code: {otp_code}\n")
        return False, "SMTP server credentials are not configured in the Admin Panel."

    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        msg = MIMEMultipart()
        msg['From'] = smtp_email
        msg['To'] = target_email
        msg['Subject'] = "PercepAI Email Verification OTP"
        
        body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #0f172a; background-color: #f8fafc; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; padding: 30px; border: 1px solid #e2e8f0; border-radius: 16px; background-color: #ffffff; box-shadow: 0 4px 10px rgba(0,0,0,0.03);">
                    <div style="text-align: center; margin-bottom: 25px;">
                        <h2 style="color: #2563eb; margin: 0; font-size: 1.75rem; font-weight: 800;">PercepAI</h2>
                        <p style="font-size: 0.8rem; color: #64748b; margin: 5px 0 0 0; text-transform: uppercase; letter-spacing: 0.05em;">Security Verification</p>
                    </div>
                    <hr style="border: 0; border-top: 1px solid #e2e8f0; margin-bottom: 25px;" />
                    <p style="font-size: 1rem; color: #334155;">Hello,</p>
                    <p style="font-size: 1rem; color: #334155;">We received a request to update the email address associated with your PercepAI account. Please use the following One-Time Password (OTP) to complete this update:</p>
                    <div style="text-align: center; margin: 35px 0;">
                        <span style="font-size: 2.5rem; font-weight: 800; letter-spacing: 6px; color: #2563eb; background-color: #eff6ff; padding: 12px 24px; border-radius: 10px; border: 1px dashed rgba(37, 99, 235, 0.3); display: inline-block;">{otp_code}</span>
                    </div>
                    <p style="color: #ef4444; font-weight: 600; font-size: 0.9rem; text-align: center;">This code is valid for 10 minutes and should not be shared with anyone.</p>
                    <p style="font-size: 0.9rem; color: #64748b; margin-top: 25px;">If you did not initiate this change, you can safely ignore this email. Your current email remains secure.</p>
                    <hr style="border: 0; border-top: 1px solid #e2e8f0; margin-top: 35px; margin-bottom: 15px;" />
                    <p style="font-size: 0.75rem; color: #94a3b8; text-align: center; margin: 0;">&copy; 2026 PercepAI. Powered by on-the-fly continuous learning.</p>
                </div>
            </body>
        </html>
        """
        msg.attach(MIMEText(body, 'html'))
        
        # Connect to SMTP Server
        server = smtplib.SMTP(smtp_host, smtp_port, timeout=10)
        server.starttls()
        server.login(smtp_email, smtp_password)
        server.sendmail(smtp_email, target_email, msg.as_string())
        server.quit()
        
        print(f"[+] Real OTP email sent successfully to {target_email}!")
        return True, "OTP successfully sent to your email address."
    except Exception as e:
        err_msg = str(e)
        print(f"[-] Failed to send SMTP email: {err_msg}")
        print(f"[FALLBACK - EMAIL VERIFICATION OTP] Sent to: {target_email} | OTP Code: {otp_code}\n")
        return False, f"SMTP delivery failed: {err_msg}"

# Initialize and Seed Database Context
with app.app_context():
    try:
        # Check if tables exist
        User.query.first()
        ModelState.query.first()
    except Exception:
        print("[!] Database schema mismatch or missing tables. Recreating tables...")
        db.drop_all()
        db.create_all()
        
    db.create_all()
    
    # Seed default model weights if empty
    state = ModelState.query.first()
    if not state:
        from model import ContinuousLearner
        default_learner = ContinuousLearner()
        weights = default_learner.weights.tolist()
        bias = default_learner.bias
        
        # Fit default learner on baseline seeds
        seeds = ContinuousLearner.generate_baseline_seeds()
        weights, bias, mse = default_learner.retrain(seeds)
        
        state = ModelState(
            weights_json=json.dumps(weights),
            bias=bias,
            mse=mse,
            training_samples=len(seeds)
        )
        db.session.add(state)
        db.session.commit()
        print(f"[+] Seeded baseline model state. MSE: {round(mse, 4)}")
        
    # Seed/Enforce admin credentials
    admin_email = 'admin@percep.ai'
    admin_user = User.query.filter_by(email=admin_email).first()
    if not admin_user:
        # Check if there is already an admin in the system
        has_admin = User.query.filter_by(role='admin').first()
        if not has_admin:
            admin_user = User(email=admin_email, role='admin')
            admin_user.set_password('admin123')
            db.session.add(admin_user)
            db.session.commit()
            print(f"[+] Seeded default Admin user: {admin_email} / admin123")
    else:
        # If it exists, ensure it is an admin, but DO NOT overwrite their customized password
        if admin_user.role != 'admin':
            admin_user.role = 'admin'
            db.session.commit()
            print(f"[+] Enforced Admin role for: {admin_email}")

# Helper Decorators for Auth check
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Authentication required. Please login.'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            return jsonify({'error': 'Unauthorized. Admin role required.'}), 403
        return f(*args, **kwargs)
    return decorated_function

# Middleware to update presence status of active users
@app.before_request
def update_last_seen():
    # Skip static files and images to save DB transaction cost
    if request.path.startswith('/static/') or request.path.startswith('/uploads/'):
        return
    if 'user_id' in session:
        try:
            # We use scoped sessions safely. Do not crash if DB is not bootstrapped yet.
            user = db.session.get(User, session['user_id'])
            if user:
                user.last_seen = datetime.now(timezone.utc)
                db.session.commit()
        except Exception:
            pass

def get_current_learner():
    """
    Instantiates the ContinuousLearner with the latest weights from the database.
    """
    state = ModelState.query.first()
    if state:
        try:
            weights = json.loads(state.weights_json)
            return ContinuousLearner(weights=weights, bias=state.bias)
        except Exception:
            pass
    return ContinuousLearner()

def retrain_model_globally():
    """
    Gathers all user feedback (weighted by ModelState.feedback_weight) plus synthetic baseline seeds (weighted 1.0),
    runs the closed-form Weighted Ridge Regression with ModelState.ridge_lambda, and updates the ModelState.
    """
    state = ModelState.query.first()
    lambda_val = state.ridge_lambda if state else 0.5
    feedback_w = state.feedback_weight if state else 5.0
    
    feedbacks = Feedback.query.all()
    
    # Generate baseline seeds (weight 1.0)
    seeds = ContinuousLearner.generate_baseline_seeds()
    training_data = []
    
    # Add seeds first (prior)
    for feats, rating, weight in seeds:
        training_data.append((feats, rating, weight))
        
    # Add actual user feedback (weighted by feedback_weight)
    for fb in feedbacks:
        img = fb.image
        if img:
            feats = {
                'brightness': img.brightness,
                'contrast': img.contrast,
                'saturation': img.saturation,
                'warmth': img.warmth,
                'edge_density': img.edge_density,
                'color_entropy': img.color_entropy,
                'edge_contrast': img.edge_contrast,
                'color_temp': img.color_temp
            }
            training_data.append((feats, fb.rating, feedback_w))
            
    # Retrain using weighted ridge regression
    learner = ContinuousLearner()
    weights, bias, mse = learner.retrain(training_data, regularization=lambda_val)
    
    # Update DB
    if not state:
        state = ModelState(
            weights_json=json.dumps(weights),
            bias=bias,
            mse=mse,
            training_samples=len(training_data),
            ridge_lambda=lambda_val,
            feedback_weight=feedback_w
        )
        db.session.add(state)
    else:
        state.weights_json = json.dumps(weights)
        state.bias = bias
        state.mse = mse
        state.training_samples = len(training_data)
        
    db.session.commit()
    return state

# --- AUTHENTICATION ROUTES ---

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    """Registers a new user. The first registered user gets the admin role."""
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
        
    existing_user = User.query.filter_by(email=email).first()
    if existing_user:
        return jsonify({'error': 'User with this email already exists'}), 400
        
    # Determine role (first user = admin)
    role = 'user'
    if User.query.count() == 0:
        role = 'admin'
        
    user = User(email=email, role=role)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    
    # Log the user in immediately
    session['user_id'] = user.id
    session['email'] = user.email
    session['role'] = user.role
    
    return jsonify({
        'success': True,
        'user': user.to_dict()
    })

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Logs in an existing user."""
    data = request.get_json() or {}
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
        
    user = User.query.filter_by(email=email).first()
    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401
        
    # Set session
    session['user_id'] = user.id
    session['email'] = user.email
    session['role'] = user.role
    
    # Update presence last_seen on login
    user.last_seen = datetime.now(timezone.utc)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'user': user.to_dict()
    })

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """Logs out the active user session."""
    session.clear()
    return jsonify({'success': True})

@app.route('/api/auth/status', methods=['GET'])
def auth_status():
    """Returns active session profile if authenticated."""
    if 'user_id' in session:
        user = db.session.get(User, session['user_id'])
        if user:
            return jsonify({
                'logged_in': True,
                'user': user.to_dict()
            })
    return jsonify({
        'logged_in': False,
        'user': None
    })


# --- PLATFORM FUNCTIONAL API ---

@app.route('/')
def home():
    """Renders the main SPA dashboard."""
    return render_template('index.html')

@app.route('/uploads/<filename>')
def serve_upload(filename):
    """Serves uploaded images securely."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/upload', methods=['POST'])
@login_required
def upload_image():
    """
    Accepts an uploaded image, extracts its features (8 dimensions),
    computes prediction, and stores record associated with active user.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file:
        import time
        ts = int(time.time())
        filename = secure_filename(f"{ts}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Extract features (8-dimensional)
        features = extract_image_features(filepath)
        
        # Get active model prediction
        learner = get_current_learner()
        prediction = learner.predict(features)
        
        # Save image record to database linked to logged in user
        img_upload = ImageUpload(
            user_id=session['user_id'],
            filename=filename,
            filepath=filepath,
            brightness=features['brightness'],
            contrast=features['contrast'],
            saturation=features['saturation'],
            warmth=features['warmth'],
            edge_density=features['edge_density'],
            color_entropy=features['color_entropy'],
            edge_contrast=features['edge_contrast'],
            color_temp=features['color_temp'],
            initial_prediction=prediction
        )
        db.session.add(img_upload)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'image': img_upload.to_dict()
        })

@app.route('/api/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """
    Records a user's ground truth perspective rating, averages it
    into the image record, triggers online global weighted retraining,
    and returns updated model coefficients and re-predicted scores.
    """
    data = request.get_json() or {}
    image_id = data.get('image_id')
    rating = data.get('rating')
    
    if image_id is None or rating is None:
        return jsonify({'error': 'Missing image_id or rating'}), 400
        
    try:
        rating = float(rating)
        if not (0.0 <= rating <= 100.0):
            return jsonify({'error': 'Rating must be between 0 and 100'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid rating value'}), 400
        
    img = db.session.get(ImageUpload, image_id)
    if not img:
        return jsonify({'error': 'Image not found'}), 404
        
    # Save feedback entry linked to active user
    feedback = Feedback(
        image_id=image_id,
        user_id=session['user_id'],
        rating=rating,
        user_ip=request.remote_addr
    )
    db.session.add(feedback)
    
    # Recalculate average rating on image
    feedbacks = Feedback.query.filter_by(image_id=image_id).all()
    all_ratings = [f.rating for f in feedbacks]
    img.average_rating = sum(all_ratings) / len(all_ratings)
    img.num_ratings = len(all_ratings)
    
    db.session.commit()
    
    # Trigger online model retraining (weighted)
    new_state = retrain_model_globally()
    
    # Re-evaluate prediction under the new weights
    new_learner = get_current_learner()
    features = {
        'brightness': img.brightness,
        'contrast': img.contrast,
        'saturation': img.saturation,
        'warmth': img.warmth,
        'edge_density': img.edge_density,
        'color_entropy': img.color_entropy,
        'edge_contrast': img.edge_contrast,
        'color_temp': img.color_temp
    }
    updated_prediction = new_learner.predict(features)
    
    return jsonify({
        'success': True,
        'updated_prediction': round(updated_prediction, 2),
        'image': img.to_dict(),
        'model_state': new_state.to_dict()
    })

@app.route('/api/activity', methods=['GET'])
@login_required
def get_activity():
    """
    Returns recent uploads to populate the dashboard activity feed.
    """
    recent_uploads = ImageUpload.query.order_by(ImageUpload.uploaded_at.desc()).limit(12).all()
    return jsonify({
        'uploads': [img.to_dict() for img in recent_uploads]
    })

@app.route('/api/model-stats', methods=['GET'])
@login_required
def get_model_stats():
    """
    Exposes active model weights, biases, MSE, and training size.
    """
    state = ModelState.query.first()
    if not state:
        # Initialize default if it doesn't exist yet
        learner = ContinuousLearner()
        state = ModelState(
            weights_json=json.dumps(learner.weights.tolist()),
            bias=learner.bias,
            mse=0.0,
            training_samples=20 # Seeds
        )
        db.session.add(state)
        db.session.commit()
        
    return jsonify(state.to_dict())


# --- PROFILE SETTINGS ROUTES ---

@app.route('/api/profile/update-password', methods=['POST'])
@login_required
def update_password():
    """Changes password for logged in user after verifying current password."""
    data = request.get_json() or {}
    current_pass = data.get('current_password')
    new_pass = data.get('new_password')
    
    if not current_pass or not new_pass:
        return jsonify({'error': 'Both current and new passwords are required'}), 400
        
    user = db.session.get(User, session['user_id'])
    if not user or not user.check_password(current_pass):
        return jsonify({'error': 'Invalid current password'}), 401
        
    user.set_password(new_pass)
    db.session.commit()
    
    return jsonify({'success': True})

@app.route('/api/profile/request-email-otp', methods=['POST'])
@login_required
def request_email_otp():
    """Generates OTP for email update, attempts SMTP delivery, falls back to console printing."""
    data = request.get_json() or {}
    new_email = data.get('new_email', '').strip().lower()
    
    if not new_email:
        return jsonify({'error': 'New email address is required'}), 400
        
    # Check duplicate email
    existing = User.query.filter_by(email=new_email).first()
    if existing:
        return jsonify({'error': 'Email address already registered'}), 400
        
    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))
    
    # Store in session
    session['email_otp'] = otp
    session['pending_email'] = new_email
    
    # Attempt SMTP delivery
    sent, msg = send_otp_email(new_email, otp)
    
    if not sent:
        # If we are in TESTING mode, fall back to console code logging and succeed
        if app.config.get('TESTING'):
            print(f"[TESTING FALLBACK] OTP Code: {otp}")
            return jsonify({
                'success': True,
                'warning': f'SMTP delivery failed ({msg}). Falling back to test mode.',
                'message': 'OTP printed to server console (fallback).'
            }), 200

        # If it failed because SMTP is not configured
        if "not configured" in msg:
            return jsonify({
                'success': True,
                'warning': 'SMTP server is not configured. The verification code has been printed to the server terminal.',
                'message': 'OTP code printed to server terminal (fallback).'
            }), 200
        else:
            return jsonify({
                'error': f'Failed to send OTP email: {msg}. Please check your SMTP settings in the Admin Panel.'
            }), 500
            
    return jsonify({
        'success': True,
        'message': 'OTP verification code has been sent directly to your new email address.'
    }), 200

@app.route('/api/profile/verify-email-otp', methods=['POST'])
@login_required
def verify_email_otp():
    """Verifies OTP and commits the new email address."""
    data = request.get_json() or {}
    otp = data.get('otp', '').strip()
    
    if not otp:
        return jsonify({'error': 'OTP code is required'}), 400
        
    saved_otp = session.get('email_otp')
    pending_email = session.get('pending_email')
    
    if not saved_otp or not pending_email:
        return jsonify({'error': 'No pending email change request found'}), 400
        
    if otp != saved_otp:
        return jsonify({'error': 'Invalid OTP verification code'}), 400
        
    # Commit changes
    user = db.session.get(User, session['user_id'])
    user.email = pending_email
    db.session.commit()
    
    # Update active session credentials
    session['email'] = pending_email
    
    # Clear temp variables
    session.pop('email_otp', None)
    session.pop('pending_email', None)
    
    return jsonify({
        'success': True,
        'user': user.to_dict()
    })


# --- ADMIN ONLY DASHBOARD ROUTES ---

@app.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_admin_stats():
    """Exposes high-level server parameters for the admin dashboard."""
    user_count = User.query.count()
    upload_count = ImageUpload.query.count()
    feedback_count = Feedback.query.count()
    state = ModelState.query.first()
    
    return jsonify({
        'total_users': user_count,
        'total_uploads': upload_count,
        'total_feedbacks': feedback_count,
        'mse': state.mse if state else 0.0,
        'training_samples': state.training_samples if state else 20,
        'weights': json.loads(state.weights_json) if state else ContinuousLearner.DEFAULT_WEIGHTS,
        'bias': state.bias if state else ContinuousLearner.DEFAULT_BIAS
    })

@app.route('/api/admin/users', methods=['GET'])
@admin_required
def get_admin_users():
    """Exposes registered clients database audits with last seen status."""
    users = User.query.order_by(User.created_at.desc()).all()
    user_list = []
    for u in users:
        user_list.append({
            'id': u.id,
            'email': u.email,
            'role': u.role,
            'created_at': u.created_at.isoformat(),
            'last_active_str': u.get_last_seen_str(),
            'uploads_count': ImageUpload.query.filter_by(user_id=u.id).count(),
            'feedbacks_count': Feedback.query.filter_by(user_id=u.id).count()
        })
    return jsonify({'users': user_list})

@app.route('/api/admin/activity', methods=['GET'])
@admin_required
def get_admin_activity():
    """Exposes all detailed user feed transactions."""
    feedbacks = Feedback.query.order_by(Feedback.created_at.desc()).all()
    uploads = ImageUpload.query.order_by(ImageUpload.uploaded_at.desc()).all()
    
    return jsonify({
        'feedbacks': [fb.to_dict() for fb in feedbacks],
        'uploads': [up.to_dict() for up in uploads]
    })

@app.route('/api/admin/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    """Cascade deletes a user, their uploaded files, and recalculates weights."""
    if user_id == session['user_id']:
        return jsonify({'error': 'Cannot delete your own administrator account'}), 400
        
    user = db.session.get(User, user_id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    # Delete local upload files of this user
    for img in user.uploads:
        if os.path.exists(img.filepath):
            try:
                os.remove(img.filepath)
            except Exception:
                pass
                
    # Delete user from DB (Cascades uploads and feedbacks)
    db.session.delete(user)
    db.session.commit()
    
    # Retrain continuous model globally since feedback has been deleted
    retrain_model_globally()
    
    return jsonify({'success': True})

@app.route('/api/admin/images/<int:image_id>', methods=['DELETE'])
@admin_required
def delete_image(image_id):
    """Deletes an image upload file, cascades feedbacks, and retrains weights."""
    img = db.session.get(ImageUpload, image_id)
    if not img:
        return jsonify({'error': 'Image not found'}), 404
        
    # Delete local file
    if os.path.exists(img.filepath):
        try:
            os.remove(img.filepath)
        except Exception:
            pass
            
    # Delete image upload record (Cascades feedback deletion)
    db.session.delete(img)
    db.session.commit()
    
    # Retrain weights post feedback deletion
    retrain_model_globally()
    
    return jsonify({'success': True})
@app.route('/api/admin/smtp-settings', methods=['GET'])
@admin_required
def get_smtp_settings():
    """Returns active SMTP configuration (hiding the password for safety)."""
    config = get_smtp_config()
    # Mask password for safety
    masked_password = ''
    if config['smtp_password']:
        masked_password = '••••••••'
    return jsonify({
        'smtp_host': config['smtp_host'],
        'smtp_port': config['smtp_port'],
        'smtp_email': config['smtp_email'],
        'has_password': bool(config['smtp_password']),
        'smtp_password': masked_password
    })

@app.route('/api/admin/smtp-settings', methods=['POST'])
@admin_required
def update_smtp_settings():
    """Updates the SMTP config json."""
    data = request.get_json() or {}
    smtp_host = data.get('smtp_host', '').strip()
    smtp_port = data.get('smtp_port')
    smtp_email = data.get('smtp_email', '').strip()
    smtp_password = data.get('smtp_password', '')
    
    if not smtp_host or not smtp_port or not smtp_email:
        return jsonify({'error': 'Host, Port, and Sender Email are required.'}), 400
        
    try:
        smtp_port = int(smtp_port)
    except ValueError:
        return jsonify({'error': 'Port must be a valid integer.'}), 400
        
    # Read existing config to keep password if not modified
    existing = get_smtp_config()
    if smtp_password == '••••••••' or smtp_password == '':
        smtp_password = existing['smtp_password']
        
    config_data = {
        'smtp_host': smtp_host,
        'smtp_port': smtp_port,
        'smtp_email': smtp_email,
        'smtp_password': smtp_password,
        'smtp_use_tls': True
    }
    
    if save_smtp_config(config_data):
        return jsonify({'success': True, 'message': 'SMTP settings saved successfully.'})
    else:
        return jsonify({'error': 'Failed to write config file.'}), 500


# --- ADVANCED MODEL CALIBRATION, API & BACKUP ENDPOINTS ---

def token_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = request.headers.get('Authorization')
        if token and token.startswith('Bearer '):
            token = token[7:]
        else:
            token = request.args.get('token')
            
        if not token:
            return jsonify({'error': 'API token required. Add header "Authorization: Bearer <TOKEN>"'}), 401
            
        user = User.query.filter_by(api_token=token).first()
        if not user:
            return jsonify({'error': 'Invalid API token'}), 401
            
        # Temporarily inject user details in session context
        session['user_id'] = user.id
        session['email'] = user.email
        session['role'] = user.role
        
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/predict', methods=['POST'])
@token_required
def api_predict():
    """
    Programmatic endpoint for MNC client integration.
    Accepts multipart image file in key "image", returns extracted 8D features and prediction score.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided in key "image"'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    import time
    ts = int(time.time())
    filename = secure_filename(f"api_{ts}_{file.filename}")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Extract features and predict
    features = extract_image_features(filepath)
    learner = get_current_learner()
    prediction = learner.predict(features)
    
    # Save image record to database
    img_upload = ImageUpload(
        user_id=session['user_id'],
        filename=filename,
        filepath=filepath,
        brightness=features['brightness'],
        contrast=features['contrast'],
        saturation=features['saturation'],
        warmth=features['warmth'],
        edge_density=features['edge_density'],
        color_entropy=features['color_entropy'],
        edge_contrast=features['edge_contrast'],
        color_temp=features['color_temp'],
        initial_prediction=prediction
    )
    db.session.add(img_upload)
    db.session.commit()
    
    return jsonify({
        'success': True,
        'image_id': img_upload.id,
        'filename': filename,
        'features': features,
        'predicted_appeal_score': round(prediction, 2)
    })

@app.route('/api/profile/token', methods=['GET'])
@login_required
def get_api_token():
    """Retrieves current API token for the logged in user."""
    user = db.session.get(User, session['user_id'])
    return jsonify({'api_token': user.api_token})

@app.route('/api/profile/token/generate', methods=['POST'])
@login_required
def generate_api_token():
    """Generates a new token for the logged in user."""
    user = db.session.get(User, session['user_id'])
    token = user.generate_api_token()
    db.session.commit()
    return jsonify({'api_token': token})

@app.route('/api/model/hyperparameters', methods=['POST'])
@login_required
def update_hyperparameters():
    """Updates model regularization lambda and feedback weighting coefficient."""
    data = request.get_json() or {}
    ridge_lambda = data.get('ridge_lambda')
    feedback_weight = data.get('feedback_weight')
    
    if ridge_lambda is None or feedback_weight is None:
        return jsonify({'error': 'ridge_lambda and feedback_weight are required.'}), 400
        
    try:
        ridge_lambda = float(ridge_lambda)
        feedback_weight = float(feedback_weight)
        if ridge_lambda <= 0 or feedback_weight <= 0:
            return jsonify({'error': 'Values must be positive numbers.'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid parameter types.'}), 400
        
    state = ModelState.query.first()
    if state:
        state.ridge_lambda = ridge_lambda
        state.feedback_weight = feedback_weight
        db.session.commit()
        
    # Trigger global retraining under the new hyperparameters
    retrain_model_globally()
    
    return jsonify({'success': True, 'message': 'Model calibration parameters updated and retrained.'})

@app.route('/api/model/presets', methods=['POST'])
@login_required
def apply_preset():
    """Applies a thematic weights preset directly to the continuous learning model."""
    data = request.get_json() or {}
    preset_name = data.get('preset')
    
    # 8 Dimensions order: brightness, contrast, saturation, warmth, edge_density, color_entropy, edge_contrast, color_temp
    presets = {
        'default': {
            'weights': [0.1, 0.3, 0.2, 0.1, 0.1, 0.05, 0.2, 0.1],
            'bias': -0.5
        },
        'vibrant': {
            'weights': [0.05, 0.25, 0.45, 0.2, 0.05, 0.05, 0.15, 0.25],
            'bias': -0.7
        },
        'minimalist': {
            'weights': [0.35, 0.15, -0.2, 0.1, 0.25, 0.2, 0.2, -0.15],
            'bias': -0.3
        },
        'dramatic': {
            'weights': [-0.15, 0.45, 0.1, -0.1, 0.35, 0.1, 0.4, -0.05],
            'bias': -0.6
        }
    }
    
    if preset_name not in presets:
        return jsonify({'error': 'Invalid preset name.'}), 400
        
    preset = presets[preset_name]
    state = ModelState.query.first()
    if state:
        state.weights_json = json.dumps(preset['weights'])
        state.bias = preset['bias']
        db.session.commit()
        
    # Retrain model globally to anchor it with the new preset weights as baseline seeds
    retrain_model_globally()
    
    return jsonify({
        'success': True,
        'message': f"Preset '{preset_name}' applied and continuous model re-anchored.",
        'weights': preset['weights'],
        'bias': preset['bias']
    })

@app.route('/api/model/reset', methods=['POST'])
@login_required
def reset_model():
    """Deletes all user calibrations, resetting the model to factory baseline weights."""
    Feedback.query.delete()
    db.session.commit()
    
    from model import ContinuousLearner
    default_learner = ContinuousLearner()
    state = ModelState.query.first()
    if state:
        state.weights_json = json.dumps(default_learner.weights.tolist())
        state.bias = default_learner.bias
        state.ridge_lambda = 0.5
        state.feedback_weight = 5.0
        db.session.commit()
        
    retrain_model_globally()
    return jsonify({'success': True, 'message': 'All user calibrations deleted. Model reset to default factory weights.'})

@app.route('/api/model/export', methods=['GET'])
@login_required
def export_model():
    """Exports current weights and configuration as a JSON file download."""
    state = ModelState.query.first()
    if not state:
        return jsonify({'error': 'Model state not found.'}), 404
        
    config = {
        'weights': json.loads(state.weights_json),
        'bias': state.bias,
        'ridge_lambda': state.ridge_lambda,
        'feedback_weight': state.feedback_weight,
        'mse': state.mse,
        'training_samples': state.training_samples,
        'exported_at': datetime.now(timezone.utc).isoformat()
    }
    
    from flask import Response
    return Response(
        json.dumps(config, indent=4),
        mimetype="application/json",
        headers={"Content-disposition": "attachment; filename=percepai_model_weights.json"}
    )

@app.route('/api/model/import', methods=['POST'])
@login_required
def import_model():
    """Imports weights and configuration from an uploaded JSON file."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    try:
        data = json.load(file)
        weights = data.get('weights')
        bias = data.get('bias')
        ridge_lambda = data.get('ridge_lambda', 0.5)
        feedback_weight = data.get('feedback_weight', 5.0)
        
        if not weights or len(weights) != 8 or bias is None:
            return jsonify({'error': 'Invalid JSON format. Must contain 8 weights and a bias.'}), 400
            
        weights = [float(w) for w in weights]
        bias = float(bias)
        
        state = ModelState.query.first()
        if state:
            state.weights_json = json.dumps(weights)
            state.bias = bias
            state.ridge_lambda = float(ridge_lambda)
            state.feedback_weight = float(feedback_weight)
            db.session.commit()
            
        retrain_model_globally()
        return jsonify({'success': True, 'message': 'Model weights and hyperparameters imported successfully.'})
    except Exception as e:
        return jsonify({'error': f'Failed to parse JSON file: {str(e)}'}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
