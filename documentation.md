# PercepAI: Complete Technical & File-by-File Documentation

This document provides a highly detailed, file-by-file breakdown of the entire **PercepAI** project. It is intended for developers, machine learning engineers, and system administrators who want to understand the inner workings of every single file in the workspace.

---

## 📂 Project Directory Structure

```text
PercepAI/
├── app.py                     # Main Flask Application & REST API router
├── database.py                # Flask-SQLAlchemy DB Models & Schema Definitions
├── model.py                   # 8D Image Feature Extraction & Continuous Learner Class
├── run.py                     # App Initializer, Seeder, and Development Server Launcher
├── view_db.py                 # SQLite database CLI inspector tool
├── verify_endpoints.py        # Integration test suite validating all API actions
├── capacitor.config.json      # Capacitor Mobile App Target configuration
├── package.json               # Developer Node package dependencies for Capacitor
├── requirements.txt           # Python application package requirements
├── config.json                # Local configuration cache (e.g. SMTP parameters)
├── templates/
│   └── index.html             # Single Page Application (SPA) dashboard layout
├── static/
│   ├── css/
│   │   └── style.css          # Core Styling (Glassmorphism, Dark grid theme, Flex layout)
│   └── js/
│   │   └── app.js             # Frontend MVC controller and state manager
├── tests/
│   └── test_model.py          # Unit tests verifying ML models and math properties
└── uploads/                   # Local directory containing user uploaded image files
```

---

## 🔧 Component Details & Code Explanations

### 1. [model.py](file:///c:/Tharun%20Prg/PercepAI/model.py)
This file handles the image processing, visual feature engineering, and the regression-based mathematical engine.

#### **Feature Extraction Pipeline**
The function `extract_image_features(image_path)` takes a path to an image, converts it to RGB mode, and resizes it to $200 \times 200$ pixels. It extracts an 8-dimensional normalized feature vector:

1. **Brightness**: Maps mean intensity from grayscale values:
   $$\text{Brightness} = \frac{\text{Mean}(\text{Gray})}{255.0}$$
2. **Contrast**: Measures standard deviation of grayscale intensities:
   $$\text{Contrast} = \frac{\text{StdDev}(\text{Gray})}{127.5}$$
3. **Saturation**: Mean saturation channel in HSV space:
   $$S_i = \frac{\max(R,G,B)_i - \min(R,G,B)_i}{\max(R,G,B)_i}$$
4. **Warmth**: Counts pixels with warm hues (Reds, Oranges, Yellows, Magentas) having saturation $> 0.15$ divided by total pixels.
5. **Edge Density**: Computes average spatial gradient intensity representing texture complexity:
   $$\text{Edge Density} = \frac{\text{Mean}(|\nabla_x \text{Gray}| + |\nabla_y \text{Gray}|)}{30.0}$$
6. **Color Entropy**: Shannon entropy of grayscale histogram representing color diversity.
7. **Edge Contrast**: Standard deviation of spatial gradients representing image sharpness.
8. **Color Temp Balance**: Ratio of warm tones to cool tones ($120^\circ - 240^\circ$), describing thermal bias.

#### **ContinuousLearner Class**
Manages live regression weights. Key methods include:
- `predict(features)`: Predicts appreciation score in $[0.0, 100.0]$ using a logistic Sigmoid projection:
  $$p = \frac{100.0}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- `retrain(training_data, regularization)`: Recalibrates model coefficients globally using a closed-form **Weighted Ridge Regression** on logit-transformed rating values:
  $$\mathbf{w}_{bias} = \left(\mathbf{X}_{bias}^T \mathbf{W} \mathbf{X}_{bias} + \lambda \mathbf{I}^*\right)^{-1} \mathbf{X}_{bias}^T \mathbf{W} \mathbf{y}$$
  - $\mathbf{X}_{bias}$: Feature matrix appended with a column of ones.
  - $\mathbf{W}$: Diagonal sample weight matrix (1.0 for synthetic anchors, `feedback_weight` for user ratings).
  - $\lambda$: Regularization penalty (prevents weights from growing too large).
- `generate_baseline_seeds()`: Generates 20 baseline synthetic anchor samples representing neutral default behavior.

---

### 2. [database.py](file:///c:/Tharun%20Prg/PercepAI/database.py)
Configures the SQLite relational structure using Flask-SQLAlchemy.

- **`User`**:
  - `id` (PK, INTEGER)
  - `email` (VARCHAR(255), Unique, Indexed)
  - `password_hash` (VARCHAR(255))
  - `role` (VARCHAR(50)) - Defaults to `'user'`, first registered user becomes `'admin'`.
  - `created_at` / `last_seen` (DATETIME)
  - `api_token` (VARCHAR(255), Unique, Indexed) - Token for programmatic calls (`pat_...`).
  - Methods: `set_password()`, `check_password()`, `generate_api_token()`, `get_last_seen_str()` (relative time e.g. "Active now", "10m ago").
- **`ImageUpload`**:
  - Contains image metadata (`filename`, `filepath`), feature scores (8 dimensions), prediction score, and aggregated rating counts (`average_rating`, `num_ratings`).
- **`Feedback`**:
  - Stores individual user ground truth rating values linked to an image upload. Includes IP tracking `user_ip`.
- **`ModelState`**:
  - Persists active weights as `weights_json` and bias value. Tracks MSE, training sample counts, $\lambda$, and `feedback_weight`.

---

### 3. [app.py](file:///c:/Tharun%20Prg/PercepAI/app.py)
This is the central routing backend and REST API engine. It manages:
- **Authentication**: Registration (`/api/auth/signup`), Login (`/api/auth/login`), Logout (`/api/auth/logout`), Status (`/api/auth/status`).
- **Middleware**: 
  - `login_required`: Restricts routes to authenticated sessions.
  - `admin_required`: Restricts routes to administrators.
  - `token_required`: Validates programmatic API requests with Bearer Tokens in header/query parameters.
- **Model Synchronization**:
  - `retrain_model_globally()`: Triggers when new feedback is posted. Re-evaluates regression weights globally using active feedback rows, updates `ModelState` parameters, and commits changes.
- **OTP Verification Flow**:
  - `request_email_otp()` & `verify_email_otp()`: Changes user emails securely by mailing a 6-digit OTP code using SMTP server variables. Fallbacks to server console output if SMTP config is missing.
- **Admin Dashboards & Deletions**:
  - Cascade deletes images or user files securely, updating weights globally after deletion.
  - Exports weights (`/api/model/export`) as JSON download; imports configuration files (`/api/model/import`).

---

### 4. [run.py](file:///c:/Tharun%20Prg/PercepAI/run.py)
Performs application initialization routines:
- Creates local `uploads/` folder.
- Bootstraps SQLite database tables and resolves schema mismatches automatically.
- Seeds baseline model weights (using 20 synthetic anchors) if database is empty.
- Creates/enforces default Administrator credentials:
  - **Email**: `admin@percep.ai`
  - **Password**: `admin123`
- Launches the Flask app at `http://127.0.0.1:5000`.

---

### 5. [templates/index.html](file:///c:/Tharun%20Prg/PercepAI/templates/index.html)
The template layout for the Single Page Application (SPA). It renders:
- Authentication portals (Login/Signup cards).
- Main Dashboard containing image upload drop-zones, circular SVGs for predictions, feature parameter progress bars, and feedback slider inputs.
- Active weights monitor visualizing regression coefficients in real-time.
- Admin system settings containing registered user grids, ground truth logs, SMTP mail configurations, and report panels.
- Profile settings sliders to adjust learning penalty, feedback weight, apply presets, import/export json weights, and reset the model state.

---

### 6. [static/js/app.js](file:///c:/Tharun%20Prg/PercepAI/static/js/app.js)
Contains frontend MVC controller logic. Highlights:
- **Session Manager**: Calls `/api/auth/status` on load, hides/shows appropriate views based on active session state.
- **Feature Visualizer**: Updates circular progress ring using SVG stroke calculations:
  ```javascript
  const offset = 440 - (scoreValue / 100) * 440;
  ring.style.strokeDashoffset = offset;
  ```
- **Live Weights Grid**: Queries `/api/model-stats`, dynamically rendering color-coded progress bars for each regression coefficient.
- **PDF Report Exporter**: Uses `html2pdf.js` to compile the admin overview dashboard, weight profiles, and database tables into a clean, printable PDF report format.

---

### 7. [static/css/style.css](file:///c:/Tharun%20Prg/PercepAI/static/css/style.css)
Declares root design tokens and layouts:
- Harmonious dark styling using rich gray tones, bright blue gradients (`--accent: #2563eb`), and clean borders (`--border-color`).
- Premium glassmorphism effects (`backdrop-filter`) and smooth transitions (`all 0.3s ease`).
- High readability using modern sans-serif typography (`Inter` & `Outfit` Google Fonts).

---

### 8. [verify_endpoints.py](file:///c:/Tharun%20Prg/PercepAI/verify_endpoints.py)
A full integration test suite simulating multiple API workflows:
- Validates signup constraints (prevents duplicate emails).
- Tests password changes and logins.
- Assesses OTP flows and fallback verification mechanisms.
- Validates cascade deletion routes, ensuring that deleting an upload or user sweeps related files off disk and database tables, subsequently retraining the model state.

---

### 9. [tests/test_model.py](file:///c:/Tharun%20Prg/PercepAI/tests/test_model.py)
Unit tests for core machine learning functionality:
- Tests feature extraction bounds (features must be floats in $[0.0, 1.0]$).
- Assesses specific color representations (solid blue must yield high saturation and low temperature, while orange must yield high warmth).
- Confirms prediction bounds $[0.0, 100.0]$.
- Validates that closed-form Weighted Ridge Regression successfully adapts weights to target feedback trends.

---

## 🚀 Execution & Verification Guides

### **To Run the Dev Server**
```bash
python run.py
```
*Accessible at: `http://127.0.0.1:5000` (Default Admin: `admin@percep.ai` / `admin123`)*

### **To Run Tests**
```bash
# Model Unit Tests
python -m unittest tests/test_model.py

# API Integration Tests
python verify_endpoints.py
```

### **To Inspect the SQLite Database**
```bash
python view_db.py
```
