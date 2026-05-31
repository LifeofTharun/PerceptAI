import os
import io
import json
import unittest
from app import app
from database import db, User, ImageUpload, Feedback, ModelState
from PIL import Image

class TestAPIIntegration(unittest.TestCase):
    def setUp(self):
        # Configure app for testing
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_percepai.db'
        self.client = app.test_client()
        
        # Initialize database
        with app.app_context():
            db.drop_all()
            db.create_all()
            
            # Create test accounts
            admin = User(email='admin@percep.ai', role='admin')
            admin.set_password('adminpass123')
            
            regular = User(email='user@percep.ai', role='user')
            regular.set_password('userpass123')
            
            db.session.add(admin)
            db.session.add(regular)
            db.session.commit()
            
            # Store IDs for session injection
            self.admin_id = admin.id
            self.regular_id = regular.id
            
            # Seed default weights
            from model import ContinuousLearner
            default_learner = ContinuousLearner()
            state = ModelState(
                weights_json=json.dumps(default_learner.weights.tolist()),
                bias=default_learner.bias,
                mse=0.0,
                training_samples=20
            )
            db.session.add(state)
            db.session.commit()
            
        # Create a mock image in memory
        self.img_bytes = io.BytesIO()
        img = Image.new('RGB', (100, 100), color='red')
        img.save(self.img_bytes, format='PNG')
        self.img_bytes.seek(0)

    def tearDown(self):
        with app.app_context():
            db.session.remove()

    def test_auth_status_empty(self):
        """Verify auth status returns logged_in: False when unauthenticated."""
        res = self.client.get('/api/auth/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data['logged_in'])
        self.assertIsNone(data['user'])

    def test_auth_signup_and_login_flow(self):
        """Test user signup, duplicate signup restriction, and login flow."""
        # 1. Sign up new user
        signup_data = {
            'email': 'newuser@percep.ai',
            'password': 'newpassword123'
        }
        res = self.client.post('/api/auth/signup', json=signup_data)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data['success'])
        self.assertEqual(data['user']['email'], 'newuser@percep.ai')
        self.assertEqual(data['user']['role'], 'user')
        
        # 2. Try duplicate signup (should fail)
        res_dup = self.client.post('/api/auth/signup', json=signup_data)
        self.assertEqual(res_dup.status_code, 400)
        self.assertIn('error', res_dup.get_json())
        
        # 3. Logout
        res_out = self.client.post('/api/auth/logout')
        self.assertEqual(res_out.status_code, 200)
        
        # 4. Log in
        login_data = {
            'email': 'newuser@percep.ai',
            'password': 'newpassword123'
        }
        res_in = self.client.post('/api/auth/login', json=login_data)
        self.assertEqual(res_in.status_code, 200)
        self.assertTrue(res_in.get_json()['success'])

    def test_profile_update_password(self):
        """Test profile password changes and subsequent logins."""
        # 1. Inject regular user session
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.regular_id
            sess['email'] = 'user@percep.ai'
            sess['role'] = 'user'
            
        # 2. Change password
        change_data = {
            'current_password': 'userpass123',
            'new_password': 'newsecurepass'
        }
        res = self.client.post('/api/profile/update-password', json=change_data)
        self.assertEqual(res.status_code, 200)
        
        # 3. Logout
        self.client.post('/api/auth/logout')
        
        # 4. Try logging in with old password (should fail)
        old_login = {
            'email': 'user@percep.ai',
            'password': 'userpass123'
        }
        res_old = self.client.post('/api/auth/login', json=old_login)
        self.assertEqual(res_old.status_code, 401)
        
        # 5. Log in with new password (should succeed)
        new_login = {
            'email': 'user@percep.ai',
            'password': 'newsecurepass'
        }
        res_new = self.client.post('/api/auth/login', json=new_login)
        self.assertEqual(res_new.status_code, 200)

    def test_profile_email_otp_change_flow(self):
        """Test email updating via OTP request and verification."""
        # 1. Inject regular user session
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.regular_id
            sess['email'] = 'user@percep.ai'
            sess['role'] = 'user'
            
        # 2. Request OTP
        req_data = {'new_email': 'updated_user@percep.ai'}
        res_req = self.client.post('/api/profile/request-email-otp', json=req_data)
        self.assertEqual(res_req.status_code, 200)
        self.assertIn('message', res_req.get_json())
        
        # Verify session has OTP and pending email
        with self.client.session_transaction() as sess:
            self.assertIn('email_otp', sess)
            self.assertEqual(sess['pending_email'], 'updated_user@percep.ai')
            saved_otp = sess['email_otp']
            
        # 3. Submit wrong OTP (should fail)
        res_wrong = self.client.post('/api/profile/verify-email-otp', json={'otp': '000000'})
        self.assertEqual(res_wrong.status_code, 400)
        
        # 4. Submit correct OTP (should succeed)
        res_right = self.client.post('/api/profile/verify-email-otp', json={'otp': saved_otp})
        self.assertEqual(res_right.status_code, 200)
        
        # Verify database was updated
        with app.app_context():
            user = db.session.get(User, self.regular_id)
            self.assertEqual(user.email, 'updated_user@percep.ai')

    def test_admin_deletions_cascade(self):
        """Verify admin can delete uploads and users, triggering cascading deletions."""
        # 1. Inject regular user session
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.regular_id
            sess['email'] = 'user@percep.ai'
            sess['role'] = 'user'
            
        # 2. Upload image and vote under regular user
        data = {'image': (self.img_bytes, 'test_delete.png')}
        res_up = self.client.post('/api/upload', data=data, content_type='multipart/form-data')
        self.assertEqual(res_up.status_code, 200)
        img_id = res_up.get_json()['image']['id']
        
        res_fb = self.client.post('/api/feedback', json={'image_id': img_id, 'rating': 80.0})
        self.assertEqual(res_fb.status_code, 200)
        
        # Confirm items exist in DB
        with app.app_context():
            self.assertIsNotNone(db.session.get(ImageUpload, img_id))
            self.assertEqual(Feedback.query.filter_by(image_id=img_id).count(), 1)
            
        # 3. Log in as admin
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.admin_id
            sess['email'] = 'admin@percep.ai'
            sess['role'] = 'admin'
            
        # 4. Delete the image upload first
        res_del_img = self.client.delete(f'/api/admin/images/{img_id}')
        self.assertEqual(res_del_img.status_code, 200)
        
        # Confirm image and feedback are deleted
        with app.app_context():
            self.assertIsNone(db.session.get(ImageUpload, img_id))
            self.assertEqual(Feedback.query.filter_by(image_id=img_id).count(), 0)
            
        # 5. Upload another image for user deletion cascade test
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.regular_id
            sess['email'] = 'user@percep.ai'
            sess['role'] = 'user'
        
        # Create a fresh image stream for the second upload since the first is closed by Flask
        img_bytes2 = io.BytesIO()
        img2 = Image.new('RGB', (100, 100), color='green')
        img2.save(img_bytes2, format='PNG')
        img_bytes2.seek(0)
        
        res_up2 = self.client.post('/api/upload', data={'image': (img_bytes2, 'test_cascade.png')}, content_type='multipart/form-data')
        img_id2 = res_up2.get_json()['image']['id']
        
        # Log back as admin and delete the user
        with self.client.session_transaction() as sess:
            sess['user_id'] = self.admin_id
            sess['email'] = 'admin@percep.ai'
            sess['role'] = 'admin'
            
        res_del_usr = self.client.delete(f'/api/admin/users/{self.regular_id}')
        self.assertEqual(res_del_usr.status_code, 200)
        
        # Confirm user and all their uploads are deleted
        with app.app_context():
            self.assertIsNone(db.session.get(User, self.regular_id))
            self.assertIsNone(db.session.get(ImageUpload, img_id2))

if __name__ == '__main__':
    unittest.main()
