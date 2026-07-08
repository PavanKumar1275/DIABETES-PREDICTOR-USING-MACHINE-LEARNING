"""
User Authentication Module for Diabetes Prediction App
Handles user registration, login, and session management
"""
import sqlite3
import hashlib
import re
import os
from datetime import datetime

# Database setup
DB_PATH = "users.db"

def init_db():
    """Initialize the user database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Create users table
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  email TEXT UNIQUE,
                  phone TEXT UNIQUE,
                  password_hash TEXT,
                  name TEXT,
                  created_at TIMESTAMP)''')
    
    # Create sessions table
    c.execute('''CREATE TABLE IF NOT EXISTS sessions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  session_token TEXT UNIQUE,
                  created_at TIMESTAMP,
                  expires_at TIMESTAMP,
                  FOREIGN KEY (user_id) REFERENCES users (id))''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash a password for storing"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(stored_password, provided_password):
    """Verify a stored password against one provided by user"""
    return stored_password == hash_password(provided_password)

def is_valid_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_phone(phone):
    """Validate phone number format"""
    pattern = r'^[0-9]{10,15}$'
    return re.match(pattern, phone) is not None

def register_user(email, phone, password, name):
    """Register a new user"""
    # Validate inputs
    if not is_valid_email(email):
        return False, "Invalid email format"
    
    if not is_valid_phone(phone):
        return False, "Invalid phone number format (10-15 digits only)"
    
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    
    if not name:
        return False, "Name is required"
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Check if user already exists
        c.execute("SELECT id FROM users WHERE email = ? OR phone = ?", (email, phone))
        if c.fetchone():
            return False, "User with this email or phone already exists"
        
        # Insert new user
        password_hash = hash_password(password)
        c.execute("INSERT INTO users (email, phone, password_hash, name, created_at) VALUES (?, ?, ?, ?, ?)",
                  (email, phone, password_hash, name, datetime.now()))
        conn.commit()
        return True, "User registered successfully"
    
    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}"
    
    finally:
        conn.close()

def authenticate_user(identifier, password):
    """Authenticate user with email/phone and password"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Check if user exists with email or phone
        c.execute("SELECT id, email, phone, password_hash, name FROM users WHERE email = ? OR phone = ?", 
                  (identifier, identifier))
        user = c.fetchone()
        
        if user and verify_password(user[3], password):
            return True, {"id": user[0], "email": user[1], "phone": user[2], "name": user[4]}
        else:
            return False, "Invalid credentials"
    
    except sqlite3.Error as e:
        return False, f"Database error: {str(e)}"
    
    finally:
        conn.close()

def create_session(user_id):
    """Create a new session for user"""
    import uuid
    session_token = str(uuid.uuid4())
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # Insert new session
        c.execute("INSERT INTO sessions (user_id, session_token, created_at, expires_at) VALUES (?, ?, ?, ?)",
                  (user_id, session_token, datetime.now(), datetime.now()))
        conn.commit()
        return session_token
    
    except sqlite3.Error as e:
        return None
    
    finally:
        conn.close()

def validate_session(session_token):
    """Validate if session is still active"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        c.execute("SELECT user_id FROM sessions WHERE session_token = ?", (session_token,))
        result = c.fetchone()
        return result[0] if result else None
    
    except sqlite3.Error:
        return None
    
    finally:
        conn.close()

# Initialize database when module is imported
init_db()