import bcrypt
from flask import Blueprint, request, jsonify
from database import get_db

auth_bp = Blueprint("auth", __name__)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())


@auth_bp.post("/api/auth/register")
def register():
    data     = request.get_json(force=True)
    username = data.get("username", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not username or not email or not password:
        return jsonify({"error": "All fields required"}), 400

    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400

    conn   = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (%s, %s, %s, %s)",
            (username, email, hash_password(password), "user"),
        )
        conn.commit()
        return jsonify({"message": "User registered successfully"}), 201
    except Exception:
        return jsonify({"error": "Username or email already exists"}), 409
    finally:
        conn.close()


@auth_bp.post("/api/auth/login")
def login():
    data     = request.get_json(force=True)
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    conn   = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
    user = cursor.fetchone()
    conn.close()

    if not user:
        return jsonify({"error": "User not found"}), 404

    if not verify_password(password, user["password_hash"]):
        return jsonify({"error": "Wrong password"}), 401

    return jsonify({
        "message":  "Login successful",
        "user_id":  user["id"],
        "username": user["username"],
        "email":    user["email"],
        "role":     user["role"],
    }), 200


@auth_bp.get("/api/auth/history/<int:user_id>")
def get_history(user_id):
    from models import get_user_history
    return jsonify(get_user_history(user_id)), 200
