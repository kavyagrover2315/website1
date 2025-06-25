from flask import Flask, render_template, request, jsonify
import csv
import os

app = Flask(__name__)

CSV_FILE = os.path.join(os.path.dirname(__file__), 'users.csv')

# Ensure CSV exists with headers
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Full Name', 'Username', 'Password', 'Welcome Message'])

# Generative AI-inspired welcome message for mood tracker
def generate_welcome_message(name):
    return (
        f"Hi {name} 🌈\n"
        f"Welcome! Here, every mood has a voice and every emotion has a place."
    )

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['POST'])
def register():
    try:
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form['password']

        # Validation
        if not full_name or not username or not password:
            return jsonify({'error': 'All fields are required'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400

        # Generate mood-focused welcome message
        welcome_msg = generate_welcome_message(full_name)

        # Save to CSV
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([full_name, username, password, welcome_msg])

        return jsonify({
            'message': 'User registered successfully',
            'welcome': welcome_msg
        }), 200

    except Exception as e:
        return jsonify({'error': 'Internal server error'}), 500

# For deployment
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
