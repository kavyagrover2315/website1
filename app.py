import os, csv, torch, pandas as pd, warnings, requests
from flask import Flask, render_template, request, redirect, session, jsonify
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn, torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
from dotenv import load_dotenv
load_dotenv()

# ===== Flask Setup =====
app = Flask(__name__)
app.secret_key = "secret"

# ====== Fix Spotipy warning ======
class SafeSpotifyOAuth(SpotifyOAuth):
    def __del__(self):
        try:
            if hasattr(self, "cache_handler") and hasattr(self.cache_handler, "save_token_to_cache"):
                if hasattr(self, "token_info"):
                    self.cache_handler.save_token_to_cache(self.token_info)
        except Exception:
            pass

sp_oauth = SafeSpotifyOAuth(
    client_id="77cf44ff9c5b48eb9d241ef572eaeb35",
    client_secret="6d69a6146d5d49c6b88077dbf3024e12",
    redirect_uri="https://moodmate-kavya.onrender.com/callback"
,
    scope="playlist-modify-public"
)

# ===== CSV Setup for Login =====
CSV_FILE = os.path.join(os.path.dirname(__file__), 'users.csv')
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Full Name', 'Username', 'Password', 'Welcome Message'])

def generate_welcome_message(name):
    return (
        f"Hi {name} 🌈\n"
        f"Welcome! Here, every mood has a voice and every emotion has a place."
    )

# ===== Mood Dataset =====
data = pd.DataFrame({
    "text": [
        "I am feeling awesome today!", "I'm super happy right now!", "Life is amazing!",
        "Smiling all day!", "Best day ever!", "So joyful and cheerful!", "I'm full of energy!",
        "Everything is working out perfectly!", "I feel so satisfied!", "This made me smile!",
        "I'm feeling down.", "So many tears today.", "Nothing feels good.",
        "I'm just not okay.", "I miss everything.", "It hurts a lot inside.",
        "I feel hopeless today.", "Why do I feel like this?", "Lost and sad.", "I'm crying in my room.",
        "I'm furious right now!", "This is so frustrating!", "I hate this feeling.",
        "Why is everyone so annoying?", "I'm done with this!", "I want to scream.",
        "Nothing is going right.", "Everything is making me mad.", "I'm boiling with anger.",
        "Stop testing my patience!",
        "I'm just okay.", "Today was average.", "Nothing special happened.",
        "Same day, different story.", "I'm feeling fine.", "Could be better, could be worse.",
        "Just another day.", "No big emotions today.", "All calm here.", "I'm neutral.",
        "This is so romantic.", "Feeling the love!", "Everything feels so beautiful.",
        "I'm in love.", "Heart is full today.", "Love is in the air!",
        "What a sweet day!", "I feel butterflies!", "So many cute moments today!", "I'm blushing so much!",
        "I feel completely broken.", "I'm numb and can't feel anything.", "Why even try anymore?",
        "So tired of pretending.", "I want to disappear.", "Dark thoughts all day.",
        "Nothing makes sense anymore.", "Can't stop overthinking.", "Mentally exhausted.",
        "Crying silently all night.",
        "I'm shocked beyond words.", "What just happened?!", "Totally unexpected!",
        "I can't believe it!", "That surprised me a lot.", "So stunned right now!",
        "I'm speechless!", "Whoa! Didn’t see that coming!", "This twist is crazy!", "Mind blown!"
    ],
    "label": ["happy"]*10 + ["sad"]*10 + ["angry"]*10 + ["neutral"]*10 +
             ["lovely"]*10 + ["depression"]*10 + ["shocked"]*10
})

vec = CountVectorizer(max_features=1000)
X = vec.fit_transform(data["text"]).toarray()
enc = LabelEncoder()
y = enc.fit_transform(data["label"])

class MoodDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
loader = DataLoader(MoodDataset(X_train, y_train), batch_size=8, shuffle=True)

class MoodClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x): return self.net(x)

model = MoodClassifier(X.shape[1], 64, len(enc.classes_))
loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    for xb, yb in loader:
        opt.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred, yb)
        loss.backward()
        opt.step()

def predict(text):
    text = text.lower().strip()
    keywords = ["happy", "sad", "angry", "neutral", "lovely", "depression", "shocked"]
    for word in keywords:
        if word in text:
            return word
    vec_input = vec.transform([text]).toarray()
    with torch.no_grad():
        out = model(torch.tensor(vec_input, dtype=torch.float32))
    idx = torch.argmax(out, dim=1).item()
    return enc.inverse_transform([idx])[0]

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_ai_suggestion(mood):
    prompt = f"I'm feeling {mood}. Can you give me 3 helpful suggestions?"
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-70b-8192",
            "messages": [
                {"role": "system", "content": "You're a helpful emotional assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 150
        }
    )
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"].strip()
    else:
        return f"⚠️ Groq error {response.status_code}: {response.text}"
# Emojis, Tips, Spotify
EMOJI = {"happy": "😄", "sad": "😢", "angry": "😡", "neutral": "😐",
         "depression": "💔", "shocked": "😲", "lovely": "💕"}
TIPS = {
    "happy": "Celebrate your joy!",
    "sad": "Talk to a friend or write your feelings.",
    "angry": "Take a walk or listen to calming music.",
    "neutral": "Try something new today!",
    "depression": "You're not alone. Please reach out.",
    "shocked": "Stay calm and breathe.",
    "lovely": "Cherish the love around you today 💖"
}
SPOTIFY = {
    "happy": "https://open.spotify.com/playlist/37i9dQZF1DXdPec7aLTmlC",
    "sad": "https://open.spotify.com/playlist/37i9dQZF1DX7qK8ma5wgG1",
    "angry": "https://open.spotify.com/playlist/37i9dQZF1DX3YSRoSdA634",
    "neutral": "https://open.spotify.com/playlist/37i9dQZF1DX3rxVfibe1L0",
    "depression": "https://open.spotify.com/playlist/37i9dQZF1DWZqd5JICZI0u",
    "shocked": "https://open.spotify.com/playlist/37i9dQZF1DX5P7P5gG3dyt",
    "lovely": "https://open.spotify.com/playlist/37i9dQZF1DWYkaDif7Ztbp"
}

# === Routes ===
@app.route('/')
def index():
    return render_template('index.html')  # login/registration page

@app.route("/mood", methods=["GET", "POST"])
def mood_tracker():
    if request.method == "POST":
        user_input = request.form["text"]
        mood = predict(user_input)
        emoji = EMOJI.get(mood, "😐")
        tip = TIPS.get(mood, "Take care of yourself.")
        ai_suggestion = get_ai_suggestion(mood)
        playlist_url = SPOTIFY.get(mood, "")
        return render_template("index1.html", mood=mood, emoji=emoji, tip=tip,
                               ai_suggestion=ai_suggestion, playlist_url=playlist_url)
    return render_template("index1.html")


@app.route("/login")
def login():
    auth_url = sp_oauth.get_authorize_url()
    return redirect(auth_url)

@app.route("/callback")
def callback():
    code = request.args.get("code")
    token_info = sp_oauth.get_access_token(code)
    session["token"] = token_info["access_token"]
    return redirect("/mood")  # ✅ This must redirect to /mood


@app.route("/create_playlist/<mood>")
def create_playlist(mood):
    if "token" not in session:
        return redirect("/login")
    sp = spotipy.Spotify(auth=session["token"])
    user = sp.current_user()
    playlist = sp.user_playlist_create(user["id"], f"{mood.capitalize()} Mood Playlist")
    results = sp.search(q=mood, type="track", limit=15)
    uris = [track["uri"] for track in results["tracks"]["items"]]
    if uris:
        sp.playlist_add_items(playlist["id"], uris)
    return redirect(playlist["external_urls"]["spotify"])

@app.route('/register', methods=['POST'])
def register():
    try:
        full_name = request.form['full_name']
        username = request.form['username']
        password = request.form['password']
        if not full_name or not username or not password:
            return jsonify({'error': 'All fields are required'}), 400
        if len(password) < 8:
            return jsonify({'error': 'Password must be at least 8 characters'}), 400
        welcome_msg = generate_welcome_message(full_name)
        with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([full_name, username, password, welcome_msg])
        return jsonify({'message': 'User registered successfully', 'welcome': welcome_msg}), 200
    except Exception:
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
