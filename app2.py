from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_session import Session
from flask_bcrypt import Bcrypt
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from dotenv import load_dotenv
import PyPDF2
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import tiktoken
import openai
from flask_socketio import SocketIO
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import redis

app1 = Flask(__name__)
CORS(app1, supports_credentials=True)
bcrypt = Bcrypt(app1)
socketio = SocketIO(app1, cors_allowed_origins="*")

# Configure Redis for session storage and caching
app1.config['SESSION_TYPE'] = 'redis'
app1.config['SESSION_PERMANENT'] = False
app1.config['SESSION_USE_SIGNER'] = True
app1.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')

# Configure SQLite database
app1.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app1.db'
app1.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')
db = SQLAlchemy(app1)

# Configure session
server_session = Session(app1)

# Configure rate limiting
limiter = Limiter(
    get_remote_address,
    app1=app1,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="redis://localhost:6379"
)

load_dotenv()

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)

# Feedback model
class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# UserStats model
class UserStats(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    total_questions = db.Column(db.Integer, default=0)
    total_difficulty = db.Column(db.Integer, default=0)

with app1.app1_context():
    db.create_all()

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() for page in pdf_reader.pages)

def count_tokens_accurate(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")#encoding scheme
    return len(encoding.encode(text))

def generate_questions(text_chunk: str, api_key: str, num_questions: int, question_type: str, difficulty: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    prompt = (
        f"Create {num_questions} {question_type} questions based on the following text. "
        f"The difficulty level should be {difficulty}:\n\n"
        f"{text_chunk}\n\n"
    )
    
    if question_type == "multiple-choice":
        prompt += (
            "Format each question as follows:\n\n"
            "1. [Question text]\n\n"
            "[A] Option A\n\n"
            "[B] Option B\n\n"
            "[C] Option C\n\n"
            "[D] Option D\n\n"
            "Correct answer: [Letter]\n\n"
            "Ensure there is an empty line between each question for readability.\n"
        )
    elif question_type == "fill-in-the-blank":
        prompt += (
    "Create engaging fill-in-the-blank questions based on the provided content. Follow this structured format for each question:\n\n"
    
    "1. Present a clear sentence with a blank (represented by '____') where a word or short phrase should be filled in. Ensure the context provides clues for the correct answer.\n\n"
    
    "2. On the next line, provide a brief explanation or hint (if app1licable) to give additional context or learning value.\n\n"
    
    "3. On the following line, provide the correct answer in **bold** (surrounded by **asterisks**). This helps to highlight the expected response clearly.\n\n"
    
    "4. Leave an empty line between each question for improved readability.\n\n"
    
    "Example:\n\n"
    "1. The capital of France is ____.\n\n"
    "   *(Hint: It's known as the City of Light. )*\n\n"
    "   **Answer:** Paris \n\n"
    
    "2. Water freezes at ____ degrees Celsius.\n\n"
    "   *(Hint: This is the temperature at which liquid water turns into ice at standard atmospheric pressure.)*\n\n"
    "   **Answer:** 0 \n\n"
    
    "Now, generate similar fill-in-the-blank questions based on the given context, ensuring each question is educational and thought-provoking."
)

    elif question_type == "true-false":
        prompt += (
    "Create True/False questions based on the provided content. Use the following format for each question:\n\n"
    "1. [Statement]\n"
    "   Is the above statement True or False?\n\n"
    # "   **Answer:** [True/False]\n\n"
    "Guidelines:\n"
    "- The statements should be clear, concise, and fact-based.\n\n"
    "- Ensure the statements test knowledge, critical thinking, or common misconceptions.\n"
    "- Provide an accurate and unambiguous answer (True or False).\n\n"
    "- Leave a blank line between each question to enhance readability.\n\n"
    "Example:\n\n"
    "1. The Earth is the third planet from the Sun.\n\n"
    "   Is the above statement True or False?\n\n"
    "   **Answer:** True\n\n"
    "2. Water boils at 90 degrees Celsius at sea level.\n\n"
    "   Is the above statement True or False?\n\n"
    "   **Answer:** False\n\n"
    "Now, generate similar True/False questions based on the given context."
)

    elif question_type == "short-answer":
        prompt += (
            "Format each question as follows:\n\n"
            "1. [Question text]\n\n"
            "Sample answer: [Provide a brief sample answer]\n\n"
            "Ensure there is an empty line between each question for readability.\n\n"
        )
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[ 
            {"role": "system", "content": f"You are a helpful assistant that creates {question_type} questions exactly as instructed. Ensure proper formatting with new lines between options and questions."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1500,
        n=1,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

@app1.route('/register', methods=['POST'])
def register():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username already exists"}), 400
    
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    
    return jsonify({"message": "User  registered successfully"}), 201

@app1.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    user = User.query.filter_by(username=username).first()
    if user and bcrypt.check_password_hash(user.password, password):
        session['user_id'] = user.id
        return jsonify({"message": "Login successful"}), 200
    
    return jsonify({"error": "Invalid username or password"}), 401

@app1.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({"message": "Logout successful"}), 200

@app1.route('/check-auth', methods=['GET'])
def check_auth():
    return jsonify({"isAuthenticated": 'user_id' in session}), 200

@app1.route('/user-stats', methods=['GET'])
def get_user_stats():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401
    
    user_id = session['user_id']
    stats = UserStats.query.filter_by(user_id=user_id).first()
    
    if not stats:
        return jsonify({"error": "No stats found"}), 404
    
    avg_difficulty = stats.total_difficulty / stats.total_questions if stats.total_questions > 0 else 0
    return jsonify({
        "totalQuestions": stats.total_questions,
        "avgDifficulty": round(avg_difficulty, 2)
    }), 200

@app1.route('/process', methods=['POST'])
@limiter.limit("10 per minute")
def process():
    if 'user_id' not in session:
        return jsonify({"error": "Not authenticated"}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    action = request.form.get('action')
    if not action:
        return jsonify({"error": "No action specified"}), 400

    try:
        text = extract_text_from_pdf(file)
        tokens = count_tokens_accurate(text)

        api_key = os.getenv("api_key")
        if not api_key:
            return jsonify({"error": "OpenAI API key not found"}), 500

        if action in ["MCQ", "Fill-in-the-Blank", "True-False", "Short-Answer"]:
            num_questions = int(request.form.get('numQuestions', 1))
            difficulty = request.form.get('difficulty', 'medium')
            question_type = {
                "MCQ": "multiple-choice",
                "Fill-in-the-Blank": "fill-in-the-blank",
                "True-False": "true-false",
                "Short-Answer": "short-answer"
            }[action]
            result = generate_questions(text, api_key, num_questions, question_type, difficulty)

            # Update user stats
            user_id = session['user_id']
            stats = UserStats.query.filter_by(user_id=user_id).first()
            if not stats:
                stats = UserStats(user_id=user_id)
                db.session.add(stats)
            stats.total_questions += num_questions
            stats.total_difficulty += {"easy": 1, "medium": 2, "hard": 3}[difficulty] * num_questions
            db.session.commit()

        elif action == "QA":
            question = request.form.get('question')
            if not question:
                return jsonify({"error": "No question provided for QA"}), 400
            
            # Use Redis cache for embeddings
            cache_key = f"embeddings:{hash(text)}"
            cached_embeddings = app1.config['SESSION_REDIS'].get(cache_key)
            
            if cached_embeddings:
                vector_store = FAISS.deserialize_from_bytes(cached_embeddings)
            else:
                embeddings = OpenAIEmbeddings(api_key=api_key)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                texts = text_splitter.split_text(text)
                vector_store = FAISS.from_texts(texts, embeddings)
                app1.config['SESSION_REDIS'].set(cache_key, vector_store.serialize_to_bytes())
            
            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, api_key=api_key)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            answer = chain.invoke(question)
            result = answer['result']

        # Simulate a delay to show the notification feature
        socketio.emit('processing_complete', {'message': 'Question generation complete!'})

        return jsonify({
            "result": result,
            "token_count": tokens,
            "embedding_cost": tokens / 1000 * 0.0004
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app1.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    new_feedback = Feedback(
        question=data['question'],
        rating=data['rating'],
        comment=data.get('comment', '')
    )
    db.session.add(new_feedback)
    db.session.commit()
    return jsonify({"message": "Feedback submitted successfully"}), 200

@app1.route('/')
def home():
    return "Hello, World! The server is running."

if __name__ == "__main__":
    socketio.run(app1, debug=True, allow_unsafe_werkzeug=True)