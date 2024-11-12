from flask import Flask, request, jsonify
from flask_cors import CORS
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
from flask_mail import Mail, Message
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///feedback.db'
db = SQLAlchemy(app)

load_dotenv()

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER')

mail = Mail(app)

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.now())

with app.app_context():
    db.create_all()

def create_pdf(content, cost_details):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    elements.append(Paragraph("Generated Questions Report", title_style))
    elements.append(Spacer(1, 20))
    
    cost_data = [
        ['Cost Type', 'Amount'],
        ['Token Count', str(cost_details['token_count'])],
        ['Embedding Cost', f"${cost_details['embedding_cost']:.4f}"],
        ['Total Cost', f"${cost_details['total_cost']:.4f}"]
    ]
    
    cost_table = Table(cost_data, colWidths=[200, 100])
    cost_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    elements.append(cost_table)
    elements.append(Spacer(1, 30))
    
    content_style = ParagraphStyle(
        'ContentStyle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=12
    )
    elements.append(Paragraph("Generated Questions:", content_style))
    elements.append(Spacer(1, 10))
    
    # Format the content
    if isinstance(content, str):
        questions = content.split('\n\n')
    else:
        questions = [str(item) for item in content]
        
    for question in questions:
        if question.strip():
            elements.append(Paragraph(question.replace('\n', '<br/>'), content_style))
            elements.append(Spacer(1, 10))
    
    doc.build (elements)
    buffer.seek(0)
    return buffer

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    return " ".join(page.extract_text() for page in pdf_reader.pages)

def count_tokens_accurate(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def generate_questions(text_chunk: str, api_key: str, num_questions: int = None, question_type: str = "multiple-choice", difficulty: str = None, taxonomy: str = None) -> str:
    client = openai.OpenAI(api_key=api_key)
    
    taxonomy_descriptions = {
        "remember": "Test recall of specific facts, terms, concepts, or basic information. Questions should focus on memorization and recognition.",
        "understand": "Test comprehension and interpretation. Questions should assess ability to explain ideas or concepts in their own words.",
        "apply": "Test ability to use information in new situations. Questions should focus on applying learned material to new scenarios.",
        "analyze": "Test ability to break information into parts and understand relationships. Questions should focus on drawing connections and finding patterns.",
        "evaluate": "Test ability to justify a stand or decision. Questions should focus on making judgments based on criteria and standards.",
        "create": "Test ability to produce new or original work. Questions should focus on combining elements to form something new."
    }
    
    system_prompt = (
        "You are a precise and accurate educational question generator. "
        f"Generate questions at the {taxonomy.upper()} level of Bloom's Taxonomy. "
        f"{taxonomy_descriptions.get(taxonomy, '')}\n"
        "When creating questions:\n"
        "1. Ensure questions match the specified cognitive level\n"
        "2. Make answers unambiguous and clearly correct\n"
        "3. Base all questions directly on the provided text\n"
        "4. Align question complexity with both difficulty level and taxonomy level\n"
        "5. Use appropriate verbs and question stems for the taxonomy level\n"
    )
    
    valid_question_types = ["multiple-choice", "fill-in-the-blank", "true-false", "short-answer", "Summary", "KeyPoints"]
    if question_type not in valid_question_types:
        raise ValueError(f"Invalid question type. Must be one of: {', '.join(valid_question_types)}")
    
    # Taxonomy-specific question stems
    taxonomy_stems = {
        "remember": ["Define", "List", "Recall", "Name", "Identify", "State"],
        "understand": ["Explain", "Describe", "Discuss", "Interpret", "Summarize"],
        "apply": ["Demonstrate", "Use", "Solve", "Implement", "Show how"],
        "analyze": ["Compare", "Contrast", "Examine", "Differentiate", "Analyze"],
        "evaluate": ["Judge", "Justify", "Critique", "Defend", "Evaluate"],
        "create": ["Design", "Develop", "Compose", "Propose", "Construct"]
    }
    
    if question_type in ["multiple-choice", "fill-in-the-blank", "true-false", "short-answer"]:
        if num_questions is None or difficulty is None or taxonomy is None:
            raise ValueError("num_questions, difficulty, and taxonomy are required for question generation types")
        
        user_prompt = (
            f"Create {num_questions} {question_type} questions at the {taxonomy.upper()} level "
            f"of Bloom's Taxonomy. Use these question stems as appropriate: {', '.join(taxonomy_stems[taxonomy])}. "
            f"The difficulty level should be {difficulty}. "
            f"Questions should clearly demonstrate {taxonomy}-level thinking.\n\n"
            f"Text to base questions on:\n{text_chunk}\n\n"
        )
    else:
        user_prompt = f"{text_chunk}\n\n"
    
    if question_type == "multiple-choice":
        user_prompt += (
            "Format each question as follows:\n\n"
            "1. [Question text using appropriate taxonomy-level stem]\n\n"
            "[A] Option A\n\n"
            "[B] Option B\n\n"
            "[C] Option C\n\n"
            "[D] Option D\n\n"
            "Correct answer: [Correct option - answer]\n\n"
            "Ensure there is an empty line between each question for readability.\n"
        )
    elif question_type == "fill-in-the-blank":
        user_prompt += (
            "Format each question as:\n\n"
            "Q. [Sentence with _____ for blank]\n"
            "On the next line, provide a brief explanation or hint (if applicable) to give additional context or learning value.\n\n"
            "Example:\n\n"
            "1. The capital of France is ____.\n\n"
            "   (Hint: It's known as the City of Light.)\n\n"
            "   *Answer:* Paris \n\n"
            "2. Water freezes at ____ degrees Celsius.\n\n"
            "   *(Hint: This is the temperature at which liquid water turns into ice at standard atmospheric pressure.)*\n\n"
            "   **Answer:** 0 \n\n"
            
            "Answer: [Exact word or phrase]\n\n"
        )
    elif question_type == "true-false":
        user_prompt += (
            f"Create {num_questions} high-quality True/False questions based on the provided content. Questions should thoroughly test understanding of key concepts and details from the text. Use the following format:\n\n"
            "Format for each question:\n\n"
            "Q[number]. [Clear, specific statement based on the text]\n\n"
            "Answer: [True/False]\n\n"
            "Explanation: [Brief explanation referencing the specific part of the text]\n\n"
            
            "Guidelines for creating effective True/False questions:\n\n"
            "1. Focus on significant concepts and important details from the text\n\n"
            "2. Avoid obvious or trivial statements\n\n"
            "3. Test both explicit information and implied relationships\n\n"
            "4. Use precise language to avoid ambiguity\n\n"
            "5. Include a mix of true and false statements\n\n"
            "6. For false statements:\n\n"
            "   - Make subtle but clear modifications to true facts\n\n"
            "   - Avoid obviously false statements\n\n"
            "   - Change only one aspect of a true statement\n\n"
            "7. For statements testing relationships:\n\n"
            "   - Focus on cause-effect relationships\n\n"
            "   - Test understanding of sequential events\n\n"
            "   - Examine connections between concepts\n\n"
            
            "Example format:\n\n"
            "Q1. [A specific, clear statement that tests understanding of a key concept from the text]\n\n"
            "Answer: True\n\n"
            "Explanation: According to paragraph 2, [reference to specific text content]\n\n"
            
            "Q2. [A statement that subtly modifies a fact from the text]\n\n"
            "Answer: False\n\n"
            "Explanation: The text actually states that [correct information with reference]\n\n"
            
            "Requirements:\n\n"
            "- Each question must be answerable solely from the provided text\n\n"
            "- Maintain approximately 50% true and 50% false statements\n\n"
            "- Include direct references to the text in explanations\n\n"
            "- Focus on testing comprehension, not just recall\n\n"
            "- Ensure statements are specific and unambiguous\n\n"
            f"- Match the specified difficulty level: {difficulty}\n\n"
            
            "Now, generate thought-provoking True/False questions based on the provided content, following these guidelines."
        )
    elif question_type == "short-answer":
        user_prompt += (
            "Format each question as:\n\n"
            "Q. [Question text]\n"
            "Answer: [Concise answer directly from the text]\n\n"
        )
    elif question_type == "Summary":
        user_prompt = (
            "Please provide a comprehensive summary of the following text. "
            "The summary should be well-structured, clear, and capture the main ideas "
            "and important details from the text:\n\n"
            f"{text_chunk}\n\n"
            "Format the summary in clear paragraphs with proper spacing."
        )
    elif question_type == "KeyPoints":
        user_prompt = (
            "Please extract the key points from the following text. "
            "Identify the most important concepts, facts, and ideas. "
            "Format the output as a bullet-point list with clear, concise points:\n\n"
            f"{text_chunk}\n\n"
            "Format as:\n\n"
            "• Key Point 1\n\n"
            "• Key Point 2\n\n"
            "etc."
        )
            
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=1500,
        n=1,
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

@app.route('/process', methods=['POST'])
def process():
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
            taxonomy = request.form.get('taxonomy', 'understand')  # Default to understand level
            question_type = {
                "MCQ": "multiple-choice",
                "Fill-in-the-Blank": "fill-in-the-blank",
                "True-False": "true-false",
                "Short-Answer": "short-answer"
            }[action]
            result = generate_questions(text, api_key, num_questions, question_type, difficulty, taxonomy)
        elif action == "QA":
            question = request.form.get('question')
            if not question:
                return jsonify({"error": "No question provided for QA"}), 400
            
            embeddings = OpenAIEmbeddings(api_key=api_key)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(text)
            vector_store = FAISS.from_texts(texts, embeddings)
            
            llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1, api_key=api_key)
            retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={'k': 3})
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
            answer = chain.invoke(question)
            result = answer['result']
        elif action in ["Summary", "KeyPoints"]:
            result = generate_questions(text, api_key, question_type=action)
        else:
            return jsonify({"error": "Invalid action"}), 400
        
        socketio.emit('processing_complete', {'message': 'Question generation complete!'})

        total_cost = tokens / 1000 * 0.0004
        return jsonify({
            "result": result,
            "token_count": tokens,
            "embedding_cost": total_cost,
            "total_cost": total_cost
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/send-pdf', methods=['POST'])
def send_pdf():
    try:
        data = request.get_json()
        print("Received data:", data) 
        email = data.get('email')
        content = data.get('content')
        cost_details = {
            'token_count': data.get('token_count'),
            'embedding_cost': data.get('embedding_cost', 0),
            'total_cost': data.get('total_cost', 0)
        }
        print("Cost details:", cost_details) 
        
        if not email or not content:
            return jsonify({'error': 'Email and content are required'}), 400
            
        
        pdf_buffer = create_pdf(content, cost_details)
        
        msg = Message(
            subject='Your Generated Questions PDF',
            recipients=[email],
            body=f"""Hello,User here is your Content which is generated by our "Automation Question Builder"

Please find attached your generated questions and cost details.

Thank you for using our service!

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
        )
        
        msg.attach(
            filename='generated_questions.pdf',
            content_type='application/pdf',
            data=pdf_buffer.getvalue()
        )
        
        mail.send(msg)
        
        return jsonify({
            'message': 'PDF has been sent successfully',
            'status': 'success'
        }), 200
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/ feedback', methods=['POST'])
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

@app.route('/')
def home():
    return "Hello, World! The server is running."

if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)