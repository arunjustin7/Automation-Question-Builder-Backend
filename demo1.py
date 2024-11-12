# from flask import Flask, request, jsonify, send_file
# from flask_cors import CORS
# import os
# from dotenv import load_dotenv
# import PyPDF2
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.chains import RetrievalQA
# import tiktoken
# import openai
# from flask_socketio import SocketIO
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime
# import json
# from io import BytesIO
# from reportlab.pdfgen import canvas
# from docx import Document

# app = Flask(__name__)
# CORS(app)
# socketio = SocketIO(app, cors_allowed_origins="*")
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///questionbank.db'
# db = SQLAlchemy(app)

# # Enhanced Database Models
# class User(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     username = db.Column(db.String(80), unique=True, nullable=False)
#     email = db.Column(db.String(120), unique=True, nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

# class Template(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100), nullable=False)
#     content = db.Column(db.Text, nullable=False)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

# class QuestionBank(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     title = db.Column(db.String(200), nullable=False)
#     questions = db.Column(db.Text, nullable=False)
#     question_type = db.Column(db.String(50))
#     difficulty = db.Column(db.String(20))
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)
#     user_id = db.Column(db.Integer, db.ForeignKey('user.id'))

# class Analytics(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     question_type = db.Column(db.String(50))
#     difficulty = db.Column(db.String(20))
#     generation_time = db.Column(db.Float)
#     token_count = db.Column(db.Integer)
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)

# # Create all database tables
# with app.app_context():
#     db.create_all()

# # New route for templates
# @app.route('/templates', methods=['GET', 'POST'])
# def manage_templates():
#     if request.method == 'POST':
#         data = request.json
#         new_template = Template(
#             name=data['name'],
#             content=data['content'],
#             user_id=data.get('user_id', 1)  # Default user_id for now
#         )
#         db.session.add(new_template)
#         db.session.commit()
#         return jsonify({"message": "Template saved successfully"})
#     else:
#         templates = Template.query.all()
#         return jsonify([{
#             "id": t.id,
#             "name": t.name,
#             "content": t.content
#         } for t in templates])

# # Analytics endpoints
# @app.route('/analytics', methods=['GET'])
# def get_analytics():
#     analytics = Analytics.query.all()
    
#     # Process analytics data
#     question_types = {}
#     difficulties = {}
#     avg_generation_time = 0
#     total_questions = len(analytics)
    
#     for record in analytics:
#         # Count question types
#         question_types[record.question_type] = question_types.get(record.question_type, 0) + 1
#         # Count difficulties
#         difficulties[record.difficulty] = difficulties.get(record.difficulty, 0) + 1
#         # Sum generation times
#         avg_generation_time += record.generation_time
    
#     if total_questions > 0:
#         avg_generation_time /= total_questions
    
#     return jsonify({
#         "total_questions": total_questions,
#         "question_types": question_types,
#         "difficulties": difficulties,
#         "avg_generation_time": avg_generation_time
#     })

# # Export functionality
# @app.route('/export', methods=['POST'])
# def export_questions():
#     data = request.json
#     questions = data['questions']
#     format_type = data['format']
    
#     if format_type == 'pdf':
#         # Create PDF
#         buffer = BytesIO()
#         p = canvas.Canvas(buffer)
#         y = 800
#         for idx, question in enumerate(questions.split('\n\n'), 1):
#             p.drawString(72, y, f"{idx}. {question}")
#             y -= 20
#         p.save()
#         buffer.seek(0)
#         return send_file(
#             buffer,
#             as_attachment=True,
#             download_name='questions.pdf',
#             mimetype='application/pdf'
#         )
    
#     elif format_type == 'docx':
#         # Create Word document
#         doc = Document()
#         doc.add_heading('Generated Questions', 0)
#         for question in questions.split('\n\n'):
#             doc.add_paragraph(question)
        
#         buffer = BytesIO()
#         doc.save(buffer)
#         buffer.seek(0)
#         return send_file(
#             buffer,
#             as_attachment=True,
#             download_name='questions.docx',
#             mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
#         )

# # Enhanced process route with analytics tracking
# @app.route('/process', methods=['POST'])
# def process():
#     start_time = datetime.now()
    
#     if 'file' not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     file = request.files['file']
#     if file.filename == '':
#         return jsonify({"error": "No selected file"}), 400

#     action = request.form.get('action')
#     if not action:
#         return jsonify({"error": "No action specified"}), 400

#     try:
#         text = extract_text_from_pdf(file)
#         tokens = count_tokens_accurate(text)

#         api_key = os.getenv("api_key")
#         if not api_key:
#             return jsonify({"error": "OpenAI API key not found"}), 500

#         result = None
#         if action in ["MCQ", "Fill-in-the-Blank", "True-False", "Short-Answer"]:
#             num_questions = int(request.form.get('numQuestions', 1))
#             difficulty = request.form.get('difficulty', 'medium')
#             question_type = {
#                 "MCQ": "multiple-choice",
#                 "Fill-in-the-Blank": "fill-in-the-blank",
#                 "True-False": "true-false",
#                 "Short-Answer": "short-answer"
#             }[action]
#             result = generate_questions(text, api_key, num_questions, question_type, difficulty)
            
#             # Save to question bank
#             question_bank = QuestionBank(
#                 title=f"{action} Questions - {datetime.now()}",
#                 questions=result,
#                 question_type=action,
#                 difficulty=difficulty
#             )
#             db.session.add(question_bank)
            
#             # Track analytics
#             generation_time = (datetime.now() - start_time).total_seconds()
#             analytics = Analytics(
#                 question_type=action,
#                 difficulty=difficulty,
#                 generation_time=generation_time,
#                 token_count=tokens
#             )
#             db.session.add(analytics)
#             db.session.commit()
            
#         elif action == "QA":
#             question = request.form.get('question')
#             if not question:
#                 return jsonify({"error": "No question provided for QA"}), 400
            
#             result = handle_qa(text, question, api_key)

#         socketio.emit('processing_complete', {'message': 'Question generation complete!'})

#         return jsonify({
#             "result": result,
#             "token_count": tokens,
#             "embedding_cost": tokens / 1000 * 0.0004,
#             "generation_time": (datetime.now() - start_time).total_seconds()
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# if __name__ == "__main__":
#     socketio.run(app, debug=True, allow_unsafe_werkzeug=True) 