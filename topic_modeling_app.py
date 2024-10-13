import pandas as pd
import numpy as np
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QTextEdit, QVBoxLayout, QWidget, QLabel,
                             QPushButton)
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
import fitz  # PyMuPDF

# Ensure you download NLTK stopwords
nltk.download('stopwords')

class TopicModelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data = None

    def initUI(self):
        self.setWindowTitle('Research Paper Topic Categorizer')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        self.load_csv_button = QPushButton('Load CSV', self)
        self.load_csv_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_csv_button)

        self.load_pdf_button = QPushButton('Load PDF', self)
        self.load_pdf_button.clicked.connect(self.load_pdf)
        layout.addWidget(self.load_pdf_button)

        self.topic_button = QPushButton('Analyze Topics', self)
        self.topic_button.clicked.connect(self.analyze_topics)
        layout.addWidget(self.topic_button)

        self.result_label = QLabel('', self)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            self.data = pd.read_csv(file_name)
            self.text_area.setPlainText(f"Loaded {len(self.data)} papers from CSV.")

    def load_pdf(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load PDF File", "", "PDF Files (*.pdf);;All Files (*)", options=options)
        if file_name:
            self.load_pdf_data(file_name)
            self.text_area.setPlainText(f"Loaded PDF file: {file_name}.")

    def load_pdf_data(self, file_path):
        # Extract text from the PDF
        pdf_document = fitz.open(file_path)
        text = ""
        for page in pdf_document:
            text += page.get_text()
        pdf_document.close()

        # Create a DataFrame from the extracted text
        abstract_start = text.lower().find("abstract")
        introduction_start = text.lower().find("introduction")
        
        if abstract_start != -1 and introduction_start != -1:
            abstract = text[abstract_start:introduction_start].strip()
            body = text[introduction_start:].strip()
        else:
            abstract = text.strip()
            body = ""  # In case we don't find specific sections

        # Combine abstract and body for analysis
        self.data = pd.DataFrame({"title": ["PDF Paper"], "abstract": [abstract], "body": [body]})

    def preprocess_text(self, text):
        if isinstance(text, str):
            text = text.lower()
            text = re.sub(r'\W+', ' ', text)  # Remove punctuation
            return text
        return ""

    def analyze_topics(self):
        if self.data is not None:
            # Preprocess abstracts
            self.data['processed_abstract'] = self.data['abstract'].apply(self.preprocess_text)

            # If 'body' column exists, preprocess it as well
            if 'body' in self.data.columns:
                self.data['processed_body'] = self.data['body'].apply(self.preprocess_text)
                texts = self.data['processed_abstract'].tolist() + self.data['processed_body'].tolist()
            else:
                texts = self.data['processed_abstract'].tolist()  # Only use abstracts

            # Tokenize texts
            texts = [text.split() for text in texts]  

            # Create a dictionary and corpus for LDA
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]

            # Apply LDA
            num_topics = 5
            lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

            # Display topics
            topics = lda_model.print_topics(num_words=3)
            topics_output = ""
            for topic in topics:
                topics_output += f"Topic {topic[0]}: {topic[1]}\n"
            self.text_area.setPlainText(topics_output)

            # Check for science and technology relevance
            science_keywords = ['science', 'technology', 'engineering', 'research', 'development', 'innovation']
            self.data['is_science_tech'] = self.data['processed_abstract'].apply(
                lambda x: any(keyword in x for keyword in science_keywords)
            )

            # Prepare output for titles and science/tech relevance
            titles_output = "Titles and Science/Tech Relevance:\n\n"
            for index, row in self.data.iterrows():
                titles_output += f"Title: {row['title']}\n"
                relevance = "Related to Science and Technology" if row['is_science_tech'] else "Not Related"
                titles_output += f" - {relevance}\n\n"

            self.text_area.setPlainText(titles_output)
            self.result_label.setText(f"Total Papers Analyzed: {len(self.data)}")

if __name__ == '__main__':
    app = QApplication([])
    window = TopicModelingApp()
    window.show()
    app.exec_()
