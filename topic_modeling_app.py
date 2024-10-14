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

    def highlight_keywords(self, text, keywords, specific_keyword=None):
        """Highlight keywords in green and a specific keyword in yellow."""
        if not isinstance(text, str):
            text = ""  # Ensure text is a string

        # Highlight general science-related keywords in green
        for keyword in keywords:
            text = re.sub(f"\\b({keyword})\\b", r'<span style="background-color: #90EE90;">\1</span>', text, flags=re.IGNORECASE)

        # Highlight specific "science and technology" in yellow
        if specific_keyword:
            text = re.sub(f"\\b({specific_keyword})\\b", r'<span style="background-color: #FFFF00;">\1</span>', text, flags=re.IGNORECASE)
    
        return text

    def categorize_article(self, abstract):
        """Categorize article topics based on keywords."""
         # Define keywords for different categories
        categories = {
        'Health and Medicine': ['health', 'medicine', 'disease', 'therapy', 'medical', 'hospital', 'diabetes', 'cancer'],
        'Economics and Finance': ['economy', 'finance', 'business', 'market', 'trade', 'investment', 'stock'],
        'Education and Social Science': ['education', 'teaching', 'learning', 'psychology', 'sociology', 'community'],
        'Environmental Science': ['environment', 'climate', 'ecology', 'sustainability', 'pollution', 'biodiversity'],
        'Computer Science and Engineering': [
            'algorithm', 'data', 'network', 'computing', 'software', 'hardware', 'machine learning', 'AI',
            'artificial intelligence', 'robotics', 'optimization', 'programming', 'information', 'database',
            'web', 'internet', 'mobile', 'cybersecurity', 'signal processing', 'pattern recognition', 'image processing',
            'virtual reality', 'augmented reality'
        ],
        'Physics and Mathematics': [
            'physics', 'mathematics', 'statistic', 'theorem', 'equation', 'geometry', 'calculus', 'linear algebra',
            'probability', 'quantum', 'relativity', 'statistical', 'numerical'
        ]
    }

        # Science and Technology keywords are explicitly handled here
        science_keywords = ['science', 'technology', 'engineering', 'research', 'development', 'innovation']
    
        abstract_lower = abstract.lower() if isinstance(abstract, str) else ""

        # Check for keywords in each category
        for category, keywords in categories.items():
            if any(keyword in abstract_lower for keyword in keywords):
                return category

        return 'Uncategorized'  # Default if no specific category is matched


        # Prioritize Science and Technology keywords
        if any(keyword in abstract_lower for keyword in science_keywords):
            return 'Science and Technology'

        # Check for other categories
        for category, keywords in categories.items():
            if any(keyword in abstract_lower for keyword in keywords):
                return category

        return None  # Return None if no specific category is matched

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
            specific_keyword = 'science and technology'
            
            self.data['is_science_tech'] = self.data['processed_abstract'].apply(
                lambda x: any(keyword in x for keyword in science_keywords)
            )

            # Categorize articles by topics
            self.data['category'] = self.data['abstract'].apply(self.categorize_article)

            # Prepare output for titles, science/tech relevance, and category with highlighted keywords
            titles_output = "<b>Titles and Science/Tech Relevance:</b><br><br>"
            for index, row in self.data.iterrows():
                # Highlight science keywords and specific "science and technology"
                highlighted_abstract = self.highlight_keywords(row['abstract'], science_keywords, specific_keyword)
                titles_output += f"<b>Title:</b> {row['title']}<br>"
                relevance = "Related to Science and Technology" if row['is_science_tech'] else "Not Related"
                category = row['category'] if row['category'] else "Uncategorized"
                titles_output += f" - {relevance} ({category})<br>"
                titles_output += f"<b>Abstract:</b> {highlighted_abstract}<br><br>"

            # Set output with HTML formatting
            self.text_area.setHtml(titles_output)
            self.result_label.setText(f"Total Papers Analyzed: {len(self.data)}")


if __name__ == '__main__':
    app = QApplication([])
    window = TopicModelingApp()
    window.show()
    app.exec_()
