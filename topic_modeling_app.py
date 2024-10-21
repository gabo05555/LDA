import pandas as pd
import numpy as np
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog,
                             QTextEdit, QVBoxLayout, QWidget, QLabel,
                             QPushButton, QMessageBox)
from gensim import corpora
from gensim.models import LdaModel
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import fitz  # PyMuPDF

# Ensure you download NLTK stopwords
nltk.download('stopwords')


class TopicModelingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data = None

    def initUI(self):
        self.setWindowTitle('Research Paper Topic Categorizer and Evolution')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        self.text_area = QTextEdit(self)
        self.text_area.setReadOnly(True)
        layout.addWidget(self.text_area)

        # Create buttons with custom styles
        self.load_csv_button = QPushButton('Load CSV', self)
        self.load_csv_button.setStyleSheet("QPushButton {"
                                    "border-radius: 15px;"
                                    "background-color: #81cdc6;"
                                    "padding: 10px;"
                                    "font-size: 16px;"
                                    "border: 1px solid gray;"  # Thin gray border
                                    "}"
                                    "QPushButton:hover {"
                                    "background-color: #a2e3d4;"
                                    "}")
        self.load_csv_button.clicked.connect(self.load_csv)
        layout.addWidget(self.load_csv_button)

        self.load_pdf_button = QPushButton('Load PDF', self)
        self.load_pdf_button.setStyleSheet("QPushButton {"
                                    "border-radius: 15px;"
                                    "padding: 10px;"
                                    "font-size: 16px;"
                                    "border: 1px solid gray;"  # Thin gray border
                                    "}"
                                    "QPushButton:hover {"
                                    "background-color: #a2e3d4;"
                                    "}")
        self.load_pdf_button.clicked.connect(self.load_pdf)
        layout.addWidget(self.load_pdf_button)

        self.analyze_button = QPushButton('Analyze Topics', self)
        self.analyze_button.setStyleSheet("QPushButton {"
                                    "border-radius: 15px;"
                                    "padding: 10px;"
                                    "font-size: 16px;"
                                    "border: 1px solid gray;"  # Thin gray border
                                    "}"
                                    "QPushButton:hover {"
                                    "background-color: #a2e3d4;"
                                    "}")
        self.analyze_button.clicked.connect(self.analyze_topics)
        layout.addWidget(self.analyze_button)

        self.visualize_button = QPushButton('Visualize Topic Evolution', self)
        self.visualize_button.setStyleSheet("QPushButton {"
                                     "border-radius: 15px;"
                                     "background-color: #81cdc6;"
                                     "padding: 10px;"
                                     "font-size: 16px;"
                                     "border: 1px solid gray;"  # Thin gray border
                                     "}"
                                     "QPushButton:hover {"
                                     "background-color: #a2e3d4;"
                                     "}")
        self.visualize_button.clicked.connect(self.visualize_topics)
        layout.addWidget(self.visualize_button)

        self.result_label = QLabel('', self)
        layout.addWidget(self.result_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        
    def load_csv(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load CSV File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if file_name:
            try:
                self.data = pd.read_csv(file_name)
                if 'abstract' in self.data.columns and 'title' in self.data.columns and 'year' in self.data.columns:
                    self.text_area.setPlainText(f"Loaded {len(self.data)} papers from CSV.")
                else:
                    QMessageBox.critical(self, "Error", "CSV must contain 'abstract', 'title', and 'year' columns.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")

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
            body = ""

        # Get the year from the filename or some other logic
        year = "2024"  # Default or extract from PDF metadata if available

        # Combine abstract and body for analysis
        self.data = pd.DataFrame({"title": ["PDF Paper"], "abstract": [abstract], "body": [body], "year": [year]})

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
            specific_keyword = 'science and technology'

            self.data['is_science_tech'] = self.data['processed_abstract'].apply(
                lambda x: any(keyword in x for keyword in science_keywords)
            )

            # Categorize articles by topics
            self.data['category'] = self.data['abstract'].apply(self.categorize_article)

            # Prepare output for titles, science/tech relevance, and category
            titles_output = "<b>Titles and Science/Tech Relevance:</b><br><br>"
            for index, row in self.data.iterrows():
                relevance = "Related to Science and Technology" if row['is_science_tech'] else "Not Related"
                category = row['category'] if row['category'] else "Uncategorized"
                titles_output += f"<b>Title:</b> {row['title']}<br>"
                titles_output += f" - {relevance} ({category})<br>"
                titles_output += f"<b>Abstract:</b> {row['abstract']}<br><br>"

            # Set output with HTML formatting
            self.text_area.setHtml(titles_output)
            self.result_label.setText(f"Total Papers Analyzed: {len(self.data)}")

    def categorize_article(self, abstract):
        """Categorize article topics based on keywords."""
        categories = {
           'Health and Medicine': [
            'health', 'medicine', 'disease', 'therapy', 'medical', 'hospital', 'diabetes', 'cancer',
            'treatment', 'nutrition', 'public health', 'clinical', 'pharmaceutical', 'wellness', 
            'symptom', 'diagnosis', 'patient', 'healthcare', 'chronic', 'mental health', 
            'preventive', 'vaccine', 'epidemiology', 'health policy', 'cardiology', 'neurology', 
            'endocrinology', 'pathology', 'immunology', 'surgery', 'rehabilitation', 'nursing', 
            'health technology', 'telemedicine'
        ],
        'Economics and Finance': [
            'economy', 'finance', 'business', 'market', 'trade', 'investment', 'stock', 'currency',
            'economic policy', 'inflation', 'banking', 'financial analysis', 'capital', 'assets', 
            'debt', 'wealth management', 'risk management', 'economic growth', 'employment', 
            'monetary policy', 'fiscal policy', 'microeconomics', 'macroeconomics', 'trade balance',
            'supply chain', 'consumer behavior', 'market research'
        ],
        'Education and Social Science': [
            'education', 'teaching', 'learning', 'psychology', 'sociology', 'community', 'curriculum',
            'pedagogy', 'student', 'educational technology', 'literacy', 'research methods', 
            'social behavior', 'cultural studies', 'developmental psychology', 'human behavior', 
            'public policy', 'social justice', 'inequality', 'community engagement', 'mental health', 
            'cognitive development', 'educational psychology', 'global education'
        ],
            
    
            'Physics and Mathematics': [
                'physics', 'mathematics', 'statistic', 'theorem', 'equation', 'geometry', 'calculus', 
            'linear algebra', 'probability', 'quantum', 'relativity', 'statistical', 'numerical', 
            'applied mathematics', 'differential equations', 'topology', 'mathematical modeling', 
            'complexity', 'theoretical physics', 'mechanics', 'astrophysics', 'thermodynamics', 
            'chaos theory', 'mathematical physics', 'combinatorics', 'graph theory', 
            'functional analysis'
            ]
        }

        # Science and Technology keywords are explicitly handled here
        science_keywords = ['science', 'technology', 'engineering', 'research', 'development', 'innovation','algorithm', 'data', 'network', 
                            'computing', 'software', 'hardware', 'machine learning', 'AI','artificial intelligence', 'robotics', 'optimization', 
                            'programming', 'information', 'database','web', 'internet', 'mobile', 'cybersecurity', 'signal processing', 'pattern recognition', 
                            'image processing','virtual reality', 'augmented reality', 'environment', 'climate', 'ecology', 'sustainability', 'pollution', 'biodiversity']

        abstract_lower = abstract.lower() if isinstance(abstract, str) else ""

        # Prioritize Science and Technology keywords
        if any(keyword in abstract_lower for keyword in science_keywords):
            return 'Science and Technology'

        # Check for keywords in each category
        for category, keywords in categories.items():
            if any(keyword in abstract_lower for keyword in keywords):
                return category

        return 'Uncategorized'  # Default if no specific category is matched

    def visualize_topics(self):
        if self.data is not None:
        # Handle NaN values in 'title' and 'abstract' columns
            self.data['abstract'] = self.data['abstract'].fillna('')  # Replace NaN in 'abstract' with empty string
            self.data['title'] = self.data['title'].fillna('')        # Replace NaN in 'title' with empty string

        # Combine title and abstract into a single text
            self.data['combined'] = self.data['title'] + " " + self.data['abstract']
        
        # Create year-wise frequency of topics
            year_topic_counts = {}
            for index, row in self.data.iterrows():
                year = row['year']
                if year not in year_topic_counts:
                    year_topic_counts[year] = {'total': 0, 'categories': {}}
                year_topic_counts[year]['total'] += 1
            
            # Categorize each article
                category = self.categorize_article(row['abstract'])
                if category not in year_topic_counts[year]['categories']:
                    year_topic_counts[year]['categories'][category] = 0
                year_topic_counts[year]['categories'][category] += 1
        
        # Prepare data for visualization
            year_labels = sorted(year_topic_counts.keys())
            total_counts = [year_topic_counts[year]['total'] for year in year_labels]
        
        # Determine maximum count and scaling factor
            max_count = max(total_counts)
            scaling_factor = 1_000_000 / max_count  # Scale so that the max is 1 million
        
        # Scale total counts for visualization
            total_counts_scaled = [count * scaling_factor for count in total_counts]

            category_counts = {cat: [0] * len(year_labels) for cat in year_topic_counts[year_labels[0]]['categories']}

        # Ensure all categories are accounted for
            all_categories = set()
            for year in year_labels:
                all_categories.update(year_topic_counts[year]['categories'].keys())

        # Initialize all categories in category_counts
            for cat in all_categories:
                category_counts[cat] = [0] * len(year_labels)

        # Populate category counts with scaling
            for i, year in enumerate(year_labels):
                for cat, count in year_topic_counts[year]['categories'].items():
                    category_counts[cat][i] = count * scaling_factor  # Scale by the scaling factor

        # Plot the results
            plt.figure(figsize=(12, 6))
            for cat, counts in category_counts.items():
                plt.plot(year_labels, counts, marker='o', label=cat)

        # Plot total counts as well
           

            plt.title('Topic Evolution Over Time (Number of Papers)')
            plt.xlabel('Year')
            plt.ylabel('Number of Papers')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            QMessageBox.critical(self, "Error", "No data available to visualize.")


if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    mainWin = TopicModelingApp()
    mainWin.show()
    sys.exit(app.exec_())
