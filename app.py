import streamlit as st 
import nltk
import spacy
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
spacy.load('en_core_web_sm')
import pandas as pd
import base64
import time
import datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import PDFPageAggregator
from pdfminer3.converter import TextConverter
import io
import random
from streamlit_tags import st_tags
from PIL import Image
import pymysql
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

import pafy
import plotly.express as px
from Courses import ds_course,web_course,android_course,ios_course,uiux_course, resume_videos, interview_videos

# Initialize ML components
def initialize_ml_components():
    """Initialize machine learning components for resume scoring"""
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Initialize TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', lowercase=True)
    
    return lemmatizer, stop_words, vectorizer

# Text preprocessing function
def preprocess_text(text, lemmatizer, stop_words):
    """Preprocess text for ML analysis"""
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    processed_tokens = [lemmatizer.lemmatize(token) for token in tokens 
                       if token not in stop_words and len(token) > 2]
    
    return ' '.join(processed_tokens)

# Extract features from resume text
def extract_resume_features(resume_text, resume_data):
    """Extract comprehensive features from resume for ML scoring"""
    features = {}
    
    # Basic features
    features['word_count'] = len(resume_text.split())
    features['char_count'] = len(resume_text)
    features['sentence_count'] = len(resume_text.split('.'))
    features['paragraph_count'] = len(resume_text.split('\n\n'))
    features['page_count'] = resume_data.get('no_of_pages', 1)
    
    # Skills features
    skills = resume_data.get('skills', [])
    features['skill_count'] = len(skills)
    features['unique_skills'] = len(set([skill.lower() for skill in skills]))
    
    # Education features
    education = resume_data.get('degree', [])
    features['education_count'] = len(education) if education else 0
    
    # Experience features
    experience = resume_data.get('experience', [])
    features['experience_count'] = len(experience) if experience else 0
    
    # Section presence features (binary)
    sections = ['objective', 'summary', 'education', 'experience', 'skills', 
               'projects', 'achievements', 'certifications', 'awards', 'hobbies',
               'interests', 'declaration', 'references']
    
    for section in sections:
        features[f'has_{section}'] = 1 if section.lower() in resume_text.lower() else 0
    
    # Contact information completeness
    features['has_email'] = 1 if resume_data.get('email') else 0
    features['has_phone'] = 1 if resume_data.get('mobile_number') else 0
    
    # Text quality features
    features['avg_word_length'] = np.mean([len(word) for word in resume_text.split()])
    features['capitalization_ratio'] = sum(1 for c in resume_text if c.isupper()) / len(resume_text)
    
    return features

# Calculate skill relevance score
def calculate_skill_relevance(candidate_skills, field_keywords):
    """Calculate how relevant candidate skills are to the predicted field"""
    if not candidate_skills:
        return 0
    
    candidate_skills_lower = [skill.lower() for skill in candidate_skills]
    field_keywords_lower = [keyword.lower() for keyword in field_keywords]
    
    # Calculate intersection
    relevant_skills = set(candidate_skills_lower).intersection(set(field_keywords_lower))
    
    # Calculate relevance score
    relevance_score = len(relevant_skills) / len(field_keywords_lower) * 100
    return min(relevance_score, 100)  # Cap at 100

# ML-based resume scoring function
def ml_resume_scoring(resume_text, resume_data, predicted_field):
    """Advanced ML-based resume scoring"""
    lemmatizer, stop_words, vectorizer = initialize_ml_components()
    
    # Extract features
    features = extract_resume_features(resume_text, resume_data)
    
    # Define field-specific keywords for relevance scoring
    field_keywords = {
        'Data Science': ['python', 'machine learning', 'data analysis', 'statistics', 'sql', 'pandas', 
                        'numpy', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'visualization'],
        'Web Development': ['html', 'css', 'javascript', 'react', 'angular', 'vue', 'node.js', 'express',
                           'django', 'flask', 'php', 'mysql', 'mongodb', 'rest api'],
        'Android Development': ['java', 'kotlin', 'android studio', 'xml', 'sqlite', 'firebase',
                               'gradle', 'mvvm', 'retrofit', 'room database'],
        'IOS Development': ['swift', 'objective-c', 'xcode', 'ios sdk', 'cocoa touch', 'core data',
                           'storyboard', 'auto layout', 'mvvm', 'combine'],
        'UI-UX Development': ['figma', 'adobe xd', 'sketch', 'prototyping', 'wireframing', 'user research',
                             'usability testing', 'design systems', 'interaction design']
    }
    
    # Calculate component scores
    scores = {}
    
    # 1. Content Quality Score (25%)
    content_score = 0
    if features['word_count'] > 200:
        content_score += 20
    if features['word_count'] > 500:
        content_score += 10
    if features['sentence_count'] > 10:
        content_score += 15
    if features['avg_word_length'] > 4:
        content_score += 5
    
    scores['content_quality'] = min(content_score, 25)
    
    # 2. Section Completeness Score (25%)
    essential_sections = ['has_objective', 'has_education', 'has_experience', 'has_skills']
    bonus_sections = ['has_projects', 'has_achievements', 'has_certifications']
    
    section_score = sum(features[section] for section in essential_sections) * 5
    section_score += sum(features[section] for section in bonus_sections) * 3
    
    scores['section_completeness'] = min(section_score, 25)
    
    # 3. Skills Relevance Score (25%)
    if predicted_field in field_keywords:
        skills_relevance = calculate_skill_relevance(
            resume_data.get('skills', []), 
            field_keywords[predicted_field]
        )
        scores['skills_relevance'] = min(skills_relevance * 0.25, 25)
    else:
        scores['skills_relevance'] = min(features['skill_count'] * 2, 25)
    
    # 4. Professional Presentation Score (25%)
    presentation_score = 0
    if features['has_email'] and features['has_phone']:
        presentation_score += 8
    if features['page_count'] >= 1 and features['page_count'] <= 3:
        presentation_score += 7
    if features['capitalization_ratio'] > 0.02 and features['capitalization_ratio'] < 0.1:
        presentation_score += 5
    if features['has_declaration']:
        presentation_score += 5
    
    scores['professional_presentation'] = min(presentation_score, 25)
    
    # Calculate total score
    total_score = sum(scores.values())
    
    return total_score, scores

# Enhanced resume analysis with ML insights
def get_resume_insights(resume_text, resume_data, predicted_field):
    """Get detailed insights about the resume"""
    insights = []
    
    # Analyze resume structure
    word_count = len(resume_text.split())
    if word_count < 200:
        insights.append("❌ Your resume seems too brief. Consider adding more details about your experience and achievements.")
    elif word_count > 1000:
        insights.append("⚠️ Your resume might be too lengthy. Consider condensing information to key highlights.")
    else:
        insights.append("✅ Your resume has an appropriate length.")
    
    # Analyze skills
    skills = resume_data.get('skills', [])
    if len(skills) < 5:
        insights.append("❌ Consider adding more technical skills relevant to your field.")
    elif len(skills) > 15:
        insights.append("⚠️ You have many skills listed. Focus on the most relevant ones for better impact.")
    else:
        insights.append("✅ You have a good balance of skills listed.")
    
    # Analyze contact information
    if not resume_data.get('email'):
        insights.append("❌ Email address is missing. This is essential for recruiters to contact you.")
    if not resume_data.get('mobile_number'):
        insights.append("❌ Phone number is missing. Consider adding it for better accessibility.")
    
    # Analyze sections
    important_sections = ['objective', 'projects', 'achievements', 'certifications']
    missing_sections = [section for section in important_sections 
                       if section.lower() not in resume_text.lower()]
    
    if missing_sections:
        insights.append(f"⚠️ Consider adding these sections: {', '.join(missing_sections)}")
    
    return insights

# Function to generate improvement recommendations
def generate_improvement_recommendations(scores, predicted_field):
    """Generate specific recommendations based on ML analysis"""
    recommendations = []
    
    if scores['content_quality'] < 15:
        recommendations.append("📝 **Content Quality**: Add more detailed descriptions of your work experience, achievements, and projects.")
    
    if scores['section_completeness'] < 15:
        recommendations.append("📋 **Section Completeness**: Include essential sections like Objective, Education, Experience, and Skills.")
    
    if scores['skills_relevance'] < 15:
        recommendations.append(f"🎯 **Skills Relevance**: Add more skills relevant to {predicted_field} to improve your profile match.")
    
    if scores['professional_presentation'] < 15:
        recommendations.append("✨ **Professional Presentation**: Ensure proper contact information, appropriate length, and professional formatting.")
    
    return recommendations

# Function to fetch YouTube video title
def fetch_yt_video(link):
    video = pafy.new(link)
    return video.title

# Function to generate download link for a DataFrame
def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Function to read PDF and extract text
def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
        text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

# Function to display PDF in Streamlit
def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# Function to recommend courses
def course_recommender(course_list):
    st.subheader("**Courses & Certificates🎓 Recommendations**")
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 4)
    random.shuffle(course_list)
    for idx, (c_name, c_link) in enumerate(course_list[:no_of_reco]):
        st.markdown(f"({idx+1}) [{c_name}]({c_link})")
        rec_course.append(c_name)
    return rec_course

# Database connection
connection = pymysql.connect(host="localhost", user="root1", password="Bharu@1234", database="resume")
cursor = connection.cursor()

# Function to insert data into the database
def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    DB_table_name = 'resume_data'
    insert_sql = f"""
    INSERT INTO {DB_table_name}
    (ID, Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses)
    VALUES (0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    rec_values = (name, email, str(res_score), timestamp, str(no_of_pages), reco_field, cand_level, skills, recommended_skills, courses)
    cursor.execute(insert_sql, rec_values)
    connection.commit()

# Streamlit page configuration
st.set_page_config(page_title="Smart Resume Analyzer", page_icon='logo.jpg')

def run():
    st.title("Smart Resume Analyser with ML-Based Scoring")
    st.sidebar.markdown("# Choose User")
    activities = ["Normal User", "Admin"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    # Create database and table if not exists
    cursor.execute("CREATE DATABASE IF NOT EXISTS resume")
    connection.select_db("resume")
    DB_table_name = 'resume_data'
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {DB_table_name} (
            ID INT NOT NULL AUTO_INCREMENT,
            Name VARCHAR(150) NOT NULL,
            Email_ID VARCHAR(100) NOT NULL,
            resume_score VARCHAR(8) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_no VARCHAR(5) NOT NULL,
            Predicted_Field VARCHAR(25) NOT NULL,
            User_level VARCHAR(30) NOT NULL,
            Actual_skills TEXT NOT NULL,
            Recommended_skills TEXT NOT NULL,
            Recommended_courses TEXT NOT NULL,
            PRIMARY KEY (ID)
        )
    """)

    if choice == 'Normal User':
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file:
            save_image_path = f'./Uploaded_Resumes/{pdf_file.name}'
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                resume_text = pdf_reader(save_image_path)
                st.header("**Resume Analysis**")
                st.success(f"Hello {resume_data.get('name')}")
                st.subheader("**Your Basic info**")
                st.text(f"Name: {resume_data.get('name')}")
                st.text(f"Email: {resume_data.get('email')}")
                st.text(f"Contact: {resume_data.get('mobile_number')}")
                st.text(f"Resume pages: {resume_data.get('no_of_pages')}")
                
                # Determine candidate level
                no_of_pages = resume_data.get('no_of_pages', 0)
                if no_of_pages == 1:
                    cand_level = "Fresher"
                    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>You are looking Fresher.</h4>''', unsafe_allow_html=True)
                elif no_of_pages == 2:
                    cand_level = "Intermediate"
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>You are at intermediate level!</h4>''', unsafe_allow_html=True)
                elif no_of_pages >= 3:
                    cand_level = "Experienced"
                    st.markdown('''<h4 style='text-align: left; color: #fba171;'>You are at experience level!</h4>''', unsafe_allow_html=True)

                st.subheader("**Skills Recommendation💡**")
                keywords = st_tags(label='### Skills that you have', text='See our skills recommendation', value=resume_data.get('skills', []), key='1')

                # Skill recommendations
                ds_keyword = ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep Learning', 'flask', 'streamlit']
                web_keyword = ['react', 'django', 'node jS', 'react js', 'php', 'laravel', 'magento', 'wordpress', 'javascript', 'angular js', 'c#', 'flask']
                android_keyword = ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy']
                ios_keyword = ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode']
                uiux_keyword = ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 'wireframes', 'storyframes', 'adobe photoshop', 'photoshop', 'editing', 'adobe illustrator', 'illustrator', 'adobe after effects', 'after effects', 'adobe premier pro', 'premier pro', 'adobe indesign', 'indesign', 'wireframe', 'solid', 'grasp', 'user research', 'user experience']

                recommended_skills = []
                reco_field = ''
                rec_course = ''
                
                # Field prediction logic
                for i in resume_data['skills']:
                    if i.lower() in ds_keyword:
                        reco_field = 'Data Science'
                        st.success("** Our analysis says you are looking for Data Science Jobs.**")
                        recommended_skills = ['Data Visualization', 'Predictive Analysis', 'Statistical Modeling',
                                              'Data Mining', 'Clustering & Classification', 'Data Analytics',
                                              'Quantitative Analysis', 'Web Scraping', 'ML Algorithms', 'Keras',
                                              'Pytorch', 'Probability', 'Scikit-learn', 'Tensorflow', "Flask",
                                              'Streamlit']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='2')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(ds_course)
                        break
                    elif i.lower() in web_keyword:
                        reco_field = 'Web Development'
                        st.success("** Our analysis says you are looking for Web Development Jobs **")
                        recommended_skills = ['React', 'Django', 'Node JS', 'React JS', 'php', 'laravel', 'Magento',
                                              'wordpress', 'Javascript', 'Angular JS', 'c#', 'Flask', 'SDK']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='3')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(web_course)
                        break
                    elif i.lower() in android_keyword:
                        reco_field = 'Android Development'
                        st.success("** Our analysis says you are looking for Android App Development Jobs **")
                        recommended_skills = ['Android', 'Android development', 'Flutter', 'Kotlin', 'XML', 'Java',
                                              'Kivy', 'GIT', 'SDK', 'SQLite']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='4')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(android_course)
                        break
                    elif i.lower() in ios_keyword:
                        reco_field = 'IOS Development'
                        st.success("** Our analysis says you are looking for IOS App Development Jobs **")
                        recommended_skills = ['IOS', 'IOS Development', 'Swift', 'Cocoa', 'Cocoa Touch', 'Xcode',
                                              'Objective-C', 'SQLite', 'Plist', 'StoreKit', "UI-Kit", 'AV Foundation',
                                              'Auto-Layout']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='5')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(ios_course)
                        break
                    elif i.lower() in uiux_keyword:
                        reco_field = 'UI-UX Development'
                        st.success("** Our analysis says you are looking for UI-UX Development Jobs **")
                        recommended_skills = ['UI', 'User Experience', 'Adobe XD', 'Figma', 'Zeplin', 'Balsamiq',
                                              'Prototyping', 'Wireframes', 'Storyframes', 'Adobe Photoshop', 'Editing',
                                              'Illustrator', 'After Effects', 'Premier Pro', 'Indesign', 'Wireframe',
                                              'Solid', 'Grasp', 'User Research']
                        recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                       text='Recommended skills generated from System',
                                                       value=recommended_skills, key='6')
                        st.markdown(
                            '''<h4 style='text-align: left; color: #1ed760;'>Adding this skills to resume will boost🚀 the chances of getting a Job💼</h4>''',
                            unsafe_allow_html=True)
                        rec_course = course_recommender(uiux_course)
                        break

                # ML-based Resume Scoring
                st.subheader("**🤖 AI-Powered Resume Score & Analysis**")
                
                # Get ML-based score
                ml_score, score_breakdown = ml_resume_scoring(resume_text, resume_data, reco_field)
                
                # Display score breakdown
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("**Overall Score**", f"{ml_score:.1f}/100")
                    
                    # Progress bar for overall score
                    st.markdown(
                        """
                        <style>
                            .stProgress > div > div > div > div {
                                background-color: #1ed760;
                            }
                        </style>""",
                        unsafe_allow_html=True,
                    )
                    progress_bar = st.progress(0)
                    for i in range(int(ml_score)):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                
                with col2:
                    st.write("**Score Breakdown:**")
                    for category, score in score_breakdown.items():
                        st.write(f"• {category.replace('_', ' ').title()}: {score:.1f}/25")
                
                # Display insights
                st.subheader("**📊 Resume Insights**")
                insights = get_resume_insights(resume_text, resume_data, reco_field)
                for insight in insights:
                    st.write(insight)
                
                # Display recommendations
                st.subheader("**🎯 Improvement Recommendations**")
                recommendations = generate_improvement_recommendations(score_breakdown, reco_field)
                for rec in recommendations:
                    st.write(rec)
                
                # Timestamp
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date + '_' + cur_time)

                # Insert data into database
                insert_data(resume_data['name'], resume_data['email'], str(int(ml_score)), timestamp,
                            str(resume_data['no_of_pages']), reco_field, cand_level, str(resume_data['skills']),
                            str(recommended_skills), str(rec_course))

                st.balloons()
                connection.commit()
            else:
                st.error('Something went wrong..')
    else:
        ## Admin Side
        st.success('Welcome to Admin Side')
        
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'bharathi' and ad_password == '123':
                st.success("Welcome bharathi")
                # Display Data
                cursor.execute('''SELECT*FROM resume_data''')
                data = cursor.fetchall()
                st.header("**User's👨‍💻 Data**")
                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                                                 'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                                                 'Recommended Course'])
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'Resume_Data.csv', 'Download Report'), unsafe_allow_html=True)
                
                ## Admin Side Data
                query = 'select * from resume_data;'
                plot_data = pd.read_sql(query, connection)

                ## Pie chart for predicted field recommendations
                values = plot_data['Predicted_Field'].value_counts()
                labels = values.index
                st.subheader("📈 **Pie-Chart for Predicted Field Recommendations**")
                fig = px.pie(values=values, names=labels, title='Predicted Field according to the Skills')
                st.plotly_chart(fig)

                ### Pie chart for User's👨‍💻 Experienced Level
                values_user_level = plot_data['User_level'].value_counts()
                labels_user_level = values_user_level.index
                st.subheader("📈 **Pie-Chart for User's Experienced Level**")
                fig_user_level = px.pie(values=values_user_level, names=labels_user_level, title="Pie-Chart for User's Experienced Level")
                st.plotly_chart(fig_user_level)

                # Additional ML Analytics
                st.subheader("📊 **Resume Score Distribution**")
                scores = pd.to_numeric(plot_data['resume_score'], errors='coerce')
                fig_hist = px.histogram(scores, title='Distribution of Resume Scores', nbins=10)
                st.plotly_chart(fig_hist)

            else:
                st.error("Wrong ID & Password Provided")

if __name__ == '__main__':
    run()
