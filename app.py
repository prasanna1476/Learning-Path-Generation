import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Set a random seed for reproducibility in dummy data generation
np.random.seed(42)
random.seed(42)

# --- Functions for Data Generation and ML Logic ---

@st.cache_data # Cache data loading for performance
def load_and_preprocess_data():
    """
    Loads or generates student learning data and performs initial preprocessing.
    In a real app, you would load from a CSV/DB:
    df = pd.read_csv('student_learning_data.csv')
    """
    # Dummy Data Creation (as in the Jupyter Notebook)
    num_students = 100
    num_courses = 5
    num_modules_per_course = 3
    num_resources_per_module = 5

    data = []
    student_ids = [f'S{i:03d}' for i in range(1, num_students + 1)]
    course_ids = [f'C{i:03d}' for i in range(1, num_courses + 1)]
    learning_styles = ['visual', 'auditory', 'kinesthetic', 'reading/writing']
    resource_types = ['video', 'article', 'quiz']

    for s_id in student_ids:
        for c_id in random.sample(course_ids, k=random.randint(1, num_courses)):
            for m_id in [f'M{i:03d}' for i in range(1, num_modules_per_course + 1)]:
                for r_id in [f'R{i:03d}' for i in range(1, num_resources_per_module + 1)]:
                    if random.random() > 0.1: # Simulate some missing interactions
                        time_spent = np.random.randint(10, 120)
                        quiz_score = np.random.randint(40, 100) if random.random() > 0.2 else np.nan
                        completion_status = 1 if time_spent > 30 and quiz_score > 50 else 0
                        difficulty = np.random.randint(1, 5)
                        pre_req_met = 1 if random.random() > 0.3 else 0
                        learning_style = random.choice(learning_styles)
                        resource_type = random.choice(resource_types)
                        course_tags = random.sample(['programming', 'data science', 'math', 'history', 'art', 'beginner', 'advanced'], k=random.randint(1,3))

                        data.append([
                            s_id, c_id, m_id, r_id, time_spent, quiz_score,
                            completion_status, difficulty, learning_style,
                            pre_req_met, resource_type, ','.join(course_tags)
                        ])

    df = pd.DataFrame(data, columns=[
        'student_id', 'course_id', 'module_id', 'resource_id', 'time_spent_minutes',
        'quiz_score', 'completion_status', 'difficulty_level',
        'learning_style_preference', 'pre_requisite_met', 'resource_type', 'course_tags'
    ])

    # Handle Missing Values
    df['quiz_score'].fillna(df['quiz_score'].mean(), inplace=True)
    df['performance_score'] = (df['quiz_score'] * 0.7 + df['completion_status'] * 100 * 0.3) / 100

    return df

@st.cache_resource # Cache the trained model and preprocessor
def train_clustering_model(df_raw):
    """
    Trains the K-Means clustering model on aggregated student features.
    """
    student_features = df_raw.groupby('student_id').agg(
        avg_time_spent=('time_spent_minutes', 'mean'),
        avg_quiz_score=('quiz_score', 'mean'),
        total_completed_resources=('completion_status', 'sum'),
        avg_difficulty_attempted=('difficulty_level', 'mean'),
        preferred_learning_style=('learning_style_preference', lambda x: x.mode()[0] if not x.mode().empty else np.nan)
    ).reset_index()

    student_features_clust = student_features.copy()
    student_features_clust = pd.get_dummies(student_features_clust, columns=['preferred_learning_style'], prefix='style')

    numerical_features = ['avg_time_spent', 'avg_quiz_score', 'total_completed_resources', 'avg_difficulty_attempted']
    categorical_features_ohe = [col for col in student_features_clust.columns if 'style_' in col]
    all_features_for_clustering = numerical_features + categorical_features_ohe

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', 'passthrough', categorical_features_ohe)
        ])

    X_clust = preprocessor.fit_transform(student_features_clust)

    n_clusters = 3 # Based on previous analysis or domain knowledge
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    student_features['cluster'] = kmeans.fit_predict(X_clust)

    return student_features, kmeans, preprocessor, numerical_features, all_features_for_clustering

def recommend_path(student_id, student_df, df_raw, kmeans_model, preprocessor_for_student_features, numerical_features, all_features_for_clustering):
    """
    Recommends a personalized learning path for a given student.
    """
    student_data = student_df[student_df['student_id'] == student_id]
    if student_data.empty:
        return "Student not found.", None, None

    student_features_for_pred = student_data[numerical_features + ['preferred_learning_style']]
    student_features_for_pred = pd.get_dummies(student_features_for_pred, columns=['preferred_learning_style'], prefix='style')

    # Ensure all columns expected by preprocessor are present
    # This is crucial for consistent prediction in Streamlit where individual student data is processed
    known_cat_features = [col for col in preprocessor_for_student_features.named_transformers_['cat'].get_feature_names_out()]
    for col in known_cat_features:
        if col not in student_features_for_pred.columns:
            student_features_for_pred[col] = 0

    student_features_for_pred = student_features_for_pred[all_features_for_clustering] # Ensure order consistency

    student_transformed = preprocessor_for_student_features.transform(student_features_for_pred)
    student_cluster = kmeans_model.predict(student_transformed)[0]

    cluster_students = student_df[student_df['cluster'] == student_cluster]['student_id']
    relevant_interactions = df_raw[df_raw['student_id'].isin(cluster_students)]

    recommended_content_candidates = relevant_interactions.groupby(['course_id', 'module_id', 'resource_id']).agg(
        avg_quiz_score=('quiz_score', 'mean'),
        avg_completion_status=('completion_status', 'mean'),
        resource_difficulty=('difficulty_level', 'mean')
    ).reset_index()

    good_performing_content = recommended_content_candidates[
        (recommended_content_candidates['avg_quiz_score'] > 75) &
        (recommended_content_candidates['avg_completion_status'] > 0.8)
    ].sort_values(by=['avg_quiz_score', 'avg_completion_status'], ascending=False)

    student_completed_resources = df_raw[(df_raw['student_id'] == student_id) & (df_raw['completion_status'] == 1)]['resource_id'].tolist()

    top_n = 5
    personalized_recommendations = []
    for index, row in good_performing_content.iterrows():
        resource_id = row['resource_id']
        if resource_id not in student_completed_resources:
            personalized_recommendations.append(
                f"Course: {row['course_id']}, Module: {row['module_id']}, Resource: {row['resource_id']} "
                f"(Avg Quiz Score: {row['avg_quiz_score']:.1f}, Avg Completion: {row['avg_completion_status']:.1f})"
            )
        if len(personalized_recommendations) >= top_n:
            break

    if not personalized_recommendations:
        return "No specific new recommendations at this time based on your cluster's top content.", student_cluster, student_data
    return personalized_recommendations, student_cluster, student_data

# --- Streamlit Application Layout ---

st.set_page_config(layout="wide", page_title="Personalized Learning Path Recommender")

st.title("🎓 Personalized Learning Path Recommender")
st.markdown("This application uses Machine Learning (K-Means Clustering) to analyze student data and recommend relevant learning resources.")

# Load and preprocess data
with st.spinner("Loading and preprocessing data..."):
    df_raw_data = load_and_preprocess_data()
    student_features_df, kmeans_model, preprocessor, numerical_features_list, all_features_for_clustering_list = train_clustering_model(df_raw_data)

st.sidebar.header("Student Selection")
all_student_ids = sorted(student_features_df['student_id'].unique())
selected_student_id = st.sidebar.selectbox("Select a Student ID", all_student_ids)

if st.sidebar.button("Generate Recommendations"):
    st.header(f"Recommendations for {selected_student_id}")

    recommendations, student_cluster, student_info = recommend_path(
        selected_student_id,
        student_features_df,
        df_raw_data,
        kmeans_model,
        preprocessor,
        numerical_features_list,
        all_features_for_clustering_list
    )

    if isinstance(recommendations, str):
        st.warning(recommendations)
    else:
        st.subheader(f"Student {selected_student_id} is in Cluster {student_cluster}")
        st.write("---")
        st.subheader("Personalized Learning Path Recommendations:")
        for i, rec in enumerate(recommendations):
            st.write(f"*{i+1}.* {rec}")

        st.write("---")
        st.subheader("Student Profile (Aggregated Data):")
        if student_info is not None:
            st.dataframe(student_info.set_index('student_id'))

        st.write("---")
        st.subheader("Learning Style Distribution in Clusters:")
        fig_cluster_style = plt.figure(figsize=(10, 6))
        sns.countplot(y='preferred_learning_style', hue='cluster', data=student_features_df)
        plt.title('Preferred Learning Style Distribution by Cluster')
        st.pyplot(fig_cluster_style)

        st.subheader("Student Performance Visualizations:")
        student_raw_data_viz = df_raw_data[df_raw_data['student_id'] == selected_student_id]

        if not student_raw_data_viz.empty:
            fig_quiz_score = plt.figure(figsize=(10, 6))
            sns.barplot(x='module_id', y='quiz_score', data=student_raw_data_viz, estimator=np.mean, palette='viridis')
            plt.title(f'Average Quiz Scores by Module for {selected_student_id}')
            plt.ylabel('Average Quiz Score')
            st.pyplot(fig_quiz_score)

            fig_time_spent = plt.figure(figsize=(10, 6))
            sns.barplot(x='module_id', y='time_spent_minutes', data=student_raw_data_viz, estimator=np.sum, palette='plasma')
            plt.title(f'Total Time Spent by Module for {selected_student_id}')
            plt.ylabel('Total Time Spent (minutes)')
            st.pyplot(fig_time_spent)
        else:
            st.info("No detailed interaction data found for this student.")

st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤️ using Streamlit and Scikit-learn")