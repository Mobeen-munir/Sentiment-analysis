import os
import re
import ast
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

# ---------------------------------------------------------
# Page Configuration & Modern Theme Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Daraz Reviews Intelligence & Sentiment Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished aesthetic
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ff6000;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555555;
        margin-bottom: 20px;
    }
    .kpi-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px 20px;
        border-left: 5px solid #ff6000;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: bold;
        color: #222222;
    }
    .kpi-lbl {
        font-size: 0.85rem;
        color: #777777;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        font-size: 15px;
        font-weight: 600;
        border-radius: 6px 6px 0 0;
        padding: 0 16px;
    }
    .badge-pos {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-neg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
    .badge-neu {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Custom Stopwords for Roman Urdu + English
# ---------------------------------------------------------
CUSTOM_STOPWORDS = set([
    'hai', 'k', 'ki', 'ko', 'aur', 'b', 'bhi', 'ny', 'sy', 'se', 'kuch', 'yeh', 'woh',
    'mera', 'meri', 'mere', 'in', 'hi', 'to', 'tha', 'thi', 'the', 'ho', 'ka', 'ke',
    'is', 'iss', 'it', 'the', 'and', 'for', 'of', 'in', 'on', 'a', 'an', 'to', 'was',
    'at', 'by', 'with', 'from', 'as', 'are', 'this', 'that', 'these', 'those', 'or',
    'par', 'pe', 'mein', 'may', 'main', 'ap', 'aap', 'un', 'unka', 'unki', 'h', 'hy',
    'kar', 'kya', 'kr', 'raha', 'rhi', 'rha', 'rahe', 'hona', 'honi', 'gaya', 'gya',
    'gai', 'ye', 'wo', 'jo', 'jis', 'jin', 'sab', 'sb', 'liya', 'diya', 'de', 'do',
    'sirf', 'siraf', 'bhot', 'bohat', 'boht', 'bahut', 'ziada', 'zyada', 'mujhe', 'hum',
    'karein', 'karta', 'karti', 'hain', 'hon', 'thay', 'ab', 'tak'
])

# ---------------------------------------------------------
# Helper: Text Preprocessing
# ---------------------------------------------------------
def clean_review(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Strip HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Strip URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Lowercase
    text = text.lower()
    # Remove symbols while preserving characters and whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ---------------------------------------------------------
# Cache Dataset Loading
# ---------------------------------------------------------
@st.cache_data
def load_data():
    candidate_paths = [
        'Daraz Labelled Review Dataset with Sentiments and Features.xlsx',
        r'C:\Users\Asus\Downloads\archive (4)\Daraz Labelled Review Dataset with Sentiments and Features.xlsx',
        os.path.join(os.path.dirname(__file__), 'Daraz Labelled Review Dataset with Sentiments and Features.xlsx')
    ]
    file_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            file_path = p
            break
            
    if file_path is None:
        st.error("Excel dataset not found! Please place 'Daraz Labelled Review Dataset with Sentiments and Features.xlsx' in the app directory.")
        st.stop()
        
    df = pd.read_excel(file_path)
    
    # Drop unnamed index column
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
        
    df['original_reviews'] = df['Reviews'].astype(str)
    df['original_sentiment'] = df['Sentiment'].astype(str)
    df['cleaned_review'] = df['Reviews'].apply(clean_review)
    df['char_length'] = df['Reviews'].astype(str).str.len()
    df['word_count'] = df['Reviews'].astype(str).str.split().str.len()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    return df

df_full = load_data()

# ---------------------------------------------------------
# Cache Model Training (TF-IDF + Logistic Regression)
# ---------------------------------------------------------
@st.cache_resource
def train_sentiment_model(df):
    X = df['cleaned_review']
    y = df['Sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train_vec, y_train)
    
    y_pred = clf.predict(X_test_vec)
    y_prob = clf.predict_proba(X_test_vec)
    
    acc = accuracy_score(y_test, y_pred)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    labels = ['Negative', 'Neutral', 'Positive']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    report_dict = classification_report(y_test, y_pred, digits=4, output_dict=True)
    
    return {
        'model': clf,
        'vectorizer': tfidf,
        'accuracy': acc,
        'prec_weighted': prec_w,
        'rec_weighted': rec_w,
        'f1_weighted': f1_w,
        'f1_macro': f1_m,
        'cm': cm,
        'report_dict': report_dict,
        'labels': labels,
        'test_count': len(X_test)
    }

ml_results = train_sentiment_model(df_full)

# Color Scheme Mapping
COLOR_MAP = {
    'Positive': '#2ecc71',
    'Neutral': '#f39c12',
    'Negative': '#e74c3c'
}

# ---------------------------------------------------------
# Sidebar Controls & Filtering
# ---------------------------------------------------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/4/4a/Daraz_Logo.png", width=140)
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Filter and inspect reviews interactively.")

# Sentiment Filter
sentiments_available = ['Positive', 'Negative', 'Neutral']
selected_sentiments = st.sidebar.multiselect(
    "Sentiment Categories",
    options=sentiments_available,
    default=sentiments_available
)

# Rating Filter
min_rating, max_rating = int(df_full['Rating'].min()), int(df_full['Rating'].max())
selected_ratings = st.sidebar.slider(
    "Rating Range (Stars)",
    min_value=min_rating,
    max_value=max_rating,
    value=(min_rating, max_rating),
    step=1
)

# Spam Filter
spam_filter_option = st.sidebar.radio(
    "Review Authenticity",
    options=["All Reviews", "Organic Only (Non-Spam)", "Spam Only"]
)

# Keyword Search
keyword_search = st.sidebar.text_input(
    "Search Review Text (e.g., toota, quality, size)",
    ""
)

# Apply Filters
filtered_df = df_full.copy()

if selected_sentiments:
    filtered_df = filtered_df[filtered_df['Sentiment'].isin(selected_sentiments)]
else:
    filtered_df = filtered_df.iloc[0:0]

filtered_df = filtered_df[
    (filtered_df['Rating'] >= selected_ratings[0]) & 
    (filtered_df['Rating'] <= selected_ratings[1])
]

if spam_filter_option == "Organic Only (Non-Spam)":
    filtered_df = filtered_df[filtered_df['Spam(1) and Not Spam(0)'] == 0]
elif spam_filter_option == "Spam Only":
    filtered_df = filtered_df[filtered_df['Spam(1) and Not Spam(0)'] == 1]

if keyword_search.strip():
    kw = keyword_search.strip().lower()
    filtered_df = filtered_df[filtered_df['cleaned_review'].str.contains(kw, case=False, na=False)]

st.sidebar.markdown(f"**Showing:** `{len(filtered_df):,}` of `{len(df_full):,}` reviews ({len(filtered_df)/len(df_full)*100:.1f}%)")
if st.sidebar.button("🔄 Reset Filters"):
    st.rerun()

# ---------------------------------------------------------
# Main Header & Top KPI Banner
# ---------------------------------------------------------
st.markdown('<div class="main-title">🛒 Daraz Reviews Sentiment & EDA Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Natural Language Processing & Exploratory Intelligence on 3,923 Roman Urdu & English Customer Reviews</div>', unsafe_allow_html=True)

if len(filtered_df) == 0:
    st.warning("No reviews match your selected filter criteria. Please adjust your filters in the sidebar.")
    st.stop()

# Compute KPI Metrics
total_count = len(filtered_df)
pos_count = (filtered_df['Sentiment'] == 'Positive').sum()
neg_count = (filtered_df['Sentiment'] == 'Negative').sum()
neu_count = (filtered_df['Sentiment'] == 'Neutral').sum()

pos_pct = (pos_count / total_count) * 100 if total_count > 0 else 0
neg_pct = (neg_count / total_count) * 100 if total_count > 0 else 0
neu_pct = (neu_count / total_count) * 100 if total_count > 0 else 0
avg_rating = filtered_df['Rating'].mean()

# Calculate discrepancy count: 5-star negative & 1-star positive
discrepancy_count = len(filtered_df[(filtered_df['Rating'] == 5) & (filtered_df['Sentiment'] == 'Negative')]) + \
                    len(filtered_df[(filtered_df['Rating'] == 1) & (filtered_df['Sentiment'] == 'Positive')])
discrepancy_pct = (discrepancy_count / total_count) * 100 if total_count > 0 else 0

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("Total Reviews", f"{total_count:,}")
with k2:
    st.metric("Positive Sentiment", f"{pos_pct:.1f}%", f"{pos_count:,} reviews")
with k3:
    st.metric("Negative Sentiment", f"{neg_pct:.1f}%", f"-{neg_count:,} reviews", delta_color="inverse")
with k4:
    st.metric("Neutral Sentiment", f"{neu_pct:.1f}%", f"{neu_count:,} reviews", delta_color="off")
with k5:
    st.metric("Avg Star Rating", f"★ {avg_rating:.2f} / 5.0")
with k6:
    st.metric("Rating Discrepancy", f"{discrepancy_pct:.1f}%", f"{discrepancy_count:,} reviews", delta_color="inverse")

st.markdown("---")

# ---------------------------------------------------------
# Dashboard Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Overview & Sentiment EDA",
    "📝 Linguistic & Aspect Analytics",
    "🤖 ML Model & Live Predictor",
    "💡 Strategic Business Insights",
    "🔍 Interactive Data Explorer"
])

# =========================================================
# TAB 1: OVERVIEW & SENTIMENT EDA
# =========================================================
with tabs[0]:
    st.subheader("1. Sentiment & Rating Distributions")
    
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        # 1. Sentiment Distribution Bar Chart
        sent_counts_df = filtered_df['Sentiment'].value_counts().reset_index()
        sent_counts_df.columns = ['Sentiment', 'Count']
        sent_counts_df['Percentage'] = (sent_counts_df['Count'] / total_count) * 100
        
        fig_bar = px.bar(
            sent_counts_df,
            x='Sentiment',
            y='Count',
            text=sent_counts_df.apply(lambda r: f"{r['Count']:,} ({r['Percentage']:.1f}%)", axis=1),
            color='Sentiment',
            color_discrete_map=COLOR_MAP,
            title="<b>Sentiment Volume & Percentages</b>",
            labels={'Count': 'Number of Reviews'}
        )
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, height=380, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col_b:
        # 2. Sentiment Donut Pie Chart
        fig_pie = px.pie(
            sent_counts_df,
            names='Sentiment',
            values='Count',
            color='Sentiment',
            color_discrete_map=COLOR_MAP,
            hole=0.45,
            title="<b>Proportion of Sentiment Categories</b>"
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label', marker=dict(line=dict(color='#ffffff', width=2)))
        fig_pie.update_layout(height=380, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    col_c, col_d = st.columns([1, 1])
    
    with col_c:
        # 3. Rating Distribution
        rating_df = filtered_df['Rating'].value_counts().sort_index().reset_index()
        rating_df.columns = ['Rating', 'Count']
        rating_df['Label'] = rating_df['Rating'].astype(str) + " Star"
        rating_df['Percentage'] = (rating_df['Count'] / total_count) * 100
        
        fig_rating = px.bar(
            rating_df,
            x='Label',
            y='Count',
            text=rating_df.apply(lambda r: f"{r['Count']:,} ({r['Percentage']:.1f}%)", axis=1),
            title="<b>Rating Distribution (1 to 5 Stars)</b>",
            color_discrete_sequence=['#3498db'],
            labels={'Label': 'Star Rating', 'Count': 'Frequency'}
        )
        fig_rating.update_traces(textposition='outside')
        fig_rating.update_layout(height=380, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_rating, use_container_width=True)
        
    with col_d:
        # 4. Rating vs Sentiment Stacked Bar Chart
        crosstab_df = pd.crosstab(filtered_df['Rating'], filtered_df['Sentiment'], normalize='index') * 100
        crosstab_df = crosstab_df.reset_index()
        
        # Melt for Plotly
        melted = pd.melt(crosstab_df, id_vars=['Rating'], var_name='Sentiment', value_name='Percentage')
        melted['Rating_Label'] = melted['Rating'].astype(str) + " Star"
        
        fig_stack = px.bar(
            melted,
            x='Rating_Label',
            y='Percentage',
            color='Sentiment',
            color_discrete_map=COLOR_MAP,
            title="<b>Rating vs. Expressed Sentiment Alignment (%)</b>",
            text=melted['Percentage'].apply(lambda x: f"{x:.1f}%" if x > 6 else ""),
            labels={'Rating_Label': 'Star Rating', 'Percentage': 'Percentage of Reviews (%)'},
            barmode='stack'
        )
        fig_stack.update_layout(height=380, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_stack, use_container_width=True)

    # Discrepancy & Sarcasm Inspector
    st.markdown("### ⚠️ Star Rating & Sentiment Discrepancy Inspector")
    st.markdown("""
    A critical discovery in the dataset is the **divergence between star ratings and text sentiment**.
    Buyers often rate **5 stars** so their complaint appears prominently, or give **1 star** to penalize courier delays despite liking the product.
    """)
    
    disc_col1, disc_col2 = st.columns(2)
    
    with disc_col1:
        st.error("🚨 5-Star Rating but NEGATIVE Text Sentiment")
        sample_5star_neg = filtered_df[(filtered_df['Rating'] == 5) & (filtered_df['Sentiment'] == 'Negative')][['original_reviews', 'Customer_Name', 'Date']].head(5)
        if len(sample_5star_neg) > 0:
            for idx, r in sample_5star_neg.iterrows():
                st.markdown(f"> *\"{r['original_reviews']}\"* — **{r['Customer_Name']}**")
        else:
            st.info("No 5-star negative reviews in current filter.")
            
    with disc_col2:
        st.success("✨ 1-Star Rating but POSITIVE Text Sentiment")
        sample_1star_pos = filtered_df[(filtered_df['Rating'] == 1) & (filtered_df['Sentiment'] == 'Positive')][['original_reviews', 'Customer_Name', 'Date']].head(5)
        if len(sample_1star_pos) > 0:
            for idx, r in sample_1star_pos.iterrows():
                st.markdown(f"> *\"{r['original_reviews']}\"* — **{r['Customer_Name']}**")
        else:
            st.info("No 1-star positive reviews in current filter.")

# =========================================================
# TAB 2: LINGUISTIC & ASPECT ANALYTICS
# =========================================================
with tabs[1]:
    st.subheader("2. Review Length & Lexical Characteristics")
    
    l_col1, l_col2 = st.columns([1, 1])
    
    with l_col1:
        # 5. Review Length Histogram
        fig_hist = px.histogram(
            filtered_df,
            x='word_count',
            nbins=40,
            marginal='box',
            color_discrete_sequence=['#2980b9'],
            title="<b>Review Length Distribution (Word Count)</b>",
            labels={'word_count': 'Word Count in Review'}
        )
        fig_hist.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with l_col2:
        # 6. Sentiment vs Review Length Boxplot
        fig_box = px.box(
            filtered_df,
            x='Sentiment',
            y='word_count',
            color='Sentiment',
            color_discrete_map=COLOR_MAP,
            category_orders={'Sentiment': ['Positive', 'Neutral', 'Negative']},
            title="<b>Review Verbosity Across Sentiment Categories</b>",
            labels={'word_count': 'Word Count'}
        )
        fig_box.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_box, use_container_width=True)
        
    st.info("💡 **Verbosity Key Finding**: Negative reviews average **31 words** (median 22), compared to **24 words** for positive and **15 words** for neutral. Dissatisfied customers articulate specific failure points in high detail.")

    st.markdown("---")
    st.subheader("3. Word Clouds & Lexical Frequency (Roman Urdu + English)")
    
    wc_option = st.radio("Select Sentiment for Word Cloud Generation:", ["Positive", "Negative", "Neutral"], horizontal=True)
    
    wc_col1, wc_col2 = st.columns([1.2, 1])
    
    with wc_col1:
        # 7. Word Cloud
        sub_text = " ".join(filtered_df[filtered_df['Sentiment'] == wc_option]['cleaned_review'])
        filtered_words = [w for w in sub_text.split() if w not in CUSTOM_STOPWORDS and len(w) > 2]
        
        if len(filtered_words) > 0:
            colormap_dict = {'Positive': 'Greens', 'Negative': 'Reds', 'Neutral': 'YlOrBr'}
            wc = WordCloud(
                width=700, height=400,
                background_color='white',
                colormap=colormap_dict[wc_option],
                max_words=85,
                random_state=42
            ).generate(" ".join(filtered_words))
            
            fig_wc, ax_wc = plt.subplots(figsize=(8, 4.5))
            ax_wc.imshow(wc, interpolation='bilinear')
            ax_wc.axis('off')
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc)
        else:
            st.info("Not enough words to generate word cloud.")
            
    with wc_col2:
        # 8. Top 12 Frequent Words
        words_counter = Counter(filtered_words)
        top12 = words_counter.most_common(12)
        if top12:
            top_df = pd.DataFrame(top12, columns=['Term', 'Frequency']).sort_values('Frequency', ascending=True)
            fig_top = px.bar(
                top_df,
                x='Frequency',
                y='Term',
                orientation='h',
                color_discrete_sequence=[COLOR_MAP[wc_option]],
                title=f"<b>Top Words in {wc_option} Reviews</b>",
                text='Frequency'
            )
            fig_top.update_traces(textposition='outside')
            fig_top.update_layout(height=400, margin=dict(t=50, b=20, l=20, r=20))
            st.plotly_chart(fig_top, use_container_width=True)
            
    st.markdown("---")
    st.subheader("4. Aspect-Based Feature Mentions by Sentiment")
    
    # 11. Extracted Aspect Mentions
    all_aspects = []
    for idx, row in filtered_df.iterrows():
        try:
            feats = ast.literal_eval(row['Features'])
            if isinstance(feats, list):
                for f in feats:
                    fn = f.lower().strip()
                    if fn and len(fn) > 2:
                        all_aspects.append({'Aspect': fn, 'Sentiment': row['Sentiment']})
        except Exception:
            pass
            
    if all_aspects:
        aspect_df = pd.DataFrame(all_aspects)
        top_aspect_names = aspect_df['Aspect'].value_counts().head(10).index.tolist()
        aspect_sub = aspect_df[aspect_df['Aspect'].isin(top_aspect_names)]
        
        asp_crosstab = pd.crosstab(aspect_sub['Aspect'], aspect_sub['Sentiment']).loc[top_aspect_names]
        asp_melted = asp_crosstab.reset_index().melt(id_vars='Aspect', var_name='Sentiment', value_name='Count')
        
        fig_aspect = px.bar(
            asp_melted,
            x='Aspect',
            y='Count',
            color='Sentiment',
            color_discrete_map=COLOR_MAP,
            title="<b>Top Extracted Aspects/Features Mentioned Across Sentiments</b>",
            barmode='stack'
        )
        fig_aspect.update_layout(height=420, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_aspect, use_container_width=True)
        st.caption("Top aspects include: **miyaar/quality** (product craftsmanship), **qeemat** (price/value), **size** (fit/dimensions), and **order/delivery**.")

# =========================================================
# TAB 3: ML MODEL & LIVE PREDICTOR
# =========================================================
with tabs[2]:
    st.subheader("🤖 Supervised Sentiment Classifier (TF-IDF + Logistic Regression)")
    st.markdown("""
    The model was trained on an 80/20 stratified split of the 3,923 reviews using unigrams + bigrams and balanced class weighting.
    """)
    
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Test Accuracy", f"{ml_results['accuracy']*100:.2f}%")
    with m_col2:
        st.metric("Weighted Precision", f"{ml_results['prec_weighted']*100:.2f}%")
    with m_col3:
        st.metric("Weighted Recall", f"{ml_results['rec_weighted']*100:.2f}%")
    with m_col4:
        st.metric("Weighted F1-Score", f"{ml_results['f1_weighted']*100:.2f}%")

    eval_col1, eval_col2 = st.columns([1, 1])
    
    with eval_col1:
        # 9. Confusion Matrix Heatmap
        cm = ml_results['cm']
        labels = ml_results['labels']
        
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=labels,
            y=labels,
            color_continuous_scale='Blues',
            title="<b>Confusion Matrix (Evaluation on 785 Test Samples)</b>",
            labels=dict(x="Predicted Sentiment", y="Ground Truth Sentiment", color="Count")
        )
        fig_cm.update_layout(height=380, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_cm, use_container_width=True)
        
    with eval_col2:
        # 10. Metric comparison bar chart
        metric_df = pd.DataFrame({
            'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1-Score', 'Macro F1-Score'],
            'Score (%)': [
                ml_results['accuracy'] * 100,
                ml_results['prec_weighted'] * 100,
                ml_results['rec_weighted'] * 100,
                ml_results['f1_weighted'] * 100,
                ml_results['f1_macro'] * 100
            ]
        })
        fig_perf = px.bar(
            metric_df,
            x='Metric',
            y='Score (%)',
            text=metric_df['Score (%)'].apply(lambda x: f"{x:.2f}%"),
            color='Metric',
            title="<b>Model Performance Metric Benchmarks</b>",
            color_discrete_sequence=['#2980b9', '#27ae60', '#e67e22', '#8e44ad', '#16a085']
        )
        fig_perf.update_traces(textposition='outside')
        fig_perf.update_layout(showlegend=False, yaxis=dict(range=[0, 100]), height=380, margin=dict(t=50, b=20, l=20, r=20))
        st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("---")
    st.subheader("🔮 Live Sentiment Inference Engine (Test Any Review)")
    st.markdown("Enter custom feedback in **Roman Urdu** or **English** to predict the expressed customer sentiment instantly:")
    
    # Preset sample reviews for quick demonstration
    demo_samples = [
        "Select a sample review...",
        "mil gya hai acha hai aur toota howa b nai hai (Positive)",
        "bohat bekar quality hai toota howa mila aur size b chota hai (Negative)",
        "wakt par delivery agai thi aur theek thaak kaam kar rha hai (Neutral)",
        "packing achi thi lakin color galat bhej diya (Negative)",
        "daraz se pehli dafa mangwaya bhot zabardast product hai (Positive)"
    ]
    selected_sample = st.selectbox("Quick-fill with realistic Daraz samples:", demo_samples)
    
    default_text = ""
    if selected_sample != demo_samples[0]:
        # Extract text before parenthesis
        default_text = selected_sample.split(" (")[0]
        
    user_input = st.text_area("Customer Review Input:", value=default_text, height=100, placeholder="e.g., cheez achi hai lakin delivery late thi...")
    
    if st.button("🚀 Analyze Sentiment", type="primary"):
        if user_input.strip():
            clean_in = clean_review(user_input)
            vec = ml_results['vectorizer'].transform([clean_in])
            pred_class = ml_results['model'].predict(vec)[0]
            pred_probs = ml_results['model'].predict_proba(vec)[0]
            classes = ml_results['model'].classes_
            
            p_res1, p_res2 = st.columns([1, 1.5])
            with p_res1:
                st.markdown("#### Predicted Category:")
                if pred_class == 'Positive':
                    st.markdown("<div style='font-size: 2rem; color: #2ecc71; font-weight: bold;'>🟢 Positive Sentiment</div>", unsafe_allow_html=True)
                elif pred_class == 'Negative':
                    st.markdown("<div style='font-size: 2rem; color: #e74c3c; font-weight: bold;'>🔴 Negative Sentiment</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='font-size: 2rem; color: #f39c12; font-weight: bold;'>🟡 Neutral Sentiment</div>", unsafe_allow_html=True)
                
                max_prob = max(pred_probs) * 100
                st.metric("Model Confidence", f"{max_prob:.1f}%")
                
            with p_res2:
                prob_df = pd.DataFrame({
                    'Sentiment': classes,
                    'Probability (%)': [p * 100 for p in pred_probs]
                })
                fig_prob = px.bar(
                    prob_df,
                    x='Probability (%)',
                    y='Sentiment',
                    orientation='h',
                    color='Sentiment',
                    color_discrete_map=COLOR_MAP,
                    text=prob_df['Probability (%)'].apply(lambda x: f"{x:.1f}%"),
                    title="<b>Confidence Probability Distribution</b>"
                )
                fig_prob.update_traces(textposition='outside')
                fig_prob.update_layout(height=220, margin=dict(t=40, b=10, l=10, r=10), xaxis=dict(range=[0, 110]))
                st.plotly_chart(fig_prob, use_container_width=True)
        else:
            st.warning("Please type or select a review to test.")

# =========================================================
# TAB 4: STRATEGIC BUSINESS INSIGHTS
# =========================================================
with tabs[3]:
    st.subheader("💡 Strategic Marketplace Insights & Operational Roadmap")
    
    bi1, bi2 = st.columns(2)
    
    with bi1:
        st.markdown("### 🔍 Key Dataset Findings")
        st.markdown("""
        1. **Sentiment Baseline**:
           - **60.5% Positive**: Indicates overall healthy satisfaction on established product categories.
           - **25.1% Negative**: 1 in every 4 buyers encounters a significant failure point.
           - **14.5% Neutral**: Factual delivery confirmations and functional remarks.
        2. **Star Rating Discrepancy**:
           - **12% of 5-star reviews contain negative text!** Customers write complaints like *"bilkul bekar product"* under 5 stars so their feedback is not buried at the bottom.
           - **17% of 1-star reviews contain positive text!** Customers praise product quality but give 1 star to punish courier delays.
        3. **Review Verbosity as a Churn Signal**:
           - Negative reviews average **31 words** (60% longer than positive reviews). Lengthy reviews strongly signal elevated churn risk and customer frustration.
        """)
        
    with bi2:
        st.markdown("### 🏆 Top Satisfaction & Friction Drivers")
        st.markdown("""
        - **Primary Satisfaction Drivers**:
          - Affordable pricing (`qeemat`, `munasib`)
          - Comfortable materials (`aram`, `soft`)
          - Authentic items matching the listed specs (`original`, `sai kaam`)
        - **Primary Customer Pain Points**:
          - Transit breakage & poor packaging (`toota`)
          - Inaccurate apparel & accessories sizing (`size`)
          - Faulty hardware & color mismatch (`kharab`, `rang`)
          - Unreliable delivery windows and courier tracking
        """)

    st.markdown("---")
    st.markdown("### 🚀 Actionable Recommendations for Daraz Marketplace")
    
    rec1, rec2, rec3, rec4 = st.columns(4)
    with rec1:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #e74c3c; min-height: 220px;">
            <h4>1. Standardized Sizing</h4>
            <p><strong>Problem:</strong> Size is responsible for ~38% of negative aspect mentions.<br>
            <strong>Action:</strong> Mandate centimeter & inch sizing charts and real-buyer photos for all apparel and phone cases.</p>
        </div>
        """, unsafe_allow_html=True)
    with rec2:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #f39c12; min-height: 220px;">
            <h4>2. Dual-Signal Triage</h4>
            <p><strong>Problem:</strong> 12% of 5-star reviews are actually complaints.<br>
            <strong>Action:</strong> Implement NLP text validation alongside star ratings to detect rating-text conflict and route to seller QA.</p>
        </div>
        """, unsafe_allow_html=True)
    with rec3:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #3498db; min-height: 220px;">
            <h4>3. Decouple Logistics</h4>
            <p><strong>Problem:</strong> Sellers get 1-star reviews due to 3PL courier delays.<br>
            <strong>Action:</strong> Separate review prompts into <em>Product Quality (1-5★)</em> and <em>Delivery Experience (1-5★)</em>.</p>
        </div>
        """, unsafe_allow_html=True)
    with rec4:
        st.markdown("""
        <div class="kpi-card" style="border-left-color: #2ecc71; min-height: 220px;">
            <h4>4. Breakage Alerts</h4>
            <p><strong>Problem:</strong> Fragile items arrive broken (<em>toota</em>).<br>
            <strong>Action:</strong> Real-time seller portal alerts when <em>toota</em> spikes, triggering mandatory bubble-wrap enforcement.</p>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# TAB 5: DATA EXPLORER & EXPORT
# =========================================================
with tabs[4]:
    st.subheader("🔍 Interactive Review Data Explorer")
    st.markdown(f"Displaying **{len(filtered_df):,}** filtered records.")
    
    cols_to_show = st.multiselect(
        "Select columns to display:",
        options=list(filtered_df.columns),
        default=['Product_ID', 'Customer_Name', 'Rating', 'Sentiment', 'original_reviews', 'word_count', 'Spam(1) and Not Spam(0)']
    )
    
    st.dataframe(filtered_df[cols_to_show], use_container_width=True, height=450)
    
    # Download Button
    csv_data = filtered_df[cols_to_show].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Dataset as CSV",
        data=csv_data,
        file_name="daraz_filtered_reviews.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: #888888; font-size: 0.85rem;'>Daraz Reviews Sentiment Analysis & EDA System | Built with Streamlit, Scikit-learn, Plotly & WordCloud</p>", unsafe_allow_html=True)
