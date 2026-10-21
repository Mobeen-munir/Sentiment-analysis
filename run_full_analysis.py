import os
import re
import ast
import json
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix

# Configure style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

excel_path = r'C:\Users\Asus\Downloads\archive (4)\Daraz Labelled Review Dataset with Sentiments and Features.xlsx'
vis_dir = r'C:\Users\Asus\Downloads\archive (4)\visualizations'
os.makedirs(vis_dir, exist_ok=True)

print("--- 1. LOADING DATASET ---")
df = pd.read_excel(excel_path)
print(f"Dataset Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Drop unnamed index column if present
if 'Unnamed: 0' in df.columns:
    df_clean = df.drop(columns=['Unnamed: 0']).copy()
else:
    df_clean = df.copy()

# Preserve original review text and sentiment
df_clean['original_reviews'] = df_clean['Reviews'].copy()
df_clean['original_sentiment'] = df_clean['Sentiment'].copy()

# Data Cleaning function
# Preserves Roman Urdu characters, negations ('nahi', 'nai', 'na', 'not'), and emojis/sentiment terms
def clean_review_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation & special characters but keep alphanumeric and whitespace
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_clean['cleaned_review'] = df_clean['Reviews'].apply(clean_review_text)
df_clean['char_length'] = df_clean['Reviews'].astype(str).str.len()
df_clean['word_count'] = df_clean['Reviews'].astype(str).str.split().str.len()

print("\n--- 2. PREPROCESSING COMPLETED ---")
print(f"Total rows: {len(df_clean)}")
print(f"Null values in cleaned reviews: {df_clean['cleaned_review'].isnull().sum()}")

# 3. Model Training & Sentiment Analysis
print("\n--- 3. SENTIMENT CLASSIFICATION ---")
X = df_clean['cleaned_review']
y = df_clean['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf.fit(X_train_vec, y_train)

y_pred = clf.predict(X_test_vec)
acc = accuracy_score(y_test, y_pred)
prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')

labels_order = ['Negative', 'Neutral', 'Positive']
cm = confusion_matrix(y_test, y_pred, labels=labels_order)

print(f"Accuracy: {acc:.4f}")
print(f"Weighted Precision: {prec_w:.4f}, Recall: {rec_w:.4f}, F1: {f1_w:.4f}")
print(f"Macro Precision: {prec_m:.4f}, Recall: {rec_m:.4f}, F1: {f1_m:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

# Color palette for sentiment
palette = {'Positive': '#2ecc71', 'Neutral': '#f39c12', 'Negative': '#e74c3c'}
color_list = [palette['Positive'], palette['Negative'], palette['Neutral']]

# Roman Urdu + English stopwords for frequency analysis
stopwords_custom = set([
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

print("\n--- 4. GENERATING VISUALIZATIONS ---")

# Visualization 1: Sentiment Distribution (Bar chart with counts and percentages)
plt.figure(figsize=(8, 5))
sent_counts = df_clean['Sentiment'].value_counts()[labels_order[::-1]]
total_reviews = len(df_clean)
bars = plt.bar(sent_counts.index, sent_counts.values, color=[palette[s] for s in sent_counts.index], edgecolor='black', alpha=0.85)
for bar in bars:
    height = bar.get_height()
    pct = (height / total_reviews) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height + 30, f'{height:,}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.title('1. Sentiment Distribution (Counts and Percentages)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Sentiment Category', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.ylim(0, max(sent_counts.values) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '1_sentiment_distribution.png'), dpi=300)
plt.close()
print("Saved: 1_sentiment_distribution.png")

# Visualization 2: Sentiment Pie Chart
plt.figure(figsize=(7, 7))
sent_pie = df_clean['Sentiment'].value_counts()
colors = [palette[s] for s in sent_pie.index]
explode = (0.05, 0.05, 0.05)
plt.pie(sent_pie.values, labels=sent_pie.index, autopct='%1.1f%%', startangle=140,
        colors=colors, explode=explode, shadow=True,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.title('2. Proportion of Sentiment Categories', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '2_sentiment_pie_chart.png'), dpi=300)
plt.close()
print("Saved: 2_sentiment_pie_chart.png")

# Visualization 3: Rating Distribution
plt.figure(figsize=(8, 5))
rating_counts = df_clean['Rating'].value_counts().sort_index()
bars = plt.bar(rating_counts.index.astype(str) + ' Star', rating_counts.values,
               color='#3498db', edgecolor='black', alpha=0.85)
for bar in bars:
    height = bar.get_height()
    pct = (height / total_reviews) * 100
    plt.text(bar.get_x() + bar.get_width()/2., height + 25, f'{height:,}\n({pct:.1f}%)',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title('3. Customer Rating Distribution (1 to 5 Stars)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Customer Rating', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.ylim(0, max(rating_counts.values) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '3_rating_distribution.png'), dpi=300)
plt.close()
print("Saved: 3_rating_distribution.png")

# Visualization 4: Rating vs Sentiment (Stacked Bar Chart)
plt.figure(figsize=(9, 6))
rating_sent = pd.crosstab(df_clean['Rating'], df_clean['Sentiment'])[['Positive', 'Neutral', 'Negative']]
rating_sent_pct = rating_sent.div(rating_sent.sum(axis=1), axis=0) * 100

bottom = np.zeros(len(rating_sent))
for sent in ['Positive', 'Neutral', 'Negative']:
    vals = rating_sent_pct[sent].values
    plt.bar(rating_sent.index.astype(str) + ' Star', vals, bottom=bottom,
            label=sent, color=palette[sent], edgecolor='white', width=0.55)
    # Add text labels inside bars
    for idx, (b, v) in enumerate(zip(bottom, vals)):
        if v > 6:
            plt.text(idx, b + v/2, f'{v:.1f}%', ha='center', va='center',
                     color='white' if sent != 'Neutral' else 'black', fontweight='bold', fontsize=10)
    bottom += vals

plt.title('4. Rating vs Sentiment Distribution (Normalized Stacked Bar)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Customer Rating', fontsize=12)
plt.ylabel('Percentage of Reviews (%)', fontsize=12)
plt.legend(title='Sentiment', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '4_rating_vs_sentiment.png'), dpi=300)
plt.close()
print("Saved: 4_rating_vs_sentiment.png")

# Visualization 5: Review Length Analysis (Histogram)
plt.figure(figsize=(10, 5))
sns.histplot(df_clean['word_count'], bins=50, kde=True, color='#2980b9', edgecolor='black', alpha=0.6)
plt.axvline(df_clean['word_count'].median(), color='red', linestyle='--', linewidth=2,
            label=f'Median Word Count: {df_clean["word_count"].median():.0f}')
plt.axvline(df_clean['word_count'].mean(), color='orange', linestyle='-', linewidth=2,
            label=f'Mean Word Count: {df_clean["word_count"].mean():.1f}')
plt.title('5. Distribution of Review Length (Word Count)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Number of Words in Review', fontsize=12)
plt.ylabel('Review Count', fontsize=12)
plt.xlim(0, 150)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '5_review_length_histogram.png'), dpi=300)
plt.close()
print("Saved: 5_review_length_histogram.png")

# Visualization 6: Sentiment vs Review Length (Box Plot)
plt.figure(figsize=(9, 6))
sns.boxplot(x='Sentiment', y='word_count', data=df_clean, order=['Positive', 'Neutral', 'Negative'],
            palette=palette, showmeans=True,
            meanprops={'marker':'o', 'markerfacecolor':'white', 'markeredgecolor':'black', 'markersize':8})
plt.title('6. Review Length (Word Count) Across Sentiment Categories', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Sentiment Category', fontsize=12)
plt.ylabel('Word Count', fontsize=12)
plt.ylim(0, 120)  # focus on core distribution without extreme outliers
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '6_sentiment_vs_review_length.png'), dpi=300)
plt.close()
print("Saved: 6_sentiment_vs_review_length.png")

# Visualization 7: Word Clouds for Positive, Negative, Neutral
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
cmaps = {'Positive': 'Greens', 'Negative': 'Reds', 'Neutral': 'YlOrBr'}

for ax, sent in zip(axes, ['Positive', 'Negative', 'Neutral']):
    text_corpus = " ".join(df_clean[df_clean['Sentiment'] == sent]['cleaned_review'])
    # Filter stopwords
    filtered_words = [w for w in text_corpus.split() if w not in stopwords_custom and len(w) > 2]
    filtered_text = " ".join(filtered_words)
    wc = WordCloud(width=600, height=400, background_color='white', colormap=cmaps[sent],
                   max_words=80, random_state=42).generate(filtered_text)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(f'{sent} Reviews Word Cloud', fontsize=15, fontweight='bold', pad=10)
    ax.axis('off')

plt.suptitle('7. Word Clouds by Customer Sentiment Category', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '7_word_clouds.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 7_word_clouds.png")

# Visualization 8: Most Frequent Words (Horizontal Bar Charts)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, sent in zip(axes, ['Positive', 'Negative', 'Neutral']):
    words = []
    for r in df_clean[df_clean['Sentiment'] == sent]['cleaned_review']:
        words.extend([w for w in str(r).split() if w not in stopwords_custom and len(w) > 2])
    counter = Counter(words)
    top15 = counter.most_common(12)
    w_df = pd.DataFrame(top15, columns=['word', 'count']).sort_values('count', ascending=True)
    ax.barh(w_df['word'], w_df['count'], color=palette[sent], edgecolor='black', alpha=0.85)
    ax.set_title(f'Top Words in {sent} Reviews', fontsize=13, fontweight='bold')
    ax.set_xlabel('Frequency', fontsize=11)
    for idx, val in enumerate(w_df['count']):
        ax.text(val + 3, idx, str(val), va='center', fontsize=9, fontweight='bold')

plt.suptitle('8. Most Frequent Words by Sentiment Category (Filtered Meaningful Vocabulary)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '8_most_frequent_words.png'), dpi=300, bbox_inches='tight')
plt.close()
print("Saved: 8_most_frequent_words.png")

# Visualization 9: Confusion Matrix Heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels_order, yticklabels=labels_order,
            cbar=True, annot_kws={'size': 14, 'weight': 'bold'})
plt.title('9. Confusion Matrix: Logistic Regression Model', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Predicted Sentiment Label', fontsize=12)
plt.ylabel('Ground Truth Sentiment Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '9_confusion_matrix.png'), dpi=300)
plt.close()
print("Saved: 9_confusion_matrix.png")

# Visualization 10: Model Performance Metrics
metrics = {
    'Accuracy': acc,
    'Weighted Precision': prec_w,
    'Weighted Recall': rec_w,
    'Weighted F1-Score': f1_w,
    'Macro Precision': prec_m,
    'Macro Recall': rec_m,
    'Macro F1-Score': f1_m
}

plt.figure(figsize=(9, 5))
metric_names = list(metrics.keys())
metric_vals = [v * 100 for v in metrics.values()]
colors_m = ['#2980b9', '#27ae60', '#e67e22', '#8e44ad', '#16a085', '#d35400', '#2c3e50']
bars = plt.bar(metric_names, metric_vals, color=colors_m, edgecolor='black', width=0.6, alpha=0.85)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 1.5, f'{height:.2f}%',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.title('10. Sentiment Classifier Performance Metrics (Test Set)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Evaluation Metric', fontsize=12)
plt.ylabel('Score (%)', fontsize=12)
plt.ylim(0, 100)
plt.xticks(rotation=25, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '10_model_performance.png'), dpi=300)
plt.close()
print("Saved: 10_model_performance.png")

# Visualization 11: Additional Feature Analysis (Extracted Aspects vs Sentiment)
all_feat_records = []
for idx, row in df_clean.iterrows():
    try:
        f_list = ast.literal_eval(row['Features'])
        if isinstance(f_list, list):
            for feat in f_list:
                f_norm = feat.lower().strip()
                if f_norm and len(f_norm) > 2:
                    all_feat_records.append({'feature': f_norm, 'sentiment': row['Sentiment']})
    except Exception:
        pass

feat_df = pd.DataFrame(all_feat_records)
top_feats = feat_df['feature'].value_counts().head(10).index.tolist()
feat_sub = feat_df[feat_df['feature'].isin(top_feats)]
feat_crosstab = pd.crosstab(feat_sub['feature'], feat_sub['sentiment'])[['Positive', 'Neutral', 'Negative']]
feat_crosstab = feat_crosstab.loc[top_feats]

plt.figure(figsize=(11, 6))
feat_crosstab.plot(kind='bar', stacked=True, color=[palette['Positive'], palette['Neutral'], palette['Negative']],
                   edgecolor='black', figsize=(11, 6), width=0.65)
plt.title('11. Top Extracted Aspects/Features by Customer Sentiment', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('Extracted Product / Service Aspect', fontsize=12)
plt.ylabel('Mention Count', fontsize=12)
plt.legend(title='Sentiment', loc='upper right')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(vis_dir, '11_additional_features.png'), dpi=300)
plt.close()
print("Saved: 11_additional_features.png")

print("\nALL 11 VISUALIZATIONS GENERATED AND SAVED SUCCESSFULLY!")
