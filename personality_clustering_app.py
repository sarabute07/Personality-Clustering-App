import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="MBTI Personality Clustering",
    page_icon="🧠",
    layout="wide",
)

st.title("MBTI Personality Clustering & Analysis")
st.markdown("""
An interactive machine learning project analyzing personality-linked writing patterns.<br>
Powered by <strong>TF-IDF</strong>, <strong>K-Means clustering</strong>, and <strong>PCA visualization</strong>.
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload MBTI dataset (mbti_1.csv)", type=["csv"])
if not uploaded_file:
    st.info("Please upload `mbti_1.csv` (from Kaggle).")
    st.stop()

df = pd.read_csv(uploaded_file)
df['clean_posts'] = (
    df['posts']
    .astype(str)
    .str.replace(r'http\S+', '', regex=True)
    .str.replace(r'[^a-zA-Z\s]', '', regex=True)
    .str.lower()
)

st.subheader("Dataset preview")
st.dataframe(df[['type', 'posts']].head(6))

st.sidebar.header("Settings")
k = st.sidebar.slider("Number of clusters (K)", min_value=2, max_value=8, value=4)
max_features = st.sidebar.slider("TF-IDF max features", 500, 3000, 1000, step=250)

vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
X = vectorizer.fit_transform(df['clean_posts'])

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)
df['Cluster'] = clusters

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X.toarray())

st.subheader("Model summary")
sil_score = silhouette_score(X, clusters)
st.markdown(f"- **Silhouette score:** {sil_score:.3f}")
explained = pca.explained_variance_ratio_.sum()
st.markdown(f"- **PCA variance captured (2 comps):** {explained:.3f}")

st.subheader("Clusters (PCA reduced 2D)")
fig_scatter, ax = plt.subplots(figsize=(8, 5))
palette = sns.color_palette("husl", k)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette=palette, s=30, ax=ax, legend="full")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_title("Personality clusters (PCA reduced)")
ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
st.pyplot(fig_scatter)

st.subheader("Cluster composition & interpretation")
cluster_summaries = {}
cols = st.columns(min(k, 4))
for i in range(k):
    types = df.loc[df['Cluster'] == i, 'type']
    counts = types.value_counts().head(8)
    cluster_summaries[i] = counts
    with cols[i % len(cols)]:
        st.markdown(f"### Cluster {i}")
        if not counts.empty:
            fig_bar, ax_bar = plt.subplots(figsize=(4, 3))
            counts.plot(kind='bar', ax=ax_bar, color="#5DADE2")
            ax_bar.set_ylabel("Count")
            ax_bar.set_xlabel("")
            ax_bar.set_xticks(range(len(counts.index)))
            ax_bar.set_xticklabels(counts.index, rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig_bar)
            st.markdown(f"**Most common MBTI:** {counts.index[0]}  ({counts.iloc[0]} samples)")
        else:
            st.write("No samples in this cluster")

st.markdown("""
**Interpretation guide** (example labels — adjust after exploring data):
- Cluster with many INFP / ENFP: creative, emotional, value-driven writers.
- Cluster with many INTJ / INTP: analytical, idea-focused writers.
- Cluster with many ESFJ / ENFJ: social, people-oriented writers.
""")

st.subheader("Try your text — see which cluster it fits into")
user_text = st.text_area("Paste a short paragraph (1-6 sentences)", height=120)
if st.button("Predict cluster for my text"):
    if not user_text.strip():
        st.error("Please paste some text first.")
    else:
        import re
        txt = re.sub(r'http\S+', '', user_text)
        txt = re.sub(r'[^a-zA-Z\s]', '', txt).lower()
        x_user = vectorizer.transform([txt])
        pred = kmeans.predict(x_user)[0]
        st.markdown(f"### Predicted cluster: **{pred}**")

        top_types = cluster_summaries.get(pred)
        if top_types is not None and not top_types.empty:
            st.markdown("**Cluster top MBTI types:**")
            st.table(top_types.head(6).rename_axis("MBTI").reset_index().rename(columns={0: "Count"}))
        else:
            st.write("No cluster summary available.")

st.subheader("Download results")
if st.button("Download CSV with cluster labels"):
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(label="Download labeled dataset (.csv)", data=csv, file_name="mbti_labeled.csv", mime="text/csv")

st.success("Finished running. Explore clusters and tweak K or TF-IDF max features from the sidebar.")
