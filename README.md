<h2>MBTI Personality Clustering & Analysis</h2>

<p>
🔗 <a href="https://your-streamlit-link.streamlit.app" target="_blank">Live Demo on Streamlit</a>
</p>

<p>
This project performs clustering on MBTI (Myers-Briggs Type Indicator) personality data using machine learning techniques.<br>
It identifies patterns in people’s writing styles and groups them into personality clusters.<br>
The project is implemented as an interactive <strong>Streamlit</strong> web app.
</p>

<p>
This app analyzes MBTI personality types using clustering on real-world text data 
(<em>MBTI dataset from Kaggle</em>).
</p>

<h3>Overview</h3>

<p>
The goal of this project is to uncover hidden personality groupings by applying 
unsupervised learning on text data. Users can explore clusters, view their dominant MBTI types, 
and test which cluster their own writing fits into.
  <img width="1331" height="826" alt="Screenshot 2025-11-01 at 12 10 37 AM" src="https://github.com/user-attachments/assets/35202fcd-8788-4561-a308-9fbfce13ab5f" />

</p>

<h3>Key Features</h3>

<ul>
  <li>Uses <strong>TF-IDF vectorization</strong> to analyze writing patterns.</li>
  <li>Groups similar personalities using <strong>K-Means clustering</strong>.</li>
  <li>Reduces dimensions via <strong>PCA</strong> for 2D visualization.</li>
  <li>Displays cluster insights and dominant
