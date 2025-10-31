<h2>MBTI Personality Clustering & Analysis</h2>

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

  <img width="2814" height="1169" alt="image" src="https://github.com/user-attachments/assets/c79b6a97-4e35-4257-8267-26380936b2c7" />

<img width="1463" height="754" alt="Screenshot 2025-11-01 at 12 10 53 AM" src="https://github.com/user-attachments/assets/b17f59be-7404-4abb-b8d9-f0092b2e7963" />


</p>

<h3>Key Features</h3>

<ul>
  <li>Uses <strong>TF-IDF vectorization</strong> to analyze writing patterns.</li>
  <li>Groups similar personalities using <strong>K-Means clustering</strong>.</li>
  <li>Reduces dimensions via <strong>PCA</strong> for 2D visualization.</li>
  <li>Displays cluster insights and dominant
