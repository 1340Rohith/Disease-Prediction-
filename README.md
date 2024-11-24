<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body>
    <h1>Disease Diagnosis Web App</h1>
    <p>This project uses a Streamlit application to diagnose diseases based on user symptoms. It leverages logistic regression and Naive Bayes models to predict the disease and provides probability scores for the predictions.</p>
    <h2>Project Structure</h2>
    <ul>
        <li><code>app.py</code>: The main Streamlit application.</li>
        <li><code>train.py</code>: Contains preprocessing functions and model training logic.</li>
        <li><code>requirements.txt</code>: Lists the dependencies required for the project.</li>
        <li><code>NLM.csv</code>: Dataset used for training the models.</li>
    </ul>
    <h2>Installation</h2>
    <p>To run this project locally, follow these steps:</p>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone https://github.com/your-username/disease-diagnosis-web-app.git</code></pre>
        </li>
        <li>Navigate to the project directory:
            <pre><code>cd disease-diagnosis-web-app</code></pre>
        </li>
        <li>Install the required dependencies:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
        <li>Run the Streamlit app:
            <pre><code>streamlit run app.py</code></pre>
        </li>
    </ol>
    <h2>Usage</h2>
    <p>Once the Streamlit app is running, open your browser and go to <a href="http://localhost:8501">http://localhost:8501</a>. Enter your symptoms in the input box and click on the "Submit" button to get the diagnosis and probability score.</p>
    <h2>Features</h2>
    <ul>
        <li>Text preprocessing using various NLP techniques.</li>
        <li>Model prediction using logistic regression and Naive Bayes.</li>
        <li>Comparison of probabilities from both models to determine the final prediction.</li>
        <li>Displays "Need more info" if the highest probability is below 65%.</li>
    </ul>
    <h2>Contributing</h2>
    <p>Contributions are welcome! Please fork the repository and submit a pull request.</p>
</body>
</html>
