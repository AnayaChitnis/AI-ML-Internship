from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the salary prediction model
with open(r'C:/Users/siddh/OneDrive/Desktop/Git/anaya/SALARY.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        data = request.form
        
        # Get user input from the form
        field = int(data.get("field"))  # Assuming field is encoded as integer
        post = int(data.get("post"))    # Assuming post is also an integer
        
        # Pass the features to the model
        user_input = np.array([[field, post]])
        
        # Make the prediction
        model_output = model.predict(user_input)
        
        # Multiply the prediction by 1000
        prediction = f"Predicted Salary: ${model_output[0] * 1000:,.2f}"
        
        return render_template('index.html', prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=8000)
