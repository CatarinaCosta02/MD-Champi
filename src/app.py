from flask import Flask, request, jsonify
from chat import chatbot

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    # Extract the message from the POST request
    message = request.json.get('message', '')
    
    response = chatbot(message)
    
    # Return the response as JSON
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
