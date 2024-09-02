from flask import Flask, request, jsonify
from chatbot import Conversational_Chain

llm  = Conversational_Chain()
app = Flask(__name__)

@app.route('/invoke', methods=['POST'])
def invoke_conversational_rag_chain():
    # Get the input data from the request
    data = request.get_json()

    # Ensure the input is provided
    if 'input' not in data:
        return jsonify({'error': 'No input provided'}), 400

    # Set up the configuration, if any
    config = data.get('config', {})

    # Invoke the conversational RAG chain
    result = llm.invoke(
        {"input": data['input']},
        config=config
    )

    # Extract the answer from the result
    response = {'answer': result.get('answer')}

    # Return the response as JSON
    return jsonify(response)


if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5500, debug=True)
