# AI Finance Advisor
This project aims to create an AI-powered financial advisor that can help users analyze their financial data and make informed decisions.

## Functionality
The advisor can:

- Process user-uploaded CSV files containing financial data.
- Analyze the data and provide insights and recommendations.
- Respond to user questions in a conversational manner.
## How to Run
1. Download the project files:
Clone or download the project repository to your local machine.

2. Install the required libraries:
````pip install -r requirements.txt````

3. Create .env file

4. Run the main script:
````streamlit run main.py````

5. Access the application:
Open your web browser and navigate to http://localhost:8501.

6. Upload your financial data:
Click on the "Upload your financial data" button and select your CSV file.

7. Ask questions:
Type your questions in the text input field and click "Enter". The AI advisor will respond with its analysis and recommendations.

## Additional Notes
- The project uses the Ollama LLM for conversational responses.
- The FAISS vectorstore is used for efficient information retrieval.
- The ConversationBufferMemory stores the conversation history for context-aware responses.
## Disclaimer
This project is for educational purposes only and should not be used as a substitute for professional financial advice.