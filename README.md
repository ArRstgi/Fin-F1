## What is Fin-F1?

Fin-F1 is an LLM powered chatbot that allowers users to chat with the data from the Formula 1 Power Unit Financial Regulations Issue 6 (06/11/24). It is intended to provide an easy and quick way for people to get information from a document that might be difficult to parse manually.

##  Running the app

### With Docker:

Run the following line in the terminal:

`docker compose up --build`

Then go to this link to use the chatbot:

[http://localhost:8501/](http://localhost:8501/)

### Without Docker:

Create a python virtual environment:

`python3 -m venv .venv`

Activate the virtual environment:

`. .venv/bin/activate`

Run the following line in the terminal:

`python -m pip install -r requirements.txt`

Then run:

`streamlit run app.py --server.port=8501 --server.address=0.0.0.0`

Then go to this link to use the chatbot:

[http://localhost:8501/](http://localhost:8501/)

# Documentation 


## 1. Introduction

For my first Generative AI app, I have created a chatbot that leverages OpenAI's GPT-4o model to provide information about the Formula 1 Power Unit Financial Regulations. For developing the chatbot, I have used the LangChain framework and the streamlit library.
### 1.1 LangChain

I chose to use LangChain because of its ease of use and guides when it comes to making chatbots and applying Retrieval-Augmented Generation (RAG) techniques. 

### 1.2 Initial Ideation

Initially, I was planning to create a chatbot using a local model like llama-2-7b using the Ollama chat model from LangChain to avoid the need to use pain LLM APIs. However, I realized that there were more issues with this approach:

1. Firstly, it was harder to find relevant documentation and information on using LangChain to create an Ollama chatbot, which would have made the process longer and more difficult.
2. Secondly, I was unsure whether I would be able to reasonably run the llama 2 model on my current laptop. 
2. This led me to realizing that it would force the user to also have the capability to run the model locally, which wasn't a guarantee. 
3. Finally, the Open AI API and other LLM API's are the industry standard for building Generative AI applications and the value gained from some experience in using them would far outweigh the cost of some LLM calls.

Therefore, I decided to use the Open AI API to get access to gpt-4o for my chatbot.


## 2. Database Creation

![database_creation_fin.jpg](https://github.com/ArRstgi/Fin-F1/blob/f1-update/documentation/database_creation_fin.jpg)

### 2.1 Data

In terms of data, I have used a pdf download of the Formula 1 Power Unit Financial Regulations Issue 6 from the [FIA Website](https://www.fia.com/sites/default/files/fia_formula_1_pu_financial_regulations_-_issue_6_-_2024-06-11.pdf)

### 2.2 Embedding

I have stored the data as a vector database due to the speed with which data can be retrieved from them. For embedding my pdf document, I have not used OpenAI's text embeddings since it can be done with a simple open-source, hugging face model. This reduces to cost through fewer LLM calls and accomplishes the goal equally well. I have used the all-MiniLM-L6-v2 model from hugging face for this purpose.

### 2.3 Creating the Vector Database

I have decided to use ChromaDB to create the vector database since it is fully integrated with LangChain and the embedding process is very simple. Additionally, searching techniques such as similarity searches and maximum marginal relevance searches are easily applied.

## 3. Retrieval Techniques

The next decision I faced was deciding which retrieval technique to use to get data from the vector database. The following were the 2 options I was considering:
### 3.1 Semantic Similarity Search

Semantic Similarity Search is the most basic techniques used to retrieve data from the database. It takes the query entered by the user and embeds it as vectors, just like the data is in the database. Then, it finds the data in the database that is closest to the vector data of the embedded query. This method is extremely simple and provides correct responses, but it lacks diversity and tends to give the same answer repeatedly since it is a hardcoded distance being measured.

### 3.2 Maximum Marginal Relevance (MMR)

MMR carries additional considerations than the Semantic Similarity Search. It also looks for data which is similar to the query entered by the user but it also considers how much new information the document brings into the answer. Therefore, it brings diversity into the results, making the chatbot more engaging to communicate with. This is a clear benefit, but it has the side effect of a loss off accuracy.

### 3.3 My Decision

I chose to go with Semantic Similarity search since accuracy is the key for me. The document I am referencing contains complicated jargon and I need to make sure I am retrieving exactly what the user is asking for.


## 4. Prompt Creation

### 4.1 Few-shot prompt template

I used a few shot prompt template since I was able to provide my chatbot with some example query and answers so that it could act as a reference for how it should answer other questions from the user

### 4.2 Chat prompt template

Finally, I used to the few-shot prompt template withing a chat prompt template along with an initial system message and a section for the context that will be retrieved from the vector database (relevant information from the original document)


## 5. Final Steps

### 5.1 Creating the chain

Here, I simply used the LCEL to create a simple chain based on the prompt template, prompt and context given earlier.

### 5.2 Building the streamlit app

I built a streamlit application that takes in the user's query's and passes it through the RAG pipeline before displaying the response.

### 5.3 Dockerizing

Finally, I used the docker init command in the terminal to create the files required to dockerize my app. Then, I wrote the Dockerfile so that it would run with the current code and directories. 


## 6. Full Architecture Diagram

![full_fin.jpg](https://github.com/ArRstgi/Fin-F1/blob/f1-update/documentation/full_fin.jpg)

## 7. Evaluation

### 7.1 Areas for future improvement

1. ~~The Chatbot struggles to follow up on answers when asked questions such as "why" or "what" or similar that do not have specific details that can be searched for.~~ **Currently part of issue 9**

2. ~~The addition of context by concatenating 2 strings is simple and workable, but finding a more elegant solution could be better.
	1. Perhaps using the PromptTemplate or ChatPromptTemplate classes from LangChain.~~ **Implemented**

3. ~~This message is being displayed whenever a question is asked - "Number of requested results 20 is greater than number of elements in index 3, updating n_results = 3".~~
	~~1. The chatbot works perfectly but this message is sent in the terminal.~~
	~~2. It would be good to resolve this message so that it doesn't continuously appear.~~ **Done**

4. Tests (unittest, pytest) could be written to ensure each component of the chatbot is working properly.

5. Methods such as Contextual Compression or Map Reduce could be used to improve results if the resources for more LLM calls are provided.

6. Further advanced retrieval techniques such as Small-to-Big Retrieval in terms or Parent Document Retrieval or Sentence Window Retrieval could be implemented to increase the chatbot's capabilities.

7. ~~Currently, the PDF document that is providing the data being used for RAG is 3 pages long. More data would allow for the chatbot to have a wider range of possible responses, which would improve its usability.~~ **Currently not an issue**

8. The streamlit app could be converted into a custom web app using frontend tools.
9. Implementing chat history with LCEL, along with the ability for the LLM to easily answer follow up questions as simple as - "why" or "what"
10. Use technique's such as intent detection and entity extraction to improve the Chatbot's information retrieval capabilities
11. Add the ability for the chatbot do display a source - which is the part of the document from where the chatbot got it's answer, so that users can check whether the answer is correct or not.
