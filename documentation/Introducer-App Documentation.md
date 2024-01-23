
# 1. Introduction

For my Generative AI app prototype, I have created a chatbot that leverages OpenAI's GPT-3.5-Turbo model for LLM tasks. For developing the chatbot, I have used the LangChain framework and the streamlit library.
### 1.1 LangChain

I chose to use LangChain because of its ease of use and guides when it comes to making chatbots and applying Retrieval-Augmented Generation (RAG) techniques. 

### 1.2 Initial Ideation

Initially, I was planning to create a chatbot using a local model like llama-2-7b using the Ollama chat model from LangChain to avoid the need to use pain LLM APIs. However, I realized that there were more issues with this approach:

1. Firstly, it was harder to find relevant documentation and information on using LangChain to create an Ollama chatbot, which would have made the process longer and more difficult.
2. Secondly, I was unsure whether I would be able to reasonably run the llama 2 model on my current laptop. 
2. This led me to realizing that it would force the user to also have the capability to run the model locally, which wasn't a guarantee. 
3. Finally, the Open AI API and other LLM API's are the industry standard for building Generative AI applications and the value gained from some experience in using them would far outweigh the cost of some LLM calls.

Therefore, I decided to use the Open AI API to get access to gpt-3.5-turbo for my chatbot.


# 2. Database Creation

![[database_creation.jpg]]
### 2.1 Data

In terms of data, I have simply written out a word document with some information about myself (hobbies, studies, food preferences) along with the data from my resume. I converted this document and it has been used as the external source for facts about myself.

### 2.2 Embedding

I have stored the data as a vector database due to the speed with which data can be retrieved from them. For embedding my pdf document, I have not used OpenAI's text embeddings since it can be done with a simple open-source, hugging face model. This reduces to cost through fewer LLM calls and accomplishes the goal equally well. I have used the all-MiniLM-L6-v2 model from hugging face for this purpose.

### 2.3 Creating the Vector Database

I have decided to use ChromaDB to create the vector database since it is fully integrated with LangChain and the embedding process is very simple. Additionally, searching techniques such as similarity searches and maximum marginal relevance searches are easily applied.


# 3. Retrieval Techniques

![[retrieval_chain.jpg]]

The next decision I faced was deciding which retrieval technique to use to get data from the vector database. The following were the 3 options I was considering:
### 3.1 Semantic Similarity Search

Semantic Similarity Search is the most basic techniques used to retrieve data from the database. It takes the query entered by the user and embeds it as vectors, just like the data is in the database. Then, it finds the data in the database that is closest to the vector data of the embedded query. This method is extremely simple and provides correct responses, but it lacks diversity and tends to give the same answer repeatedly since it is a hardcoded distance being measured.

### 3.2 Maximum Marginal Relevance (MMR)

MMR carries additional considerations than the Semantic Similarity Search. It also looks for data which is similar to the query entered by the user but it also considers how much new information the document brings into the answer. Therefore, it brings diversity into the results, making the chatbot more engaging to communicate with. This is a clear benefit, but it has the side effect of a loss off accuracy.

### 3.3 Contextual Compression

Contextual Compression is an extremely powerful technique. It takes all the documents of data you have and passes them through the LLM and compresses them until only two or three sentences are left with the exact data that the user is asking for in their query. Then, these two or three sentences are sent to the LLM to frame a response for the user. The huge downside to this technique is the large amount of LLM calls that have to be made to compress the data, even before the answer is even being framed.

### 3.4 My Decision

I chose to go with MMR. The diversity it offered made it instantly a better option than the semantic similarity search for me. I was also not willing to have to make the large number of LLM calls needed for contextual compression.


# 4. Question Answering Techniques

Here, I needed to choose between 2 question answering techniques
### 4.1 Map Reduce

In Map Reduce, every document/chunk of data is sent to the LLM and each of them get their own answer to the query asked by the user. Then, all these answers are combined into one final answer. Again, similar to Contextual Compression,  this is a powerful technique but it involves many LLM calls

### 4.2 Refine

The other option here is the Refine technique. Here, the first document/chunk of data is passed alongside the user query to the LLM and an answer is generated. Then this answer is passed alongside the next document/chunk of data to the LLM. The LLM then refines the answer based on the new document/chunk of data. This continues until all the documents/chunks have been passed through the LLM to arrive at a final answer. This is also a powerful technique and it uses fewer LLM calls

### 4.3 My Decision

I went with the Refine technique since it involves fewer LLM calls while still being reasonable powerful.


# 5. Final Steps

### 5.1 Adding a conversational chain & memory

Here, I simply used the ConversationBufferMemory object from LangChain to create an object that stores chat history. Then, when creating the ConversationalRetrievalChain, I passed in this object for the memory parameter. This allows the LLM to keep track of the last 5 conversations with the chatbot, adding a degree of chat history to the chatbot.

### 5.2 Adding context to a prompt

To give the bot instructions before it took in the user query, I simply made a string variable with all the instructions written out. Then, when creating the prompt I concatenated the context with the prompt and sent that to the LLM as a query.

### 5.3 Building the streamlit app

I built a streamlit application that takes in the user's query's and passes it through the RAG pipeline before displaying the response.

### 5.4 Dockerizing

Finally, I used the docker init command in the terminal to create the files required to dockerize my app. Then, I wrote the Dockerfile so that it would run with the current code and directories. 


# 6. Full Architecture Diagram

![[full.jpg]]


# 7. Evaluation

### 7.1 Areas for future improvement

1. The Chatbot struggles to follow up on answers when asked questions such as "why" or "what" or similar that do not have specific details that can be searched for.

2. The addition of context by concatenating 2 strings is simple and workable, but finding a more elegant solution could be better.
	1. Perhaps using the PromptTemplate or ChatPromptTemplate classes from LangChain.

3. This message is being displayed whenever a question is asked - "Number of requested results 20 is greater than number of elements in index 3, updating n_results = 3".
	1. The chatbot works perfectly but this message is sent in the terminal.
	2. It would be good to resolve this message so that it doesn't continuously appear.

4. Tests (unittest, pytest) could be written to ensure each component of the chatbot is working properly.

5. Methods such as Contextual Compression or Map Reduce could be used to improve results if the resources for more LLM calls are provided.

6. Further advanced retrieval techniques such as Small-to-Big Retrieval in terms or Parent Document Retrieval or Sentence Window Retrieval could be implemented to increase the chatbot's capabilities.

7. Currently, the PDF document that is providing the data being used for RAG is 3 pages long. More data would allow for the chatbot to have a wider range of possible responses, which would improve its usability.

8. The streamlit app could be converted into a custom web app using frontend tools.
