from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

def get_embedding_model() -> SentenceTransformerEmbeddings:
    
    """Defines the model for embedding

    Returns:
        SentenceTransformerEmbeddings: The model being used for embedding vector data
    """
    
    embedding_model = SentenceTransformerEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    return embedding_model

def load_vector_db(collection_name:str, embedding_model: SentenceTransformerEmbeddings):
    
    """Loads the created vector database

    Returns:
        Vector Database: The vector database being loaded
    """
    
    vector_db = Chroma( 
        persist_directory="vector_db",
        collection_name=collection_name,
        embedding_function=embedding_model,
    )
    
    return vector_db
    

def get_chat_model(temperature: float=0.5, model_name: str="gpt-3.5-turbo-1106") -> ChatOpenAI:
    
    """Defines the chat model and its parameters

    Returns:
        ChatOpenAI: The ChatOpenAI model from langchain being used in the chatbot
    """
    
    model = ChatOpenAI(openai_api_key="OPEN_AI_API_KEY", temperature=temperature, model_name=model_name)
    
    return model


def get_chat_history_enabler() -> ConversationBufferMemory:
    
    """Creates the ability for chat history

    Returns:
        ConversationBufferMemory: The buffer for storing conversation memory
    """
    
    chat_history = ConversationBufferMemory(
        return_messages=True, memory_key="chat_history")
    
    return chat_history


def get_qa_chain(model: ChatOpenAI, chat_history: ConversationBufferMemory, vector_db) -> ConversationalRetrievalChain.from_llm:
    
    """Creates the answer retrieval chain for the questions

    Returns:
        ConversationalRetrievalChain.from_llm: The question-answer chain
    """

    question_answer_chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        memory=chat_history,
        retriever=vector_db.as_retriever(
            search_type='mmr', search_kwargs={"k": 1}),
        chain_type="refine",
    )
    
    return question_answer_chain


def get_response(question: str) -> str:
    
    """Gets a response from the llm based on the prompt

    Returns:
        str: The response from the LLM
    """
    
    # embedder = embedding_model()
    
    vector_database = load_vector_db("fin_f1_data", embedding_model=get_embedding_model)

    # chatbot_model = chat_model()
    
    # chat_history = chat_history_implementation()

    qa_chain = get_qa_chain(model=get_chat_model, chat_history=get_chat_history_enabler, vector_db=vector_database)
    
    context = """
    You are a chatbot. Your name is Arush. It is very important to keep your answers to 3 sentences or less.
    Never discuss the "context", simply take information from the provided data and respond.
    Your purpose is to introduce yourself based on the information in the documents provided.
    Make sure to stay polite at all times. Keep you answers short and only answer the exact question asked.
    Answer every question organically, even if it means repeating yourself.
    Keeping in mind this information, answer this question from the user: 
    
    
    """
    
    prompt = context + question
    
    response = qa_chain.invoke({"question": prompt})
    
    return (response.get("answer", ""))

    
