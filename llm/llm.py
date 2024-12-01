# from langchain_community.document_loaders.llmsherpa import LLMSherpaFileLoader

# def load_doc_Sherpa(link_file):
#     loader = LLMSherpaFileLoader(
#         file_path=link_file,
#         new_indent_parser=True,
#         apply_ocr=True,
#         strategy="chunks",
#         llmsherpa_api_url="http://llmsherpa.service.consul:15001/api/parseDocument?renderFormat=all",
#     )
#     docs = loader.load()
#     return docs

from langchain_huggingface import HuggingFaceEmbeddings

model_name_Transformer_384 = "paraphrase-MiniLM-L6-v2"

def get_embedding_Transformer(model_name):
    model_name = model_name
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

from langchain_chroma import Chroma
from docx import Document

def vector_data_chroma(docs=Document(), embeddings=[]):
    retriever = Chroma.from_documents(docs, embeddings).as_retriever(search_type="mmr")

    return retriever

from langchain_core.prompts import PromptTemplate
 
prompt_template = """
Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
You may receive vietnamese misspelling question. Try to check it. 
When you receive a question, your answer must in 6 labels: "Mở cửa", "Đóng cửa", 
"bật đèn một", "tắt đèn một", "bật đèn hai", "tắt đèn hai", "bật quạt", "tắt quạt" 

Context: {context}
Question: {question}

Answer the question and don't provide any additional information. Be succinct.

Responses should be properly formatted to be easily read.
"""

prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

from langchain_groq import ChatGroq
import os

groq_api_key = os.environ.get("GROQ_API_KEY")
llm_groq = ChatGroq(temperature=0,
               api_key = groq_api_key,
               model_name="llama3-70b-8192"
               )

from langchain_cohere import CohereRerank
from langchain.retrievers import ContextualCompressionRetriever
def create_compression_retriever_CohereRerank(retriever=[]):
    api_cohere = os.environ.get("COHERE_API_KEY")
    compressor = CohereRerank(cohere_api_key=api_cohere, model="rerank-english-v3.0")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    return compression_retriever


from langchain.chains import RetrievalQA

def create_AI_agent(llm=[], compression_retriever=[], verbose=False):    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=compression_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt, "verbose": verbose},
    )
    
    return qa