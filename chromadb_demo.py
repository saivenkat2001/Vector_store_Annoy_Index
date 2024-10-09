import re
from chromadb import PersistentClient
from chromadb.utils import embedding_functions
import PyPDF2

def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text()
        print(f"Extracted {len(text)} characters from PDF.")
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def create_short_sentences(text, max_words=30):
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    sentence_list = []
    for sentence in sentences:
        sentence = re.sub(r"\\u[e-f][0-9a-z]{3}|\n", " ", sentence)
        sentence = re.sub(r'[^\w\s]', '', sentence.lower()) 
        words = sentence.split()
        if len(words) > max_words:
            sentence = ' '.join(words[:max_words]) + "..."
        sentence_list.append(sentence.strip())
        
    print(f"Created {len(sentence_list)} truncated sentences.")
    return sentence_list

def chroma_db(query, max_words_per_result=50):
    try:
        text = extract_text_from_pdf("Cat.pdf")
        if not text:
            return [], []
        
        sentences = create_short_sentences(text)
        client = PersistentClient(path="output/cat")
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        collection = client.get_or_create_collection(name="Cat_collection", embedding_function=embedding_function)
        print(f"Collection '{collection.name}' created or retrieved.")
        
        if collection.count() == 0:
            document_ids = [f'id{idx}' for idx in range(1, len(sentences) + 1)]
            collection.upsert(documents=sentences, ids=document_ids)
            print(f"Upserted {len(sentences)} documents to the collection.")
        else:
            print(f"Collection already contains {collection.count()} documents.")
        
        results = collection.query(query_texts=[query], n_results=3)
        print(f"Query '{query}' returned {len(results['documents'][0])} results.")
        
        shortened_documents = []
        for doc in results["documents"][0]:
            words = doc.split()
            if len(words) > max_words_per_result:
                shortened_doc = ' '.join(words[:max_words_per_result]) + "..."
            else:
                shortened_doc = doc
            shortened_documents.append(shortened_doc)
        
        return shortened_documents, results["distances"][0]
    except Exception as e:
        print(f"Error in chroma_db function: {e}")
        return [], []

if __name__ == '__main__':
    documents, scores = chroma_db("How many legs does a cat have?", max_words_per_result=50)
    
    if documents:
        for i, (doc, score) in enumerate(zip(documents, scores), 1):
            print(f"{i}. Score: {score:.4f}\n   Document: {doc}\n")
    else:
        print("No results found.")
