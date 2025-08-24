import numpy as np
import pandas as pd
import re
import time
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import optional dependencies
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

class QueryPreprocessor:
    def __init__(self):
        # Common English stopwords
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
            'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more',
            'very', 'after', 'words', 'long', 'than', 'first', 'been', 'call',
            'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part'
        }
    
    def preprocess(self, text):
        """Preprocess query text: clean, lowercase, remove stopwords"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes (for contractions)
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        words = text.split()
        filtered_words = [word for word in words if word not in self.stopwords and len(word) > 1]
        
        return ' '.join(filtered_words)

class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        
        self._initialize()
    
    def _initialize(self):
        nd = len(self.corpus)
        num_doc = 0
        
        for document in self.corpus:
            tmp = {}
            doc_words = self._tokenize(document)
            self.doc_len.append(len(doc_words))
            num_doc += len(doc_words)
            
            for word in doc_words:
                tmp[word] = tmp.get(word, 0) + 1
            self.doc_freqs.append(tmp)
            
            for word in tmp.keys():
                if word not in self.idf:
                    self.idf[word] = 0
                self.idf[word] += 1
        
        self.avgdl = num_doc / nd
        
        for word, freq in self.idf.items():
            self.idf[word] = np.log((nd - freq + 0.5) / (freq + 0.5))
    
    def _tokenize(self, text):
        return re.findall(r'\b\w+\b', text.lower())
    
    def get_scores(self, query):
        query_words = self._tokenize(query)
        scores = []
        
        for i, doc_freqs in enumerate(self.doc_freqs):
            score = 0
            doc_len = self.doc_len[i]
            
            for word in query_words:
                if word in doc_freqs:
                    freq = doc_freqs[word]
                    score += self.idf.get(word, 0) * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl))
            
            scores.append(score)
        
        return np.array(scores)

class SimpleAdvancedRAG:
    def __init__(self, csv_path: str = None):
        """Initialize the Advanced RAG system"""
        self.preprocessor = QueryPreprocessor()
        self.documents = []
        self.dense_index = None
        self.sparse_index = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.use_dense = FAISS_AVAILABLE and SENTENCE_TRANSFORMERS_AVAILABLE
        self.initialized = False
        
        # Initialize sentence transformer if available
        if self.use_dense:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"Failed to load sentence transformer: {e}")
                self.use_dense = False
        
        if csv_path:
            self.load_data(csv_path)
    
    def load_data(self, csv_path: str):
        """Load data from CSV and create indexes"""
        try:
            # Load CSV data
            df = pd.read_csv(csv_path)
            
            # Create document texts from Q&A pairs
            self.documents = []
            for _, row in df.iterrows():
                doc_text = f"Question: {row['Question']} Answer: {row['Answer']}"
                self.documents.append({
                    'text': doc_text,
                    'question': row['Question'],
                    'answer': row['Answer'],
                    'id': len(self.documents)
                })
            
            document_texts = [doc['text'] for doc in self.documents]
            
            # Try to use dense embeddings if available
            if self.use_dense:
                try:
                    document_embeddings = self.model.encode(document_texts, convert_to_tensor=False)
                    
                    # Build FAISS index
                    dim = document_embeddings.shape[1]
                    self.dense_index = faiss.IndexFlatL2(dim)
                    self.dense_index.add(np.array(document_embeddings).astype('float32'))
                    
                    # Build BM25 index
                    self.sparse_index = BM25(document_texts)
                    
                    print(f"Advanced RAG initialized with dense+sparse retrieval, {len(self.documents)} documents")
                except Exception as e:
                    print(f"Dense embedding failed, falling back to TF-IDF: {e}")
                    self.use_dense = False
            
            # Fallback to TF-IDF if dense retrieval is not available
            if not self.use_dense:
                self.tfidf_vectorizer = TfidfVectorizer(
                    stop_words='english',
                    lowercase=True,
                    ngram_range=(1, 2),
                    max_features=1000
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(document_texts)
                print(f"RAG initialized with TF-IDF retrieval, {len(self.documents)} documents")
            
            self.initialized = True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.initialized = False
    
    def retrieve(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Retrieval using available methods (dense+sparse or TF-IDF fallback)"""
        if not self.initialized:
            return []
        
        if self.use_dense:
            # Use hybrid retrieval (dense + sparse)
            return self._hybrid_retrieve(query, k, alpha)
        else:
            # Use TF-IDF fallback
            return self._tfidf_retrieve(query, k)
    
    def _hybrid_retrieve(self, query: str, k: int = 5, alpha: float = 0.7) -> List[Dict[str, Any]]:
        """Hybrid retrieval using dense + sparse methods"""
        # Preprocess query
        preprocessed_query = self.preprocessor.preprocess(query)
        
        # Dense retrieval
        query_embedding = self.model.encode([query])
        dense_distances, dense_indices = self.dense_index.search(
            np.array(query_embedding).astype('float32'), k*2
        )
        dense_scores = 1 / (1 + dense_distances[0])  # Convert distances to similarity scores
        
        # Sparse retrieval
        sparse_scores = self.sparse_index.get_scores(preprocessed_query)
        sparse_indices = np.argsort(sparse_scores)[::-1][:k*2]
        
        # Combine scores
        combined_scores = {}
        
        # Add dense scores
        for i, idx in enumerate(dense_indices[0]):
            if idx < len(self.documents):  # Safety check
                combined_scores[idx] = alpha * dense_scores[i]
        
        # Add sparse scores
        for i, idx in enumerate(sparse_indices):
            if idx < len(self.documents):  # Safety check
                if idx in combined_scores:
                    combined_scores[idx] += (1 - alpha) * sparse_scores[idx]
                else:
                    combined_scores[idx] = (1 - alpha) * sparse_scores[idx]
        
        # Sort by combined score and return top k
        sorted_indices = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for idx, score in sorted_indices:
            doc = self.documents[idx].copy()
            doc['retrieval_score'] = score
            doc['rank'] = len(results) + 1
            results.append(doc)
        
        return results
    
    def _tfidf_retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """TF-IDF based retrieval as fallback"""
        # Transform query
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-k:][::-1]
        
        # Return results with scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Minimum similarity threshold
                doc = self.documents[idx].copy()
                doc['retrieval_score'] = similarities[idx]
                doc['rank'] = len(results) + 1
                results.append(doc)
        
        return results
    
    def generate_answer(self, query: str, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Generate answer from retrieved documents"""
        if not retrieved_docs:
            return "I don't have information to answer this question about TCS financial data."
        
        # Use the best matching document
        best_doc = retrieved_docs[0]
        
        # If the score is high enough, return the answer
        if best_doc['retrieval_score'] > 0.1:
            return best_doc['answer']
        else:
            return "I found some related information, but I'm not confident it directly answers your question."
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Main function to process user queries"""
        start_time = time.time()
        
        if not self.initialized:
            return {
                'success': False,
                'answer': "RAG system is not initialized.",
                'confidence_score': 0.0,
                'response_time': time.time() - start_time,
                'retrieved_docs': []
            }
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, k=5)
        
        # Generate answer
        answer = self.generate_answer(query, retrieved_docs)
        
        # Calculate confidence score
        confidence_score = retrieved_docs[0]['retrieval_score'] if retrieved_docs else 0.0
        
        return {
            'success': True,
            'answer': answer,
            'confidence_score': confidence_score,
            'response_time': time.time() - start_time,
            'retrieved_docs': retrieved_docs
        }

# Test function
if __name__ == "__main__":
    # Test the system
    rag = SimpleAdvancedRAG("data/tcs_qa_dataset.csv")
    
    test_queries = [
        "What was TCS revenue in 2025?",
        "What was TCS net profit in 2024?",
        "What were TCS employee costs in 2025?"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag.process_query(query)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence_score']:.4f}")
        print(f"Time: {result['response_time']:.4f}s")