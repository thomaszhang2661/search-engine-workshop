import numpy as np
import torch
import faiss
import os
import time
import json
import hashlib
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Try to import modern model libraries
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

class FAQVectorizationStrategy(Enum):
    """FAQ vectorization strategy"""
    QUESTION_ONLY = "question_only"    # Question-only vectorization
    HYBRID = "hybrid"                  # Hybrid strategy (default)
    TEXT_ONLY = "text_only"           # Text content only
    QUESTION_TEXT = "question_text"   # Question + text combined

@dataclass
class FAQItem:
    """FAQ item data structure compatible with LLM RAG workshop format"""
    question: str
    text: str
    id: Optional[str] = None
    course: Optional[str] = None
    section: Optional[str] = None
    keywords: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.id is None:
            # Auto-generate ID
            content = f"{self.question}_{self.text}"
            self.id = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
        
        if self.keywords is None:
            self.keywords = []
        
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def answer(self) -> str:
        """Alias for text field to maintain backward compatibility"""
        return self.text
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FAQItem':
        """Create FAQ item from dictionary"""
        # Handle both 'text' and 'answer' fields
        if 'answer' in data and 'text' not in data:
            data['text'] = data['answer']
        return cls(**data)
    
    @classmethod
    def from_rag_workshop_format(cls, data: Dict[str, Any]) -> 'FAQItem':
        """Create FAQ item from RAG workshop document format"""
        return cls(
            question=data.get('question', ''),
            text=data.get('text', ''),
            course=data.get('course', ''),
            section=data.get('section', ''),
            id=data.get('id'),
            metadata=data
        )

@dataclass
class FAQConfig:
    """Enhanced FAQ system configuration"""
    # Model configuration
    model_name: str = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'
    device: str = 'auto'  # 'auto', 'cuda', 'cpu'
    
    # Vectorization strategy configuration
    vectorization_strategy: FAQVectorizationStrategy = FAQVectorizationStrategy.HYBRID
    question_weight: float = 0.7  # Question weight in hybrid strategy
    text_weight: float = 0.3      # Text weight in hybrid strategy
    
    # Index configuration
    index_type: str = 'auto'  # 'auto', 'flat', 'ivf'
    normalize_embeddings: bool = True
    
    # Cache configuration
    cache_dir: Optional[str] = './faq_cache'
    enable_cache: bool = True
    force_rebuild: bool = False
    
    # Search configuration
    default_top_k: int = 5
    default_threshold: float = 0.0  # 关键修复：降低默认阈值
    course_filter: Optional[str] = None
    
    # Batch processing configuration
    batch_size: int = 32
    show_progress: bool = True
    
    def __post_init__(self):
        """Configuration validation and auto-correction"""
        # Weight normalization for hybrid strategy
        if self.vectorization_strategy == FAQVectorizationStrategy.HYBRID:
            total_weight = self.question_weight + self.text_weight
            if abs(total_weight - 1.0) > 1e-6:
                print(f"⚠️  Weights don't sum to 1.0, auto-normalizing: {total_weight} -> 1.0")
                self.question_weight /= total_weight
                self.text_weight /= total_weight
        
        # Device auto-detection
        if self.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to_serializable_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary"""
        config_dict = asdict(self)
        config_dict['vectorization_strategy'] = self.vectorization_strategy.value
        return config_dict
    
    @classmethod
    def from_serializable_dict(cls, data: Dict[str, Any]) -> 'FAQConfig':
        """Create configuration from serializable dictionary"""
        if isinstance(data.get('vectorization_strategy'), str):
            data['vectorization_strategy'] = FAQVectorizationStrategy(data['vectorization_strategy'])
        return cls(**data)

class EnhancedFAQSystem:
    """Enhanced FAQ system with bug fixes"""
    
    def __init__(self, config: Optional[FAQConfig] = None):
        self.config = config if config else FAQConfig()
        
        print(f"🚀 Initializing enhanced FAQ system")
        print(f"🤖 Model: {self.config.model_name}")
        print(f"📝 Strategy: {self.config.vectorization_strategy.value}")
        print(f"💻 Device: {self.config.device}")
        
        # Initialize model
        self._load_model()
        
        # Initialize system variables
        self.faq_items: List[FAQItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.is_trained = False
        
        # Performance statistics
        self.stats = {
            'build_time': 0,
            'embedding_time': 0,
            'index_time': 0,
            'search_count': 0,
            'total_search_time': 0,
            'cache_hits': 0
        }
    
    def _load_model(self) -> None:
        """Load embedding model with better error handling"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                print("📦 Loading Sentence-Transformers model...")
                self.model = SentenceTransformer(self.config.model_name, device=self.config.device)
                self.model_type = 'sentence_transformer'
                self.embedding_dim = self.model.get_sentence_embedding_dimension()
                print(f"✅ Model loaded successfully, embedding dimension: {self.embedding_dim}")
            else:
                raise RuntimeError("❌ sentence-transformers not available")
                
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("🔄 Trying fallback models...")
            self._load_fallback_model()
    
    def _load_fallback_model(self) -> None:
        """Load fallback model"""
        fallback_models = [
            'sentence-transformers/all-MiniLM-L6-v2',
            'sentence-transformers/paraphrase-MiniLM-L6-v2'
        ]
        
        for fallback in fallback_models:
            try:
                print(f"🔄 Trying fallback model: {fallback}")
                if SENTENCE_TRANSFORMERS_AVAILABLE:
                    self.model = SentenceTransformer(fallback, device=self.config.device)
                    self.model_type = 'sentence_transformer'
                    self.embedding_dim = self.model.get_sentence_embedding_dimension()
                    self.config.model_name = fallback
                    print(f"✅ Fallback model loaded successfully")
                    break
                    
            except Exception as e:
                print(f"❌ Fallback model failed: {e}")
                continue
        else:
            raise RuntimeError("❌ All models failed to load")
    
    def load_from_rag_workshop_url(self, docs_url: str = None) -> None:
        """Load documents from RAG workshop URL"""
        if docs_url is None:
            docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
        
        print(f"📥 Loading documents from RAG workshop URL...")
        try:
            response = requests.get(docs_url, timeout=30)
            response.raise_for_status()
            documents_raw = response.json()
            
            faq_items = []
            for course in documents_raw:
                course_name = course['course']
                for doc in course['documents']:
                    doc['course'] = course_name
                    faq_items.append(FAQItem.from_rag_workshop_format(doc))
            
            self.faq_items = faq_items
            print(f"✅ Successfully loaded {len(faq_items)} FAQ items from {len(documents_raw)} courses")
            
            # Show course statistics
            courses = {}
            for item in faq_items:
                courses[item.course] = courses.get(item.course, 0) + 1
            
            print("📊 Course distribution:")
            for course, count in sorted(courses.items()):
                print(f"   - {course}: {count} items")
            
        except Exception as e:
            print(f"❌ Failed to load from URL: {e}")
            raise
    
    def load_faq_items(self, faq_data: Union[List[Dict], List[FAQItem], str]) -> None:
        """Load FAQ data with enhanced format support"""
        print("📚 Loading FAQ data...")
        
        if isinstance(faq_data, str):
            if faq_data.startswith('http'):
                self.load_from_rag_workshop_url(faq_data)
                return
            else:
                with open(faq_data, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list) and len(data) > 0 and 'course' in data[0]:
                    faq_items = []
                    for course in data:
                        course_name = course['course']
                        for doc in course['documents']:
                            doc['course'] = course_name
                            faq_items.append(FAQItem.from_rag_workshop_format(doc))
                    self.faq_items = faq_items
                else:
                    self.faq_items = [FAQItem.from_dict(item) for item in data]
            
        elif isinstance(faq_data, list):
            if len(faq_data) > 0:
                if isinstance(faq_data[0], dict):
                    self.faq_items = [FAQItem.from_dict(item) for item in faq_data]
                elif isinstance(faq_data[0], FAQItem):
                    self.faq_items = faq_data
                else:
                    raise ValueError("Unsupported data format")
            else:
                self.faq_items = []
        else:
            raise ValueError("faq_data must be a list, file path, or URL")
        
        print(f"✅ Successfully loaded {len(self.faq_items)} FAQ items")
        self._validate_faq_data()
    
    def _validate_faq_data(self) -> None:
        """Validate FAQ data"""
        if not self.faq_items:
            print("⚠️  FAQ data is empty")
            return
        
        # Check required fields
        missing_questions = sum(1 for item in self.faq_items if not item.question.strip())
        missing_texts = sum(1 for item in self.faq_items if not item.text.strip())
        
        if missing_questions > 0:
            print(f"⚠️  Found {missing_questions} empty questions")
        
        if missing_texts > 0:
            print(f"⚠️  Found {missing_texts} empty texts")
    
    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode text list with better error handling"""
        if not texts:
            raise ValueError("❌ No texts to encode")
            
        # 清理空文本
        clean_texts = [text.strip() if text else "empty" for text in texts]
        
        try:
            if self.model_type == 'sentence_transformer':
                embeddings = self.model.encode(
                    clean_texts,
                    batch_size=self.config.batch_size,
                    show_progress_bar=self.config.show_progress,
                    convert_to_numpy=True,
                    normalize_embeddings=self.config.normalize_embeddings
                )
            else:
                embeddings = self._encode_with_transformers(clean_texts)
                if self.config.normalize_embeddings:
                    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            print(f"   ✅ Encoded {len(clean_texts)} texts to shape {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            print(f"❌ Encoding failed: {e}")
            raise
    
    def build_embeddings(self) -> np.ndarray:
        """Build FAQ embeddings with enhanced strategies"""
        if not self.faq_items:
            raise ValueError("❌ Please load FAQ data first")
        
        print(f"🔧 Building FAQ embeddings (strategy: {self.config.vectorization_strategy.value})...")
        embed_start = time.time()
        
        if self.config.vectorization_strategy == FAQVectorizationStrategy.QUESTION_ONLY:
            print("📝 Vectorization strategy: question-only")
            texts = [item.question for item in self.faq_items]
            self.embeddings = self._encode_texts(texts)
            
        elif self.config.vectorization_strategy == FAQVectorizationStrategy.TEXT_ONLY:
            print("📄 Vectorization strategy: text-only")
            texts = [item.text for item in self.faq_items]
            self.embeddings = self._encode_texts(texts)
            
        elif self.config.vectorization_strategy == FAQVectorizationStrategy.QUESTION_TEXT:
            print("🔗 Vectorization strategy: question+text combined")
            texts = [f"Question: {item.question} Answer: {item.text}" for item in self.faq_items]
            self.embeddings = self._encode_texts(texts)
            
        elif self.config.vectorization_strategy == FAQVectorizationStrategy.HYBRID:
            print(f"⚖️  Vectorization strategy: hybrid (question weight:{self.config.question_weight:.2f})")
            
            questions = [item.question for item in self.faq_items]
            texts = [item.text for item in self.faq_items]
            
            print("   📝 Encoding questions...")
            question_embeddings = self._encode_texts(questions)
            
            print("   📄 Encoding texts...")
            text_embeddings = self._encode_texts(texts)
            
            print("   ⚖️  Weighted mixing...")
            self.embeddings = (self.config.question_weight * question_embeddings + 
                             self.config.text_weight * text_embeddings)
            
            if self.config.normalize_embeddings:
                self.embeddings = self.embeddings / np.linalg.norm(
                    self.embeddings, axis=1, keepdims=True
                )
        
        embed_time = time.time() - embed_start
        self.stats['embedding_time'] = embed_time
        
        print(f"✅ Embedding construction completed!")
        print(f"   - Shape: {self.embeddings.shape}")
        print(f"   - Time: {embed_time:.2f}s")
        
        return self.embeddings
    
    def build_index(self) -> None:
        """Build FAISS index with cosine similarity"""
        if self.embeddings is None:
            raise ValueError("❌ Please build embeddings first")
        
        print("🔧 Building FAISS index...")
        index_start = time.time()
        
        n_items, dim = self.embeddings.shape
        
        # 关键修复：使用内积索引用于余弦相似度
        print("   📋 Using IndexFlatIP for cosine similarity")
        self.index = faiss.IndexFlatIP(dim)
        
        # 确保向量已标准化
        if not self.config.normalize_embeddings:
            print("   🔧 Normalizing embeddings for cosine similarity...")
            embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        else:
            embeddings = self.embeddings
        
        # Add vectors
        self.index.add(embeddings.astype('float32'))
        self.is_trained = True
        
        index_time = time.time() - index_start
        self.stats['index_time'] = index_time
        self.stats['build_time'] = self.stats['embedding_time'] + index_time
        
        print(f"✅ Index construction completed!")
        print(f"   - Contains vectors: {self.index.ntotal}")
        print(f"   - Build time: {index_time:.2f}s")
    
    def build_system(self) -> None:
        """One-click build entire system"""
        print("🚀 Starting enhanced FAQ system construction...")
        total_start = time.time()
        
        self.build_embeddings()
        self.build_index()
        
        total_time = time.time() - total_start
        print(f"🎉 Enhanced FAQ system construction completed! Total time: {total_time:.2f}s")
    
    def search(self, query: str, top_k: Optional[int] = None, 
              threshold: Optional[float] = None, course_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Enhanced search with better debugging"""
        if not self.is_trained:
            raise ValueError("❌ Please build system first")
        
        top_k = top_k if top_k is not None else self.config.default_top_k
        threshold = threshold if threshold is not None else self.config.default_threshold
        course_filter = course_filter if course_filter is not None else self.config.course_filter
        
        search_start = time.time()
        
        # Encode query
        try:
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            print(f"🔍 Query encoded: '{query}' -> shape {query_embedding.shape}")
        except Exception as e:
            print(f"❌ Query encoding failed: {e}")
            return []
        
        # FAISS search - 关键修复：搜索更多候选项用于过滤
        search_k = min(max(top_k * 5, 50), self.index.ntotal)  # 搜索更多候选项
        try:
            scores, indices = self.index.search(
                query_embedding.astype('float32'),
                search_k
            )
            print(f"🔍 FAISS found {len(scores[0])} candidates")
        except Exception as e:
            print(f"❌ FAISS search failed: {e}")
            return []
        
        # Format and filter results
        results = []
        valid_results = 0
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            if idx >= len(self.faq_items):  # Index out of range
                print(f"⚠️  Index {idx} out of range (max: {len(self.faq_items)-1})")
                continue
            
            valid_results += 1
            
            # 关键修复：降低阈值或移除阈值检查用于调试
            if score <= threshold:
                print(f"🔍 Result {valid_results}: score={score:.3f} <= threshold={threshold}, skipping")
                continue
            
            faq_item = self.faq_items[idx]
            
            # Apply course filter if specified
            if course_filter and faq_item.course != course_filter:
                print(f"🔍 Result {valid_results}: course '{faq_item.course}' != filter '{course_filter}', skipping")
                continue
            
            result = {
                'faq_item': faq_item,
                'similarity': float(score),
                'index': int(idx),
                'strategy': self.config.vectorization_strategy.value,
                'course': faq_item.course,
                'section': faq_item.section
            }
            
            results.append(result)
            print(f"✅ Result {len(results)}: score={score:.3f}, course='{faq_item.course}', question='{faq_item.question[:50]}...'")
            
            if len(results) >= top_k:
                break
        
        print(f"🔍 Search completed: {len(results)} results from {valid_results} valid candidates")
        
        # Update statistics
        search_time = time.time() - search_start
        self.stats['search_count'] += 1
        self.stats['total_search_time'] += search_time
        
        return results
    
    def ask(self, question: str, return_detailed: bool = False, 
           course_filter: Optional[str] = None) -> Union[str, Dict[str, Any]]:
        """Enhanced Q&A interface"""
        results = self.search(question, top_k=1, course_filter=course_filter)
        
        if not results:
            answer = "Sorry, I didn't find relevant answers. Please try rephrasing your question."
            if course_filter:
                answer += f" (searched in course: {course_filter})"
            
            if return_detailed:
                return {
                    'answer': answer,
                    'confidence': 0.0,
                    'source': None,
                    'course': course_filter
                }
            return answer
        
        best_result = results[0]
        faq_item = best_result['faq_item']
        confidence = best_result['similarity']
        
        if return_detailed:
            return {
                'answer': faq_item.answer,
                'confidence': confidence,
                'source': {
                    'question': faq_item.question,
                    'course': faq_item.course,
                    'section': faq_item.section,
                    'id': faq_item.id
                },
                'strategy': best_result['strategy']
            }
        
        return faq_item.answer
    
    def get_courses(self) -> List[str]:
        """Get list of available courses"""
        courses = list(set(item.course for item in self.faq_items if item.course))
        return sorted(courses)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        courses = self.get_courses()
        course_stats = {}
        for course in courses:
            course_stats[course] = sum(1 for item in self.faq_items if item.course == course)
        
        return {
            'system_info': {
                'model_name': self.config.model_name,
                'strategy': self.config.vectorization_strategy.value,
                'device': self.config.device,
                'n_faq_items': len(self.faq_items),
                'embedding_dim': self.embedding_dim,
                'is_trained': self.is_trained,
                'n_courses': len(courses),
                'courses': courses
            },
            'course_distribution': course_stats,
            'performance_stats': self.stats.copy()
        }


def create_demo_faq_data() -> List[FAQItem]:
    """Create demo FAQ data"""
    return [
        FAQItem(
            question="How do I install Python?",
            text="You can download Python from the official website at python.org. Choose the version for your operating system and follow the installation instructions. Make sure to add Python to your PATH during installation.",
            course="programming-basics",
            section="installation"
        ),
        FAQItem(
            question="What is machine learning?",
            text="Machine learning is a subset of artificial intelligence (AI) that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data.",
            course="machine-learning-course", 
            section="introduction"
        ),
        FAQItem(
            question="How do I join the course?",
            text="To join the course, visit our website and click on the 'Enroll' button. You'll need to create an account and complete the registration process.",
            course="general-info",
            section="enrollment"
        ),
        FAQItem(
            question="Programming environment setup",
            text="Set up your development environment by installing Python, a code editor like VS Code, and essential packages. Use virtual environments to manage dependencies.",
            course="programming-basics",
            section="environment-setup"
        )
    ]


def demo_fixed_system():
    """演示修复后的系统"""
    print("🎯 修复后的FAQ系统演示")
    print("=" * 50)
    
    # 创建配置
    config = FAQConfig(
        vectorization_strategy=FAQVectorizationStrategy.HYBRID,
        model_name='sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
        default_threshold=0.0,  # 设置为0以显示所有结果
        show_progress=False
    )
    
    try:
        # 创建系统
        faq_system = EnhancedFAQSystem(config)
        
        # 加载数据
        print("\n📚 Loading demo data...")
        faq_system.load_faq_items(create_demo_faq_data())
        
        # 构建系统
        faq_system.build_system()
        
        # 测试查询
        test_queries = [
            "How to install Python?",
            "What is machine learning?", 
            "How do I join the course?",
            "Programming environment setup"
        ]
        
        print(f"\n🔍 Testing queries:")
        print("-" * 40)
        
        for query in test_queries:
            print(f"\n❓ Query: {query}")
            results = faq_system.search(query, top_k=2)
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"  {i}. Score: {result['similarity']:.3f}")
                    print(f"     Course: {result['course']}")
                    print(f"     Q: {result['faq_item'].question}")
                    print(f"     A: {result['faq_item'].answer}")
            else:
                print("  ❌ No results found")
        
        # 显示统计信息
        stats = faq_system.get_stats()
        print(f"\n📊 System Stats:")
        print(f"   - Items: {stats['system_info']['n_faq_items']}")
        print(f"   - Courses: {stats['system_info']['n_courses']}")
        print(f"   - Model: {stats['system_info']['model_name']}")
        
        print("\n✅ 系统修复成功！现在可以正常搜索了。")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    demo_fixed_system()