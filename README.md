# RAG Document Agent

Hệ thống RAG (Retrieval Augmented Generation) để đọc tài liệu và trả lời câu hỏi dựa trên nội dung tài liệu.

## 📋 Mục lục

- [Kiến trúc RAG](#kiến-trúc-rag)
- [Cài đặt](#cài-đặt)
- [Cấu hình](#cấu-hình)
- [Sử dụng](#sử-dụng)
- [Triển khai](#triển-khai)
- [Cấu trúc dự án](#cấu-trúc-dự-án)

## 🏗️ Kiến trúc RAG

Hệ thống RAG được xây dựng với 4 thành phần chính:

### 1. **DataLoader** (`src/DataLoader.py`)
- **Chức năng**: Tải và xử lý tài liệu từ nhiều định dạng
- **Hỗ trợ**: PDF, Markdown (.md), TXT, CSV, Excel (.xlsx), Word (.docx), JSON
- **Công nghệ**: LangChain Document Loaders

### 2. **Embedding** (`src/Embedding.py`)
- **Chức năng**: 
  - Chia tài liệu thành các chunks (mặc định: 1000 ký tự, overlap 200)
  - Tạo embeddings bằng SentenceTransformer
- **Model mặc định**: `all-MiniLM-L6-v2`
- **Công nghệ**: Sentence Transformers, LangChain Text Splitters

### 3. **VectorStore** (`src/VectorStore.py`)
- **Chức năng**: 
  - Lưu trữ embeddings trong vector database
  - Tìm kiếm semantic similarity
- **Công nghệ**: ChromaDB (persistent storage)
- **Quy trình**: 
  1. Split documents → chunks
  2. Generate embeddings
  3. Store in ChromaDB

### 4. **Retrieval** (`src/Retrieval.py`)
- **Chức năng**: 
  - Tìm kiếm tài liệu liên quan từ vector store
  - Tạo câu trả lời dựa trên context
- **LLM**: Groq (Llama 3.1 8B Instant)
- **Quy trình**:
  1. Query → embeddings
  2. Semantic search → top-k documents
  3. LLM generation với context

### Luồng hoạt động

```
Documents → DataLoader → Embedding (Split + Embed) → VectorStore (ChromaDB)
                                                              ↓
User Query → Retrieval → VectorStore (Search) → LLM (Generate Answer)
```

## 🔧 Cài đặt

### Yêu cầu hệ thống

- Python 3.8+
- pip hoặc conda

### Bước 1: Clone repository

Clone repository về máy và di chuyển vào thư mục dự án.

### Bước 2: Tạo môi trường ảo (khuyến nghị)

Tạo và kích hoạt môi trường ảo Python để quản lý dependencies riêng biệt. Trên Windows sử dụng `venv\Scripts\activate`, trên Linux/Mac sử dụng `source venv/bin/activate`.

### Bước 3: Cài đặt dependencies

Cài đặt tất cả các thư viện cần thiết từ file `requirements.txt`.

### Bước 4: Cấu hình môi trường

Tạo file `.env` trong thư mục gốc và thêm `GROQ_API_KEY` với giá trị API key của bạn. Lấy API key tại https://console.groq.com/

## ⚙️ Cấu hình

### Thay đổi embedding model

Bạn có thể thay đổi embedding model trong file `src/VectorStore.py` hoặc `src/Embedding.py` bằng cách thay đổi giá trị `embedding_model`. Model mặc định là `all-MiniLM-L6-v2`, bạn có thể sử dụng các model khác từ sentence-transformers.

### Thay đổi LLM model

Trong file `src/Retrieval.py`, bạn có thể thay đổi `llm_model` để sử dụng các model khác như `llama-3.1-70b-versatile` hoặc `mixtral-8x7b-32768` thay vì model mặc định `llama-3.1-8b-instant`.

### Thay đổi chunk size

Trong file `src/Embedding.py`, bạn có thể điều chỉnh `chunk_size` (kích thước chunk) và `chunk_overlap` (độ overlap giữa các chunks) để tối ưu hóa việc chia nhỏ tài liệu.

## 🚀 Sử dụng

### 1. Chuẩn bị dữ liệu

Đặt tất cả các tài liệu bạn muốn sử dụng vào thư mục `data/`. Hệ thống hỗ trợ nhiều định dạng như PDF, Markdown, TXT, CSV, Excel, Word, và JSON.

### 2. Xây dựng Vector Store

Chạy file `app.py` để tự động load tài liệu và xây dựng vector store. Hoặc bạn có thể sử dụng trực tiếp các module bằng cách import `VectorStore` và `load_all_documents`, sau đó gọi `build_from_documents` để tạo vector store từ các tài liệu đã load.

### 3. Truy vấn

Sau khi vector store đã được xây dựng, bạn có thể khởi tạo `RAG_Retrieval` với vector store và sử dụng phương thức `search_and_summarize` để đặt câu hỏi. Phương thức này sẽ tự động tìm kiếm các tài liệu liên quan và tạo câu trả lời dựa trên context.

### 4. Sử dụng VectorStore trực tiếp

Bạn cũng có thể query vector store trực tiếp bằng phương thức `query` để lấy về các documents, metadata và distances. Điều này hữu ích khi bạn muốn xử lý kết quả tìm kiếm theo cách riêng của mình.

## 📦 Triển khai

### Triển khai local

Để triển khai trên máy local, đầu tiên cài đặt tất cả dependencies như đã hướng dẫn ở phần Cài đặt. Sau đó đảm bảo thư mục `data/` chứa các tài liệu cần thiết. Chạy `app.py` để xây dựng vector store. Khi sử dụng trong code, bạn có thể load vector store đã tồn tại từ thư mục `vector_db` và khởi tạo `RAG_Retrieval` để bắt đầu truy vấn.

### Triển khai với Streamlit (Web UI)

Để tạo giao diện web, bạn có thể tạo file `streamlit_app.py` và sử dụng Streamlit để xây dựng UI. Cài đặt Streamlit, sau đó sử dụng `@st.cache_resource` để cache RAG system và tạo các widget như text input để nhận câu hỏi từ người dùng. Hiển thị kết quả trả về từ `search_and_summarize` trên giao diện.


## 📁 Cấu trúc dự án

```
RAG_Document_Agent/
├── app.py                 # Main application
├── requirements.txt       # Dependencies
├── .env                   # Environment variables (tạo mới)
├── README.md             # Documentation
├── data/                  # Thư mục chứa tài liệu
│   ├── *.pdf
│   ├── *.md
│   └── ...
├── vector_db/            # ChromaDB storage (tự động tạo)
└── src/
    ├── __init__.py
    ├── DataLoader.py     # Load documents
    ├── Embedding.py      # Split & Embed
    ├── VectorStore.py    # Vector database
    └── Retrieval.py      # RAG retrieval & generation
```


## 📚 Tài liệu tham khảo

- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [Groq API](https://console.groq.com/docs)

