from langchain_core.documents import Document # 랭체인의 문서 단위: 본문 + 메타데이터
from langchain_core.embeddings import Embeddings # 임베딩 클래스의 부모(인터페이스)
from langchain_huggingface import HuggingFaceEmbeddings # 로컬 sentence-transformers를 랭체인 규격으로 감싼 것
from langchain_qdrant import QdrantVectorStore # 랭체인용 Qdrant 연결 객체
from load import load_docs

class E5 (Embeddings):
    def __init__(self):
        self.base = HuggingFaceEmbeddings(
                        model_name="intfloat/multilingual-e5-small", # 우리가 쓰던 같은 모델
            encode_kwargs={"normalize_embeddings": True}, # 벡터 길이를 1로 정규화
        )

    def embed_documents(self, texts): # 랭체인이 문서를 벡터화할 때 호출하는 메서드
        return self.base.embed_documents([f"passage: {t}" for t in texts]) # e5 규칙: 문서 앞에 passage: 를 붙인다

    def embed_query(self, text): # 랭체인이 질문을 벡터화할 때 호출하는 메서드
        return self.base.embed_query(f"query: {text}") # e5 규칙: 질문 앞에 query: 를 붙인다

docs = [
    Document(page_content=d["text"], metadata={k: v for k, v in d.items() if k != "text"}) # text는 본문, 나머지는 메타데이터
    for d in load_docs()
]

vs = QdrantVectorStore.from_documents( # 임베딩 + 컬렉션 생성 + 저장을 한 번에
    docs, E5(),
    url="http://localhost:6333",
    collection_name="policy_lc", # 기존 policy와 분리
    force_recreate=True, # 다시 실행해도 같은 상태가 되게 지우고 새로 만든다
)

def search(q, k=5): # eval_run.py의 evaluate가 기대하는 (제목, 점수) 형태로 맞추는 어댑터
    return [(d.metadata["title"], s) for d, s in vs.similarity_search_with_score(q, k=k)]

if __name__ == "__main__":
    for q in ["탈퇴하면 데이터 바로 지워져?", "북마크 폴더 이름 최대 몇 글자야?"]:
        print(q)
        for t, s in search(q, 3):
            print(f"  {s:.3f}  {t}")

    retriever = vs.as_retriever(search_kwargs={"k": 3}) # 랭체인 표준 검색 부품
    print([d.metadata["title"] for d in retriever.invoke("북마크 폴더 이름 최대 몇 글자야?")])
