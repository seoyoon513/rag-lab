from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from load import load_docs
from embed import embed_docs

client = QdrantClient(url="http://localhost:6333") # 도커로 띄운 쿼드란트에 접속
COL = "policy" # 컬렉션 이름

def build():
    docs = load_docs()
    vecs = embed_docs([d["text"] for d in docs]) # (58, 384)

    if client.collection_exists(COL): # 다시 실행해도 같은 결과가 나오게 기존 컬렉션 치움
        client.delete_collection(COL)

    client.create_collection(
        collection_name=COL,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE), # 차원 384, 거리는 코사인
    )

    points = [
        PointStruct(
            id=i,
            vector=vecs[i].tolist(),
            payload=docs[i],  # id, title, area, status, text 전부 payload로 저장
            )
            for i in range(len(docs))
    ]

    client.upsert(collection_name=COL, points=points) # 맞으면 덮어쓰고 없으면 삽입
    print("count:", client.count(COL).count)

if __name__ == "__main__":
        build()
