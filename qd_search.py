from qdrant_client import QdrantClient
from embed import embed_query
from qd_index import COL
from qdrant_client.models import Filter, FieldCondition, MatchValue # STEP2: 필터 조건을 만드는 도구

client = QdrantClient(url="http://localhost:6333") # 도커로 띄운 쿼드란트에 접속

def search(q, k=5):
    res = client.query_points(
        collection_name=COL,
        query=embed_query(q).tolist(), # 질문 벡터 (리스트로 변환)
        limit=k, # 상위 k개
        with_payload=True, # 결과에 payload도 같이 받는다
    )
    return [(p.payload["title"], p.score) for p in res.points] # (제목, 점수)

def search_filtered(q, area=None, status=None, k=5):
    conds = [] # 조건 목록
    if area:
        conds.append(FieldCondition(key="area", match=MatchValue(value=area))) # area가 일치
    if status:
        conds.append(FieldCondition(key="status", match=MatchValue(value=status))) # status가 일치
    res = client.query_points(
        collection_name=COL,
        query=embed_query(q).tolist(),
        query_filter=Filter(must=conds) if conds else None, # must = 모든 조건을 만족해야 함
        limit=k,
        with_payload=True,
    )
    return [(p.payload["title"], p.payload["status"], p.score) for p in res.points]

if __name__ == "__main__":
    for q in ["탈퇴하면 데이터 바로 지워져?", "북마크 폴더 이름 최대 몇 글자야?", "다크모드 지원해?"]:
        print(q)
        for t, s in search(q):
            print(f"  {s:.3f}  {t}")