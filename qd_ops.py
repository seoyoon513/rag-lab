from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList
from qd_index import COL

client = QdrantClient(url="http://localhost:6333")

print("전체 개수:", client.count(COL).count) # 58

pts = client.retrieve(COL, ids=[0,1], with_payload=True)
for p in pts:
    print(p.id, p.payload["title"])

page, nxt = client.scroll(COL, limit=3, with_payload=["title"]) # 처음부터 3개 훑기 (nxt는 다음 페이지 시작 위치)
print([p.payload["title"] for p in page])

client.delete(COL, points_selector=PointIdsList(points=[0]))
print("삭제 후:", client.count(COL).count) #57

client.set_payload(COL, payload={"memo": "수정 테스트"}, points=[1]) # 일부 필드만 수정
print(client.retrieve(COL, ids=[1])[0].payload.get("memo")) # 수정 테스트