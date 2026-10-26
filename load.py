import json
from pathlib import Path

SRC = Path.home() / "policy-bot/data/decisions" # 결정 기록 폴더
AREAS = ["북마크", "행사 등록", "계정·프로필"] # 사용 영역 3개

def load_docs(): # 결정 json을 읽고 문서 리스트로 돌려준다
    docs = []
    for area in AREAS: # 폴더 순회
        for f in sorted((SRC / area).glob("*.json")):
            d = json.loads(f.read_text())
            text = f"{d['topic']}\n상태: {d['status']}\n{d['decision']}" # 임베딩할 본문을 조립
            if d.get("rationale"):
                text += f"\n근거: {d['rationale']}" # 근거가 있을 때만 본문에 근거를 붙인다
            docs.append({ # 문서 하나를 dict로 만들어 리스트에 추가
                "id": d["id"], # 결정 고유 ID
                "title": d["topic"], # 제목. 골드셋 정답 판정에 쓰임
                "area": d["area"], # 영역. 나중에 Qdrant 필터에 쓸 예정
                "status": d["status"], # 최종결정/제안 등. 나중에 필터나 trap 문항 분석에 쓸 예정
                "text": text, # 임베딩할 본문
            })
    return docs

if __name__ == "__main__":
    docs = load_docs()
    print(len(docs))
    print(docs[0])
