import numpy as np
from load import load_docs
from embed import embed_docs, embed_query

docs = load_docs()
mat = embed_docs([d["text"] for d in docs]) # 모든 문서 본문을 벡터화한다

def search(q, k=5): # 질문 q와 가장 비슷한 문서 k개를 찾는 함수 (k 기본값은 5)
    qv = embed_query(q) # 질문을 벡터로 변환, 모양은 (384,)
    scores = mat @ qv   # 행렬곱(@): 문서 58개 각각과 질문의 내적 -> 유사도 58개. 모양은 (58,)
    top = np.argsort(-scores)[:k] # -scores로 부호를 뒤집어 오름차순 정렬 = 점수 높은 순, 앞에서 K개의 '인덱스'만 취함
    return [(docs[i]["title"], float(scores[i])) for i in top] # 각 인덱스에 해당하는 (제목, 점수) 튜플 리스트 반환. float()은 numpy 숫자를 일반 숫자로 반환

if __name__ == "__main__":  # 직접 실행했을 때만 동작하는 테스트 구간
    for q in ["탈퇴하면 데이터 바로 지워져?", "북마크 폴더 이름 최대 몇 글자야?", "다크모드 지원해?"]: # 테스트 질문 3개 간이
        print(q)
        for t, s in search(q):
            print(f"  {s:.3f}  {t}") 