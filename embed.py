import numpy as np

from sentence_transformers import SentenceTransformer # 문장을 벡터로 바꿔주는 로더

model = SentenceTransformer("intfloat/multilingual-e5-small") # 로컬 임베딩 모델 로드

def embed_docs(texts):
    return model.encode(
        [f"passage: {t}" for t in texts],   # e5 모델 규칙: 문서 앞에 "passage: "를 붙임
        normalize_embeddings=True,          # 벡터 길이를 1로 정규화 → 내적 = 코사인 유사도
    )                                       # 반환: (문서 수, 384) 모양의 배열

def embed_query(q):
    return model.encode(
        f"query: {q}",                      # e5 모델 규칙: 질문 앞에 "query: "를 붙임
        normalize_embeddings=True,          # 문서와 같은 방식으로 정규화 한다
    )                                       # 반환: (문서 수, 384) 모양의 배열
