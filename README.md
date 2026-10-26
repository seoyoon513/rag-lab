# rag-lab

RAG를 프레임워크 없이 직접 구현해보고(numpy → Qdrant), 같은 골드셋으로 측정한 뒤, 랭체인으로 다시 만들어 비교한 실습 기록.
[Mastering RAG (인프런)](https://www.inflearn.com/course/mastering-rag-as-tau?cid=342277) 커리큘럼의 개념 순서에 맞춰 정리했다.

- 코퍼스: 팀 내 정책 결정 기록 58건 (영역 3개). **공개 저장소라 데이터와 골드셋은 포함하지 않는다.**
- 임베딩: 로컬 `intfloat/multilingual-e5-small` (384차원, 다국어)
- 벡터 DB: Qdrant (Docker)
- 강의는 ChromaDB + Ollama 기반이고, 이 실습은 **Qdrant + sentence-transformers**로 같은 개념을 다룬다. 도구는 다르지만 개념은 1:1로 대응된다.

## 커리큘럼 매핑

| 강의 섹션 | 이 저장소에서 한 것 | 상태 |
|---|---|---|
| 2. 임베딩과 벡터 DB의 원리 | 임베딩, 정규화, 코사인 유사도(내적)를 numpy로 직접 계산 (`embed.py`, `search.py`) | 완료 |
| 3. 개발 환경 (Ollama 임베딩) | Ollama 대신 로컬 sentence-transformers 모델 사용 | 대체 |
| 4. 벡터 DB 기본기 (컬렉션, CRUD) | Qdrant 컬렉션 생성, upsert, retrieve, scroll, delete, set_payload (`qd_index.py`, `qd_ops.py`) | 완료 |
| 5. 유사도 검색과 첫 검색기 | numpy 검색과 Qdrant `query_points` 결과가 일치함을 확인 (`qd_search.py`) | 완료 |
| 6. 메타데이터 필터와 하이브리드 검색 | payload 필터(영역, 상태) 실습 (`qd_filter_test.py`). 하이브리드(키워드+벡터)는 미실습 | 일부 |
| 7. 운영과 RAG 실전 (백엔드, LLM 결합) | 랭체인으로 같은 검색을 재구현 (`langchain_lab/`). LLM 호출은 하지 않음 | 일부 |
| 8. 고급 청킹, HNSW 튜닝 | 1결정 = 1청크로 구성. HNSW 파라미터는 대시보드에서 확인만 함 | 일부 |
| 9. 재랭킹, Conversational RAG | 미구현. 평가 결과로 재랭킹이 필요한 근거만 확인 | 예정 |

평가 파이프라인(`eval_run.py`)은 강의 범위 밖이지만, 각 단계의 효과를 숫자로 확인하려고 만들었다.

## 개념 정리

### 1. 임베딩과 유사도 (섹션 2)
- 문장을 고정 길이 벡터(여기서는 384차원)로 바꾸면 의미가 비슷한 문장이 가까운 벡터가 된다.
- 벡터 길이를 1로 **정규화**하면 내적이 곧 코사인 유사도다. 그래서 `문서행렬 @ 질문벡터` 한 줄로 모든 문서의 유사도가 나온다.
- e5 계열 모델은 **문서에 `passage: `, 질문에 `query: ` 접두어**를 붙여야 성능이 나온다. 빠뜨려도 에러 없이 품질만 떨어져서 찾기 어렵다.
- e5의 점수는 0.8~0.9에 몰린다. 점수의 절대값보다 **순위와 격차**를 봐야 한다.

```python
mat = embed_docs(texts)            # (문서 수, 384)
scores = mat @ embed_query(q)      # 모든 문서와의 코사인 유사도
top = np.argsort(-scores)[:k]      # 점수 높은 순 인덱스
```

### 2. 벡터 DB 기본기 (섹션 4, 5)
| 개념 | 설명 |
|---|---|
| collection | 테이블에 해당. 벡터 차원과 거리 방식이 생성 시 고정된다. |
| point | 한 행. `id` + `vector` + `payload` |
| payload | 벡터에 붙는 메타데이터(JSON). 필터에 쓴다. |

- point `id`는 정수나 UUID만 가능하다. 원본 ID는 payload에 넣었다.
- `upsert`는 있으면 덮어쓰고 없으면 삽입한다. 컬렉션을 지우고 다시 만들면 몇 번을 실행해도 같은 상태가 된다.
- 기본 명령: `count`, `retrieve`(id 조회), `scroll`(훑기), `delete`, `set_payload`(일부 필드 수정).
- 설치된 클라이언트(1.20)에는 `search()`가 없고 **`query_points()`**를 쓴다.
- 문서가 적을 때는 직접 만든 numpy 검색과 Qdrant 결과가 소수점까지 같다. Qdrant가 해주는 일은 같은 계산을 인덱스와 함께 대신 하는 것이다.

### 3. 메타데이터 필터 (섹션 6)
- `Filter(must=[...])`는 AND, `should`는 OR, `must_not`은 제외다.
- 필터는 점수 임계값이 아니라 **조건에 맞는 것만 후보로 삼는다**. 조건에 맞는 문서가 적으면 k개보다 적게 나온다.
- 상태(`제안`/`최종결정`) 필터를 걸면 확정되지 않은 결정이 후보에서 빠진다. 다만 "확정됐지?"라고 묻는 질문에는 필터로 숨기는 것보다 **답변에 상태를 같이 알려주는 것**이 맞는 대응이다.

### 4. 청킹과 인덱스 (섹션 8)
- 문서가 이미 작은 단위(결정 1건)라서 **1결정 = 1청크**로 했다. 제목을 본문에 포함하면 핵심 단어가 임베딩에 반영된다.
- Qdrant는 벡터 수가 `indexing_threshold`(기본 10000)를 넘기 전에는 HNSW 인덱스를 만들지 않고 전수 비교한다 (`indexed_vectors_count: 0`). 그래서 이 규모에서는 근사 오차가 없다.
- HNSW 파라미터: `m`(노드당 연결 수, 16), `ef_construct`(인덱스 구축 시 탐색 폭, 100).

### 5. 랭체인과 직접 구현의 차이 (섹션 7)
| 항목 | 직접 구현 | 랭체인 |
|---|---|---|
| e5 접두어 | 코드에 직접 작성 | 자동으로 붙지 않는다. 어댑터(`Embeddings` 상속)를 직접 만들어야 한다. |
| 컬렉션 생성 | `create_collection` + 차원 지정 | `from_documents`가 임베딩, 생성, 저장을 한 번에 한다. |
| payload 구조 | 평평한 dict | `page_content` + `metadata`로 중첩된다. |
| 벡터 이름 | 기본(이름 없음) | 이름 있는 벡터(`''`)로 저장되어, 직접 만든 컬렉션과 호환되지 않는다. |
| 검색 결과 | `(제목, 점수)` | `(Document, 점수)`. 평가 코드에 맞추려면 어댑터가 필요하다. |
| 코드량 | 파일 3개 | 파일 1개 |
| 장점 | 무슨 일이 일어나는지 다 보인다. | 벡터 DB, 임베딩, LLM 같은 부품을 바꾸기 쉽다. |
| 단점 | 부품 교체 시 코드를 다시 짜야 한다. | 추상화 뒤에서 일어나는 일(접두어 등)을 놓치기 쉽다. |

## 평가

골드셋 36문항(정답 있음 28, 정답 없음 8)으로 검색만 측정했다. 정답 판정은 `expected_titles` 문자열이 검색된 제목에 포함되는지로 한다.

| 지표 | 뜻 |
|---|---|
| Hit@k | 정답이 상위 k개 안에 있나 |
| Top1 | 정답이 1등인가 |
| MRR | 정답 순위의 역수 평균 (1등 1.0, 2등 0.5) |
| 거절 정확도 | 문서에 없는 질문에 결과 0건을 돌려줬나 |

### 구현별 비교 (k=5, 하한선 없음)
| | Hit@5 | Top1 | MRR | 거절 |
|---|---|---|---|---|
| numpy | 1.0 | 0.893 | 0.94 | 0.0 |
| Qdrant | 1.0 | 0.893 | 0.94 | 0.0 |
| 랭체인 | 1.0 | 0.893 | 0.94 | 0.0 |

세 구현이 같은 모델과 같은 계산을 하므로 결과가 같다. 한 곳이라도 다르면 접두어나 정규화가 빠진 것이다.

### 점수 하한선 sweep (Qdrant)
| min_score | Hit@k | Top1 | 거절 | 전체 정확도 |
|---|---|---|---|---|
| 0.0 | 1.0 | 0.893 | 0.0 | 0.778 |
| 0.82 | 1.0 | 0.893 | 0.125 | 0.806 |
| 0.84 | 0.964 | 0.893 | 0.625 | 0.889 |
| **0.86** | 0.929 | 0.893 | **1.0** | **0.944** |
| 0.88 | 0.714 | 0.714 | 1.0 | 0.778 |
| 0.90 | 0.357 | 0.357 | 1.0 | 0.5 |

- 하한선을 올리면 거절은 좋아지지만 정답도 같이 잘린다. 정답 점수대와 거절 점수대가 겹쳐서 하나의 임계값으로 완전히 나눌 수 없다.
- 하한선은 후보를 버리는 도구라서 **순서는 바꾸지 못한다.** Top1은 0.893에서 움직이지 않는다.
- 36문항에 맞춘 값이라 과적합 위험이 있다. 문항을 늘려 다시 확인해야 한다.

### 관찰한 한계 (섹션 9의 동기)
- "정답은 찾는데 1등이 아닌" 경우: 비슷한 결정끼리 점수 차가 0.003 안팎이라 임베딩만으로 순서를 가리기 어렵다.
- "문서에 없는 질문"에도 0.82 안팎의 그럴듯한 결과가 나온다.
- 문서에 없는 표현으로 물으면(구어체, 줄임말) 의미가 먼 문서가 1등이 된다.

이 세 가지가 재랭킹(cross-encoder, 질문과 문서를 함께 읽고 채점)이 필요한 이유다.

## 폴더 구성

```
rag-lab/
├── load.py              결정 JSON → 문서 리스트 (1결정 = 1청크)
├── embed.py             e5 임베딩 (접두어, 정규화)
├── search.py            numpy 코사인 검색
├── qd_index.py          Qdrant 컬렉션 생성, upsert
├── qd_search.py         Qdrant 검색, payload 필터 검색
├── qd_filter_test.py    필터 있음/없음 비교
├── qd_ops.py            count, retrieve, scroll, delete, set_payload
├── evaluation.py        지표 계산 함수 (Hit@k, Top1, MRR, 거절)
├── eval_run.py          골드셋 평가, 하한선 sweep
├── golden.example.jsonl 골드셋 형식 예시
└── langchain_lab/
    ├── lc_rag.py        랭체인 구현 (E5 어댑터, QdrantVectorStore)
    └── lc_eval.py       같은 골드셋으로 랭체인 측정
```

## 실행

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
docker run -d --name qdrant -p 6333:6333 -v "$PWD/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

```bash
python search.py                      # numpy 검색
python qd_index.py                    # Qdrant에 적재
python qd_search.py                   # Qdrant 검색
python eval_run.py                    # 평가 + 하한선 sweep
python -m langchain_lab.lc_rag        # 랭체인 (저장소 루트에서 -m으로 실행)
python -m langchain_lab.lc_eval
```

데이터는 저장소에 없다. `load.py`의 `SRC`를 결정 JSON 폴더로 바꾸고(필드: `id`, `area`, `topic`, `status`, `decision`, `rationale`), `golden.example.jsonl`을 참고해 `golden.jsonl`을 만들어야 평가를 돌릴 수 있다.

## 다음에 할 것
- 재랭킹(cross-encoder) 추가 후 Top1, 거절 변화 측정
- 하이브리드 검색(키워드 + 벡터), exact 문항 개선 확인
- LLM 결합: 검색 결과로 프롬프트 조립 후 답변 생성, 답변 평가
- 골드셋 확대 후 하한선 재검증
