from evaluation import load_golden, retrieval_scores, aggregate_retrieval, by_type # 지표 함수들
from load import load_docs

def pick_items():
    titles = [d["title"] for d in load_docs()] # 우리 인덱스에 있는 제목 58개
    items = load_golden("golden.jsonl") # 골드셋. 59문항
    keep = []
    for it in items:
        if it.is_negative: # 정답이 없는 문항은 거절 테스트용으로 항상 포함
            keep.append(it)
        elif any(e in t for e in it.expected_titles for t in titles): # 정답 제목이 우리 문서 제목에 포함되면 포함
            keep.append(it)
    return keep

def evaluate(search_fn, items, k=5, min_score=0.0):
    rows = []
    for it in items:
        hits = search_fn(it.question, k) # [(제목, 점수), ...]
        titles = [t for t, s in hits if s >= min_score] # 하한선 미만 결과는 버리고 제목만 뽑는다
        rows.append(retrieval_scores(it, titles)) # 문항 하나 채점
    return rows

def sweep(search_fn, items, k=5):
    print("min_score  hit@k  top1   mrr    reject  overall  avg_n")
    for ms in [0.0, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90]: # 하한선 후보
        a = aggregate_retrieval(evaluate(search_fn, items, k, ms))
        print(f"{ms:<9}  {a['hit_at_k']:<5}  {a['top1']:<5}  {a['mrr']:<5}  {a['reject_acc']:<6}  {a['overall_acc']:<7}  {a['avg_results']}")

if __name__ == "__main__":
    import qd_search # Qdrant 검색
    import search as np_search # numpy 검색. 이름 충돌을 피하려고 별칭을 붙임
    items = pick_items()
    for name, fn in [("numpy", np_search.search), ("qdrant", qd_search.search)]:
        rows = evaluate(fn, items)
        print(name, aggregate_retrieval(rows))
    print("\n--- 하한선 sweep (qdrant)")
    sweep(qd_search.search, items)
