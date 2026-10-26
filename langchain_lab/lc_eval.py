from eval_run import pick_items, evaluate # 이전에 만든 평가 코드를 그대로 재사용
from evaluation import aggregate_retrieval
from langchain_lab import lc_rag # 같은 폴더가 아니라 패키지 경로로 가져온다

items = pick_items() # 36문항
print("langchain", aggregate_retrieval(evaluate(lc_rag.search, items)))
