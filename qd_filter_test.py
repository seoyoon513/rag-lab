from qd_search import search, search_filtered # 필터 없는 검색과 있는 검색을 비교한다

Q = "폴더 이름 글자수"

print("--- 필터 없음")
for t, s in search(Q):
    print(f"  {s:.3f}  {t}")

print("--- 필터: 영역=북마크, 상태=최종결정")
for t, st, s in search_filtered(Q, area="북마크", status="최종결정"):
    print(f"  {s:.3f}  [{st}] {t}")

print("--- 필터: 상태=제안")
for t, st, s in search_filtered(Q, status="제안"):
    print(f"  {s:.3f}  [{st}] {t}")
