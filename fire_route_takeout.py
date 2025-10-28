from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import requests, json, time, re

# ===== 설정 =====
MAIN_URL = "https://fd.forest.go.kr/ffas/pubConn/movePage/sub1.do"
DATA_URL = "https://fd.forest.go.kr/ffas/pubConn/selectSttnMapFeatureList.do"
OUTFILE = "fire_routes_auto.json"
# =================

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(MAIN_URL)
wait = WebDriverWait(driver, 30)

print("🌲 페이지가 열렸습니다.")
print("👉 날짜를 설정하고, '진화완료' 선택 후 '검색'을 누르세요.")
input("⚡ 검색 완료 후 Enter 키를 눌러주세요...")

# 1️⃣ ‘상황도 보기’ 링크 모두 찾기
wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "a.btn1.img")))
links = driver.find_elements(By.CSS_SELECTOR, "a.btn1.img")

print(f"🔗 상황도 링크 {len(links)}개 발견")

route_data = []

# 2️⃣ 각 링크의 URL 또는 sttnMapId 추출
for idx, link in enumerate(links, start=1):
    href = link.get_attribute("onclick") or link.get_attribute("href") or ""
    # onclick 속성에서 ID 추출
    match = re.search(r"['\"](\d{17,})['\"]", href)
    if not match:
        # 새 창 링크일 수도 있음
        href2 = link.get_attribute("onclick")
        match = re.search(r"sttnMapId=(\d+)", href2 or "")
    if not match:
        continue

    sttn_id = match.group(1)
    print(f"📍 {idx}. sttnMapId={sttn_id}")

    # 3️⃣ 서버에 직접 요청해서 JSON 응답 받기
    payload = {"sttnMapId": sttn_id}
    res = requests.post(DATA_URL, data=payload)
    if res.status_code != 200:
        print(f"❌ 요청 실패 ({res.status_code})")
        continue

    try:
        res_json = res.json()
        route_data.append({
            "sttnMapId": sttn_id,
            "features": res_json.get("sttnMapFeatureList", [])
        })
        print(f"✅ {sttn_id} 지도 데이터 수집 완료 ({len(res_json.get('sttnMapFeatureList', []))}개 feature)")
    except Exception as e:
        print(f"⚠️ JSON 파싱 실패 ({sttn_id}): {e}")

    time.sleep(1)

# 4️⃣ 결과 저장
with open(OUTFILE, "w", encoding="utf-8") as f:
    json.dump(route_data, f, ensure_ascii=False, indent=2)

print(f"\n🔥 총 {len(route_data)}건의 상황도 데이터 저장 완료 → {OUTFILE}")

driver.quit()
