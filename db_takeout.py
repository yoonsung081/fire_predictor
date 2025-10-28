from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time, json

# ===== 설정 =====
URL = "https://fd.forest.go.kr/ffas/pubConn/movePage/sub1.do"
OUTFILE = "fire_data_total.json"
TARGET_COUNT = 446   # ✅ 목표 수집 건수
# =================

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get(URL)

wait = WebDriverWait(driver, 30)

print("🌲 페이지가 열렸습니다.")
print("👉 날짜를 2024-09-29 ~ 2025-12-31로 설정하고, '진화완료' 선택 후 '검색'을 누르세요.")
input("⚡ 검색이 완료되면 Enter 키를 눌러주세요...")

all_data = []
page = 1

while True:
    try:
        # 테이블 로드 대기
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "table tbody tr")))
        time.sleep(1)

        rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
        page_data = []

        for row in rows:
            cols = [td.text.strip() for td in row.find_elements(By.TAG_NAME, "td")]
            if cols:
                page_data.append(cols)

        all_data.extend(page_data)
        print(f"✅ {page}페이지 수집 완료 ({len(page_data)}건 / 누적 {len(all_data)}건)")

        # 🔸 목표 건수 도달 시 종료
        if len(all_data) >= TARGET_COUNT:
            print(f"🎯 목표 수집량({TARGET_COUNT}건) 도달 — 수집 종료")
            break

        # 🔹 다음 페이지 버튼 찾기
        next_selectors = [
            "//a[contains(., '다음') and not(contains(@class, 'disabled'))]",
            "//button[contains(., '다음') and not(@disabled)]",
            "//li[@class='paginate_button next']/a[not(contains(@class,'disabled'))]",
            "//a[@aria-label='다음 페이지']",
        ]

        next_button = None
        for sel in next_selectors:
            btns = driver.find_elements(By.XPATH, sel)
            if btns:
                next_button = btns[0]
                break

        if not next_button:
            print("📘 다음 페이지 버튼 없음 — 마지막 페이지 도달")
            break

        # 다음 페이지 클릭
        driver.execute_script("arguments[0].click();", next_button)
        page += 1
        time.sleep(2)

    except Exception as e:
        print(f"⚠️ 오류 발생 (페이지 {page}): {e}")
        break

# 결과 저장
with open(OUTFILE, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"\n🔥 총 {len(all_data)}건 저장 완료 → {OUTFILE}")
driver.quit()
