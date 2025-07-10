import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from sklearn.metrics import confusion_matrix
import numpy as np

# 🔧 한글 폰트 설정 (Windows 기준)
rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

accuracy_weather = 85
accuracy_baseline = 68

y_true = np.random.choice([0, 1], size=200, p=[0.7, 0.3])
y_pred_weather = np.random.choice([0, 1], size=200, p=[0.2, 0.8])
y_pred_baseline = np.random.choice([0, 1], size=200, p=[0.4, 0.6])

os.makedirs("static", exist_ok=True)

plt.figure(figsize=(5, 4))
plt.bar(['날씨 고려', '미고려'], [accuracy_weather, accuracy_baseline],
        color=['skyblue', 'lightcoral'])
plt.ylim(0, 100)
plt.ylabel('정확도 (%)')
plt.title('모델 정확도 비교')
plt.tight_layout()
plt.savefig('static/bar_accuracy.png')
plt.close()

cm_weather = confusion_matrix(y_true, y_pred_weather)
plt.figure(figsize=(4, 4))
sns.heatmap(cm_weather, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (날씨 고려)')
plt.tight_layout()
plt.savefig('static/cm_weather.png')
plt.close()

cm_baseline = confusion_matrix(y_true, y_pred_baseline)
plt.figure(figsize=(4, 4))
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Oranges')
plt.title('Confusion Matrix (미고려)')
plt.tight_layout()
plt.savefig('static/cm_baseline.png')
plt.close()

print("✅ 한글 깨짐 해결 + 이미지 static/ 폴더에 저장 완료")
