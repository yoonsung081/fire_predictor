import re
import sys

# Ensure stdout uses UTF-8 for correct display of Korean characters
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def parse_detailed_cv_logs(log_data):
    """
    상세 CV 로그 데이터에서 하이퍼파라미터, 평균 F1-score, 표준 편차를 파싱합니다.
    
    Args:
        log_data (str): 상세 CV 로그가 포함된 전체 텍스트.
        
    Returns:
        list: 각 CV 실행에 대한 정보를 담은 딕셔너리 리스트.
              각 딕셔너리는 'hyperparameters' (dict), 'mean_f1_score' (float),
              'std_f1_score' (float) 키를 가집니다.
    """
    results = []
    # 전체 로그 라인 패턴: PARAMS: ...; MEAN_F1_SCORE: ...; STD_F1_SCORE: ...
    main_pattern = re.compile(
        r"PARAMS: (.*); MEAN_F1_SCORE: ([\d.]+); STD_F1_SCORE: ([\d.]+)"
    )
    
    for line in log_data.split('\n'):
        main_match = main_pattern.search(line)
        if main_match:
            params_str = main_match.group(1)
            mean_f1_score = float(main_match.group(2))
            std_f1_score = float(main_match.group(3))
            
            # 개별 하이퍼파라미터 파싱
            params = {}
            param_pairs = re.findall(r'(\w+)=([\d.\-]+)', params_str)
            for key, value in param_pairs:
                try:
                    params[key] = int(value)
                except ValueError:
                    try:
                        params[key] = float(value)
                    except ValueError:
                        params[key] = value # 변환 실패 시 문자열로 유지
            
            results.append({
                'hyperparameters': params,
                'mean_f1_score': mean_f1_score,
                'std_f1_score': std_f1_score
            })
    return results

# --- 사용 방법 ---
# 1. train_model.py를 실행하여 출력된 상세 로그를 아래 'your_detailed_cv_logs' 변수에 붙여넣으세요.
#    (예시 로그는 이전에 제공해주신 형식에 맞춰 작성되었습니다.)
your_detailed_cv_logs = """
PARAMS: subsample=0.8, num_leaves=25, n_estimators=700, min_child_samples=20, max_depth=8, learning_rate=0.008, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7642; STD_F1_SCORE: 0.0063
PARAMS: subsample=0.75, num_leaves=40, n_estimators=700, min_child_samples=20, max_depth=8, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7601; STD_F1_SCORE: 0.0154
PARAMS: subsample=0.85, num_leaves=40, n_estimators=500, min_child_samples=25, max_depth=8, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7560; STD_F1_SCORE: 0.0132
PARAMS: subsample=0.75, num_leaves=31, n_estimators=400, min_child_samples=15, max_depth=12, learning_rate=0.008, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7714; STD_F1_SCORE: 0.0051
PARAMS: subsample=0.75, num_leaves=31, n_estimators=600, min_child_samples=15, max_depth=8, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7636; STD_F1_SCORE: 0.0093
PARAMS: subsample=0.8, num_leaves=40, n_estimators=500, min_child_samples=25, max_depth=10, learning_rate=0.008, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7642; STD_F1_SCORE: 0.0132
PARAMS: subsample=0.85, num_leaves=31, n_estimators=400, min_child_samples=15, max_depth=10, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7678; STD_F1_SCORE: 0.0073
PARAMS: subsample=0.75, num_leaves=25, n_estimators=600, min_child_samples=15, max_depth=10, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7652; STD_F1_SCORE: 0.0054
PARAMS: subsample=0.85, num_leaves=31, n_estimators=600, min_child_samples=25, max_depth=10, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7574; STD_F1_SCORE: 0.0077
PARAMS: subsample=0.8, num_leaves=40, n_estimators=500, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7660; STD_F1_SCORE: 0.0122
PARAMS: subsample=0.8, num_leaves=40, n_estimators=600, min_child_samples=25, max_depth=8, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7520; STD_F1_SCORE: 0.0104
PARAMS: subsample=0.85, num_leaves=31, n_estimators=600, min_child_samples=15, max_depth=12, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7686; STD_F1_SCORE: 0.0077
PARAMS: subsample=0.8, num_leaves=25, n_estimators=700, min_child_samples=25, max_depth=10, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7548; STD_F1_SCORE: 0.0091
PARAMS: subsample=0.75, num_leaves=40, n_estimators=400, min_child_samples=25, max_depth=12, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7582; STD_F1_SCORE: 0.0103
PARAMS: subsample=0.75, num_leaves=25, n_estimators=400, min_child_samples=15, max_depth=8, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7637; STD_F1_SCORE: 0.0100
PARAMS: subsample=0.8, num_leaves=31, n_estimators=400, min_child_samples=20, max_depth=12, learning_rate=0.008, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7662; STD_F1_SCORE: 0.0065
PARAMS: subsample=0.8, num_leaves=25, n_estimators=700, min_child_samples=20, max_depth=12, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7602; STD_F1_SCORE: 0.0147
PARAMS: subsample=0.8, num_leaves=31, n_estimators=500, min_child_samples=15, max_depth=10, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7671; STD_F1_SCORE: 0.0077
PARAMS: subsample=0.75, num_leaves=40, n_estimators=600, min_child_samples=15, max_depth=12, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7607; STD_F1_SCORE: 0.0140
PARAMS: subsample=0.85, num_leaves=40, n_estimators=700, min_child_samples=15, max_depth=12, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7588; STD_F1_SCORE: 0.0167
PARAMS: subsample=0.8, num_leaves=25, n_estimators=600, min_child_samples=15, max_depth=10, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7619; STD_F1_SCORE: 0.0031
PARAMS: subsample=0.85, num_leaves=40, n_estimators=500, min_child_samples=25, max_depth=10, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7642; STD_F1_SCORE: 0.0112
PARAMS: subsample=0.8, num_leaves=31, n_estimators=700, min_child_samples=25, max_depth=12, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7592; STD_F1_SCORE: 0.0100
PARAMS: subsample=0.75, num_leaves=40, n_estimators=400, min_child_samples=15, max_depth=8, learning_rate=0.008, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7696; STD_F1_SCORE: 0.0069
PARAMS: subsample=0.85, num_leaves=25, n_estimators=400, min_child_samples=15, max_depth=12, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7675; STD_F1_SCORE: 0.0070
PARAMS: subsample=0.8, num_leaves=31, n_estimators=400, min_child_samples=25, max_depth=10, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7541; STD_F1_SCORE: 0.0111
PARAMS: subsample=0.8, num_leaves=40, n_estimators=700, min_child_samples=25, max_depth=12, learning_rate=0.008, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7566; STD_F1_SCORE: 0.0161
PARAMS: subsample=0.85, num_leaves=40, n_estimators=400, min_child_samples=20, max_depth=12, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7639; STD_F1_SCORE: 0.0112
PARAMS: subsample=0.75, num_leaves=40, n_estimators=600, min_child_samples=15, max_depth=12, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7562; STD_F1_SCORE: 0.0138
PARAMS: subsample=0.75, num_leaves=25, n_estimators=500, min_child_samples=20, max_depth=10, learning_rate=0.008, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7635; STD_F1_SCORE: 0.0086
PARAMS: subsample=0.75, num_leaves=31, n_estimators=400, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7622; STD_F1_SCORE: 0.0107
PARAMS: subsample=0.8, num_leaves=31, n_estimators=600, min_child_samples=15, max_depth=8, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7606; STD_F1_SCORE: 0.0070
PARAMS: subsample=0.8, num_leaves=25, n_estimators=500, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7624; STD_F1_SCORE: 0.0084
PARAMS: subsample=0.85, num_leaves=25, n_estimators=400, min_child_samples=20, max_depth=8, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7653; STD_F1_SCORE: 0.0120
PARAMS: subsample=0.75, num_leaves=25, n_estimators=400, min_child_samples=15, max_depth=12, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7673; STD_F1_SCORE: 0.0075
PARAMS: subsample=0.8, num_leaves=31, n_estimators=600, min_child_samples=25, max_depth=10, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7574; STD_F1_SCORE: 0.0077
PARAMS: subsample=0.85, num_leaves=40, n_estimators=400, min_child_samples=20, max_depth=8, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7639; STD_F1_SCORE: 0.0113
PARAMS: subsample=0.85, num_leaves=40, n_estimators=700, min_child_samples=25, max_depth=10, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7581; STD_F1_SCORE: 0.0127
PARAMS: subsample=0.75, num_leaves=31, n_estimators=400, min_child_samples=15, max_depth=10, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7684; STD_F1_SCORE: 0.0099
PARAMS: subsample=0.75, num_leaves=31, n_estimators=600, min_child_samples=15, max_depth=8, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7578; STD_F1_SCORE: 0.0115
PARAMS: subsample=0.85, num_leaves=40, n_estimators=700, min_child_samples=15, max_depth=8, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7618; STD_F1_SCORE: 0.0105
PARAMS: subsample=0.8, num_leaves=40, n_estimators=700, min_child_samples=20, max_depth=12, learning_rate=0.008, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7604; STD_F1_SCORE: 0.0138
PARAMS: subsample=0.75, num_leaves=40, n_estimators=700, min_child_samples=15, max_depth=10, learning_rate=0.008, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7666; STD_F1_SCORE: 0.0094
PARAMS: subsample=0.85, num_leaves=31, n_estimators=400, min_child_samples=20, max_depth=8, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7627; STD_F1_SCORE: 0.0131
PARAMS: subsample=0.75, num_leaves=31, n_estimators=700, min_child_samples=15, max_depth=8, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7575; STD_F1_SCORE: 0.0140
PARAMS: subsample=0.85, num_leaves=40, n_estimators=400, min_child_samples=15, max_depth=12, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7660; STD_F1_SCORE: 0.0100
PARAMS: subsample=0.75, num_leaves=25, n_estimators=600, min_child_samples=25, max_depth=12, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7579; STD_F1_SCORE: 0.0073
PARAMS: subsample=0.75, num_leaves=31, n_estimators=700, min_child_samples=25, max_depth=10, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7602; STD_F1_SCORE: 0.0148
PARAMS: subsample=0.75, num_leaves=25, n_estimators=700, min_child_samples=20, max_depth=8, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7568; STD_F1_SCORE: 0.0034
PARAMS: subsample=0.75, num_leaves=31, n_estimators=400, min_child_samples=15, max_depth=8, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7667; STD_F1_SCORE: 0.0084
PARAMS: subsample=0.85, num_leaves=31, n_estimators=500, min_child_samples=20, max_depth=10, learning_rate=0.008, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7625; STD_F1_SCORE: 0.0129
PARAMS: subsample=0.85, num_leaves=31, n_estimators=600, min_child_samples=25, max_depth=10, learning_rate=0.008, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7571; STD_F1_SCORE: 0.0080
PARAMS: subsample=0.75, num_leaves=40, n_estimators=500, min_child_samples=20, max_depth=8, learning_rate=0.008, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7696; STD_F1_SCORE: 0.0154
PARAMS: subsample=0.85, num_leaves=25, n_estimators=500, min_child_samples=15, max_depth=10, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7715; STD_F1_SCORE: 0.0094
PARAMS: subsample=0.8, num_leaves=31, n_estimators=400, min_child_samples=15, max_depth=12, learning_rate=0.008, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7714; STD_F1_SCORE: 0.0051
PARAMS: subsample=0.8, num_leaves=25, n_estimators=700, min_child_samples=15, max_depth=8, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7637; STD_F1_SCORE: 0.0097
PARAMS: subsample=0.8, num_leaves=31, n_estimators=400, min_child_samples=15, max_depth=10, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7701; STD_F1_SCORE: 0.0058
PARAMS: subsample=0.8, num_leaves=40, n_estimators=700, min_child_samples=20, max_depth=12, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7596; STD_F1_SCORE: 0.0121
PARAMS: subsample=0.75, num_leaves=25, n_estimators=700, min_child_samples=15, max_depth=12, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7617; STD_F1_SCORE: 0.0076
PARAMS: subsample=0.85, num_leaves=40, n_estimators=500, min_child_samples=15, max_depth=12, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7626; STD_F1_SCORE: 0.0118
PARAMS: subsample=0.8, num_leaves=31, n_estimators=700, min_child_samples=20, max_depth=8, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7618; STD_F1_SCORE: 0.0143
PARAMS: subsample=0.8, num_leaves=31, n_estimators=600, min_child_samples=25, max_depth=8, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7592; STD_F1_SCORE: 0.0074
PARAMS: subsample=0.85, num_leaves=40, n_estimators=700, min_child_samples=25, max_depth=12, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7587; STD_F1_SCORE: 0.0138
PARAMS: subsample=0.85, num_leaves=31, n_estimators=600, min_child_samples=20, max_depth=12, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7576; STD_F1_SCORE: 0.0171
PARAMS: subsample=0.8, num_leaves=25, n_estimators=400, min_child_samples=25, max_depth=10, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7586; STD_F1_SCORE: 0.0088
PARAMS: subsample=0.85, num_leaves=40, n_estimators=600, min_child_samples=20, max_depth=12, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7649; STD_F1_SCORE: 0.0151
PARAMS: subsample=0.85, num_leaves=25, n_estimators=400, min_child_samples=15, max_depth=8, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7717; STD_F1_SCORE: 0.0073
PARAMS: subsample=0.8, num_leaves=25, n_estimators=700, min_child_samples=15, max_depth=12, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7617; STD_F1_SCORE: 0.0076
PARAMS: subsample=0.75, num_leaves=40, n_estimators=700, min_child_samples=25, max_depth=12, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7549; STD_F1_SCORE: 0.0182
PARAMS: subsample=0.85, num_leaves=31, n_estimators=700, min_child_samples=15, max_depth=12, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7663; STD_F1_SCORE: 0.0096
PARAMS: subsample=0.75, num_leaves=31, n_estimators=400, min_child_samples=20, max_depth=8, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7676; STD_F1_SCORE: 0.0099
PARAMS: subsample=0.85, num_leaves=40, n_estimators=700, min_child_samples=15, max_depth=12, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7588; STD_F1_SCORE: 0.0167
PARAMS: subsample=0.75, num_leaves=31, n_estimators=400, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7609; STD_F1_SCORE: 0.0093
PARAMS: subsample=0.85, num_leaves=40, n_estimators=700, min_child_samples=20, max_depth=10, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7616; STD_F1_SCORE: 0.0074
PARAMS: subsample=0.8, num_leaves=31, n_estimators=500, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7609; STD_F1_SCORE: 0.0127
PARAMS: subsample=0.85, num_leaves=25, n_estimators=500, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7624; STD_F1_SCORE: 0.0084
PARAMS: subsample=0.75, num_leaves=31, n_estimators=700, min_child_samples=25, max_depth=12, learning_rate=0.008, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7573; STD_F1_SCORE: 0.0051
PARAMS: subsample=0.85, num_leaves=31, n_estimators=600, min_child_samples=15, max_depth=12, learning_rate=0.008, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7682; STD_F1_SCORE: 0.0068
PARAMS: subsample=0.75, num_leaves=31, n_estimators=600, min_child_samples=25, max_depth=10, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7568; STD_F1_SCORE: 0.0102
PARAMS: subsample=0.85, num_leaves=31, n_estimators=700, min_child_samples=20, max_depth=12, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7600; STD_F1_SCORE: 0.0175
PARAMS: subsample=0.8, num_leaves=25, n_estimators=600, min_child_samples=15, max_depth=12, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7559; STD_F1_SCORE: 0.0061
PARAMS: subsample=0.75, num_leaves=25, n_estimators=700, min_child_samples=25, max_depth=8, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7521; STD_F1_SCORE: 0.0097
PARAMS: subsample=0.85, num_leaves=40, n_estimators=400, min_child_samples=25, max_depth=12, learning_rate=0.008, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7567; STD_F1_SCORE: 0.0120
PARAMS: subsample=0.75, num_leaves=31, n_estimators=400, min_child_samples=15, max_depth=8, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7613; STD_F1_SCORE: 0.0131
PARAMS: subsample=0.8, num_leaves=25, n_estimators=400, min_child_samples=25, max_depth=12, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7594; STD_F1_SCORE: 0.0068
PARAMS: subsample=0.8, num_leaves=25, n_estimators=400, min_child_samples=15, max_depth=12, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7675; STD_F1_SCORE: 0.0070
PARAMS: subsample=0.85, num_leaves=40, n_estimators=700, min_child_samples=15, max_depth=8, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7633; STD_F1_SCORE: 0.0092
PARAMS: subsample=0.8, num_leaves=25, n_estimators=700, min_child_samples=20, max_depth=12, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7581; STD_F1_SCORE: 0.0092
PARAMS: subsample=0.8, num_leaves=31, n_estimators=400, min_child_samples=25, max_depth=10, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7543; STD_F1_SCORE: 0.0086
PARAMS: subsample=0.75, num_leaves=25, n_estimators=700, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7617; STD_F1_SCORE: 0.0080
PARAMS: subsample=0.85, num_leaves=40, n_estimators=600, min_child_samples=20, max_depth=8, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7596; STD_F1_SCORE: 0.0162
PARAMS: subsample=0.75, num_leaves=31, n_estimators=400, min_child_samples=15, max_depth=8, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7682; STD_F1_SCORE: 0.0078
PARAMS: subsample=0.8, num_leaves=40, n_estimators=400, min_child_samples=20, max_depth=8, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7669; STD_F1_SCORE: 0.0089
PARAMS: subsample=0.75, num_leaves=25, n_estimators=700, min_child_samples=20, max_depth=10, learning_rate=0.012, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7601; STD_F1_SCORE: 0.0092
PARAMS: subsample=0.85, num_leaves=40, n_estimators=500, min_child_samples=15, max_depth=12, learning_rate=0.01, colsample_bytree=0.85; MEAN_F1_SCORE: 0.7633; STD_F1_SCORE: 0.0123
PARAMS: subsample=0.85, num_leaves=40, n_estimators=500, min_child_samples=25, max_depth=12, learning_rate=0.012, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7582; STD_F1_SCORE: 0.0164
PARAMS: subsample=0.85, num_leaves=40, n_estimators=400, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7686; STD_F1_SCORE: 0.0140
PARAMS: subsample=0.8, num_leaves=31, n_estimators=700, min_child_samples=25, max_depth=8, learning_rate=0.008, colsample_bytree=0.9; MEAN_F1_SCORE: 0.7592; STD_F1_SCORE: 0.0071
PARAMS: subsample=0.85, num_leaves=25, n_estimators=700, min_child_samples=20, max_depth=10, learning_rate=0.01, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7593; STD_F1_SCORE: 0.0084
PARAMS: subsample=0.85, num_leaves=31, n_estimators=600, min_child_samples=20, max_depth=12, learning_rate=0.012, colsample_bytree=0.95; MEAN_F1_SCORE: 0.7597; STD_F1_SCORE: 0.0137
"""

# 로그 데이터 파싱
parsed_results = parse_detailed_cv_logs(your_detailed_cv_logs)

# 파싱된 결과 출력
print("--- 파싱된 CV 결과 ---")
for result in parsed_results:
    print(f"하이퍼파라미터: {result['hyperparameters']}")
    print(f"  평균 F1-score: {result['mean_f1_score']:.4f}")
    print(f"  F1-score 표준 편차: {result['std_f1_score']:.4f}\n")

# --- 추가 분석 예시: 가장 좋은 성능을 보인 조합 찾기 ---
if parsed_results:
    best_result = max(parsed_results, key=lambda x: x['mean_f1_score'])
    print("\n--- 가장 좋은 성능을 보인 조합 ---")
    print(f"하이퍼파라미터: {best_result['hyperparameters']}")
    print(f"평균 F1-score: {best_result['mean_f1_score']:.4f}")
    print(f"F1-score 표준 편차: {best_result['std_f1_score']:.4f}")

# --- 추가 분석 예시: 특정 하이퍼파라미터의 영향 시각화 (matplotlib, seaborn 필요) ---
# 이 부분은 설치된 라이브러리에 따라 실행 가능합니다.
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd

# if parsed_results:
#     df_results = pd.DataFrame(parsed_results)
#     df_params = pd.json_normalize(df_results['hyperparameters'])
#     df_final = pd.concat([df_params, df_results[['mean_f1_score', 'std_f1_score']]], axis=1)

#     print("\n--- 하이퍼파라미터별 성능 시각화 (예시) ---")
#     # 예: learning_rate에 따른 F1-score 변화
#     if 'learning_rate' in df_final.columns:
#         plt.figure(figsize=(10, 6))
#         sns.lineplot(x='learning_rate', y='mean_f1_score', data=df_final, marker='o')
#         plt.title('Mean F1-score vs Learning Rate')
#         plt.xlabel('Learning Rate')
#         plt.ylabel('Mean F1-score')
#         plt.grid(True)
#         plt.show()

#     # 예: num_leaves에 따른 F1-score 변화
#     if 'num_leaves' in df_final.columns:
#         plt.figure(figsize=(10, 6))
#         sns.boxplot(x='num_leaves', y='mean_f1_score', data=df_final)
#         plt.title('Mean F1-score vs Num Leaves')
#         plt.xlabel('Num Leaves')
#         plt.ylabel('Mean F1-score')
#         plt.grid(True)
#         plt.show()