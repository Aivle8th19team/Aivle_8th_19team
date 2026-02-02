# Delay Prediction 연동 수정 완료 가이드

## 수정 내용

### 1. ml-service API 경로 수정 ✅
- **변경 전**: `/api/v1/delay/prediction/*`
- **변경 후**: `/api/v1/delay-prediction/*`
- **이유**: frontend가 호출하는 경로와 일치시키기 위함

변경된 엔드포인트:
- `GET /api/v1/delay-prediction/overview` - 전체 주문 예측 개요
- `GET /api/v1/delay-prediction/orders/{order_id}` - 특정 주문 예측
- `POST /api/v1/delay-prediction/train` - 모델 재훈련
- `GET /api/v1/delay-prediction/status` - 서비스 상태 확인

### 2. DB 연결 강화 ✅
**파일**: `ml-service/delay_prediction/service.py`

```python
# 개선 사항:
- extract_orders(): DB 연결 체크 및 에러 핸들링 강화
- extract_events(): 로깅 추가 (✓ 성공 / ❌ 실패)
- NULL 처리: completed_at이 NULL인 경우 actual_delay_hours = 0으로 처리
```

### 3. 환경 변수 설정 ✅
**생성 파일**: `ml-service/.env.example`

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/automobile_risk
PORT=8000
MODEL_PATH=../testbed/real_delay_model.pkl
```

실제 사용 시 `.env.example`을 `.env`로 복사하여 사용:
```bash
cd ml-service
cp .env.example .env
```

### 4. 주문 완료 상태 업데이트 SQL ✅
**생성 파일**: `backend/update_orders_completed.sql`

PostgreSQL에서 실행:
```bash
psql -U postgres -d automobile_risk -f backend/update_orders_completed.sql
```

이 스크립트는:
- 주문 10건을 COMPLETED 상태로 변경
- completed_at 시간을 due_date + 2~26시간 사이로 설정
- actual_delay_hours가 계산되도록 함

### 5. Frontend 수정 ✅
**파일**: `frontend/src/components/MainDashboard.tsx`

```typescript
// 예측 납기일 계산 로직 개선
const predictedDeadlineFromML = (() => {
  const totalDelayH = currentPrediction?.predDelayMaxH ?? totalDelayHours;
  const maxDelayH = activePrediction?.maxDelayHours ?? 0;
  const finalDelayHours = Math.max(totalDelayH, maxDelayH);
  return new Date(earliestDeadline.getTime() + finalDelayHours * 60 * 60 * 1000);
})();
```

## 테스트 절차

### 1단계: DB 준비
```sql
-- PostgreSQL 접속
psql -U postgres -d automobile_risk

-- 주문 상태 확인
SELECT order_status, COUNT(*) FROM orders GROUP BY order_status;

-- 주문 10건 완료로 업데이트
\i backend/update_orders_completed.sql

-- 결과 확인
SELECT order_id, order_status, completed_at, 
       EXTRACT(EPOCH FROM (completed_at - due_date))/3600 as delay_hours
FROM orders 
WHERE order_status = 'COMPLETED'
LIMIT 10;
```

### 2단계: ml-service 시작
```bash
cd ml-service

# 환경 설정 (필요시)
cp .env.example .env

# 의존성 설치 (필요시)
pip install -r requirements.txt

# 서비스 시작
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

서비스 상태 확인:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/delay-prediction/status
```

### 3단계: 데이터 연동 테스트
```bash
# 전체 예측 조회
curl http://localhost:8000/api/v1/delay-prediction/overview | jq

# 응답 예시:
{
  "total_orders": 15,
  "max_delay_hours": 12.5,
  "avg_delay_hours": 5.2,
  "risk_distribution": {
    "LOW": 5,
    "MEDIUM": 7,
    "HIGH": 3,
    "CRITICAL": 0
  },
  "process_breakdown": [
    {"process": "welding", "total_score": 45.2, "count": 8},
    {"process": "paint", "total_score": 38.7, "count": 6}
  ],
  "orders": [...]
}
```

### 4단계: Backend 시작
```bash
cd backend
./gradlew bootRun
```

Backend API 확인:
```bash
curl http://localhost:8080/api/v1/delay-prediction/overview
```

### 5단계: Frontend 확인
```bash
cd frontend
npm run dev
```

브라우저에서 확인:
1. http://localhost:5173 접속
2. MainDashboard 페이지로 이동
3. "납기 예측 분석" 섹션 확인:
   - **실제 납기 예측일**: ML 계산된 날짜가 표시되어야 함
   - **예상 납기일 (ML)**: 동일한 날짜 표시
   - **초기 대비 지연**: 실제 지연 시간 (시간 단위)
4. 하단 생산 상태:
   - **대기 중**: PLANNED 상태 생산 건수
   - **생산 진행중**: IN_PROGRESS 상태 생산 건수
   - **생산 완료**: COMPLETED 상태 생산 건수

## 문제 해결

### DB 연결 실패
```
❌ DB 연결 실패: could not connect to server
```

**해결**:
1. PostgreSQL 서비스 실행 확인
2. DATABASE_URL 확인 (포트, 사용자명, 비밀번호)
3. 방화벽 설정 확인

### 데이터가 없음
```json
{
  "total_orders": 0,
  "orders": []
}
```

**해결**:
1. DB에 주문 데이터 확인: `SELECT COUNT(*) FROM orders;`
2. 완료된 주문 확인: `SELECT COUNT(*) FROM orders WHERE order_status = 'COMPLETED';`
3. update_orders_completed.sql 재실행

### 모델 훈련 실패
```
완료된 주문이 5개뿐입니다. 최소 20개 필요
```

**해결**:
- 규칙 기반 예측이 자동으로 적용됨 (모델 없이 동작)
- 20개 이상 완료 주문 생성 후 재훈련:
```bash
curl -X POST http://localhost:8000/api/v1/delay-prediction/train
```

### Frontend에서 0으로 표시
**확인사항**:
1. 브라우저 DevTools Console에서 로그 확인
2. Network 탭에서 `/api/v1/delay-prediction/overview` 응답 확인
3. productions 배열의 status 필드 값 확인

## 확인 포인트

✅ **ml-service 로그**:
```
✓ 주문 데이터 추출 완료: 50건
✓ 이벤트 데이터 추출 완료: 123건
✓ 납기 예측 모델 로드 성공
```

✅ **Frontend Console**:
```
[Dashboard] Fetched: {...}
[Prediction] Fetched: {total_orders: 15, ...}
Orders fetched: 50
Productions fetched: 30
```

✅ **Dashboard 표시**:
- 실제 납기 예측일이 현재 시간이 아닌 계산된 날짜
- 지연 시간이 0이 아닌 실제 값
- 생산 상태 카운트가 0이 아닌 실제 건수

## 추가 개선 사항 (선택)

1. **캐싱 추가**: Redis를 사용하여 예측 결과 캐싱
2. **비동기 훈련**: Celery로 모델 훈련을 백그라운드 작업으로
3. **알림**: 위험도가 HIGH 이상일 때 알림 발송
4. **대시보드 확장**: 시계열 차트로 예측 추이 표시
