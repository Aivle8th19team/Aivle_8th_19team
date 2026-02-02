-- 주문 데이터 10건을 COMPLETED 상태로 업데이트
-- automobile_risk 데이터베이스에서 실행

-- 먼저 업데이트할 주문 확인
SELECT order_id, order_status, order_date, due_date 
FROM orders 
WHERE order_status != 'COMPLETED' 
LIMIT 10;

-- 주문 10건을 COMPLETED로 업데이트 (completed_at 시간도 설정)
UPDATE orders 
SET 
    order_status = 'COMPLETED',
    completed_at = due_date + INTERVAL '2 hours' + (RANDOM() * INTERVAL '24 hours'),
    updated_at = NOW()
WHERE order_id IN (
    SELECT order_id 
    FROM orders 
    WHERE order_status != 'COMPLETED' 
    ORDER BY order_date 
    LIMIT 10
);

-- 업데이트 결과 확인
SELECT 
    order_id, 
    order_status, 
    order_date,
    due_date,
    completed_at,
    EXTRACT(EPOCH FROM (completed_at - due_date))/3600 as actual_delay_hours
FROM orders 
WHERE order_status = 'COMPLETED'
ORDER BY completed_at DESC
LIMIT 10;

-- 전체 주문 상태 분포 확인
SELECT 
    order_status, 
    COUNT(*) as count 
FROM orders 
GROUP BY order_status
ORDER BY count DESC;
