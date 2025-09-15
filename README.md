# koreaaimap
세종대학교 2025-2 창의학기제 프로젝트. 서울시 도시 데이터 기반 사용자 맞춤형 AI 장소 추천 서비스

혼잡한 도시 생활 속에서 출퇴근, 관광, 데이트, 여가생활과 같이 개인의 목적에 맞게 적합한 장소와 이동 경로를 찾는 것은 중요한 문제입니다. 실제로 잡코리아의 2024년 설문조사에서는 서울 거주 직장인들의 46.8프로가 출근길에 스트레스를 받는다고 답변했으며 이중 55.9프로가 사람이 너무 많은 만원 버스와 지하철 때문에 스트레스를 받는다고 답변했습니다. 하지만 현재 장소와 경로 추천 관련 플랫폼들은 단편적인 정보나 정적인 정보를 바탕으로 장소와 경로를 추천해줘 사람들의 이동 스트레스를 줄이는 데 한계를 보이고 있으며 실시간 동적 요소를 반영한 상황인지형 개인 맞춤 추천 기능의 필요성이 증대되고 있는 상황입니다.

한편, 서울시에서는 시민 편의 증진을 목표로 다양한 도시 데이터(실시간 혼잡도, 장소별 인구수, 문화 행사 정보, 교통 정보 등)를 공공 API 형태로 적극 개방하고 있고 이는 상황인지형 개인 맞춤 추천 시스템에 활용할 수 있습니다.

이에 따라 본 프로젝트에서는 서울시의 다양한 도시 데이터를 융합하여 사용자 개개인의 목적과 취향을 정확히 반영한 맞춤형 장소 및 경로 추천 서비스를 개발하고 사람들의 이동 스트레스 문제를 해결하고자 합니다.

# 활용 데이터
## 서울 실시간 도시 데이터
<img width="1276" height="1104" alt="image" src="https://github.com/user-attachments/assets/24fdda21-f9dd-4641-ab16-2e8851d3fdf1" />


# RECO(FastAPI) 골격


## 폴더 구조

```
reco/
  app/
    __init__.py
    main.py
    config.py
    instrumentation.py
    models.py          # 내부 모델(도메인)
    schemas.py         # Pydantic I/O 스키마
    services/
      __init__.py
      citydata_client.py
      scoring.py
      cache.py
    routers/
      __init__.py
      health.py
      reco.py
  tests/
    test_health.py
  requirements.txt
  Dockerfile
  gunicorn_conf.py
  .env.example
  Makefile
```

---

## 핵심 파일들

### 1) `app/config.py` — 설정/환경변수

```python
# app/config.py
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl
from typing import Optional

class Settings(BaseSettings):
    # 서비스 공통
    APP_NAME: str = "koreaaimap-reco"
    ENV: str = "prod"  # prod|staging|dev|local
    PORT: int = 9000
    LOG_LEVEL: str = "INFO"

    # 외부 연동: 우리 API 프록시(스프링) - citydata 엔드포인트
    CITYDATA_BASE_URL: AnyHttpUrl = "https://api.koreaaimap.com"

    # 캐시(옵션): 메모리/Redis 중 선택
    CACHE_BACKEND: str = "memory"  # memory|redis
    REDIS_URL: Optional[str] = None # e.g. redis://redis:6379/0
    CACHE_TTL_SECONDS: int = 60

    # 스코어 가중치
    WEIGHT_CONGESTION: float = 0.5   # 혼잡도(낮을수록 가산)
    WEIGHT_TRAVELTIME: float = 0.3   # 이동시간(짧을수록 가산)
    WEIGHT_EVENTFIT: float = 0.2     # 행사 매칭(사용자 관심과의 적합도)

    class Config:
        env_file = ".env"

settings = Settings()
```

### 2) `app/instrumentation.py` — 로깅/메트릭 미들웨어

```python
# app/instrumentation.py
import time, logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("reco")

class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
            return response
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(
                "method=%s path=%s status=%s rt_ms=%.1f ua=%s",
                request.method, request.url.path,
                getattr(request.state, "status_code", getattr(response, "status_code", "-")),
                elapsed, request.headers.get("user-agent", "-")
            )
```


### 3) `app/schemas.py` — 입력/출력 스키마

```python
# app/schemas.py
from pydantic import BaseModel, Field, conlist
from typing import List, Optional

class RecoInput(BaseModel):
    lat: float = Field(..., description="사용자 위도")
    lng: float = Field(..., description="사용자 경도")
    when: str = Field("now", description="추천 시점 (now|iso8601)")
    interests: Optional[List[str]] = Field(default=None, description="관심사 태그 (예: ['공원','축제'])")
    limit: int = Field(10, ge=1, le=50, description="반환 개수")

class RecoItem(BaseModel):
    place_code: str
    place_name: str
    score: float
    distance_m: Optional[int] = None
    congestion_level: Optional[str] = None
    travel_time_min: Optional[int] = None
    event_summary: Optional[str] = None

class RecoResponse(BaseModel):
    count: int
    items: conlist(RecoItem, min_length=0)
    updated_at: str
```

### 4) `app/models.py` — 내부 도메인 모델(간단)

```python
# app/models.py
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class Citydata:
    code: str
    name: str
    raw: Dict[str, Any]  # 원문(필요시 정규화)
```

### 5) `app/services/cache.py` — 캐시(메모리/Redis)

```python
# app/services/cache.py
import time
from typing import Any, Optional
from ..config import settings

class MemoryCache:
    def __init__(self, ttl: int):
        self.ttl = ttl
        self.store = {}

    def get(self, key: str) -> Optional[Any]:
        v = self.store.get(key)
        if not v: return None
        exp, data = v
        if time.time() > exp:
            self.store.pop(key, None)
            return None
        return data

    def set(self, key: str, value: Any):
        self.store[key] = (time.time() + self.ttl, value)

_cache = MemoryCache(ttl=settings.CACHE_TTL_SECONDS)

def cache_get(key: str):
    if settings.CACHE_BACKEND == "memory":
        return _cache.get(key)
    # Redis 구현 시 확장
    return None

def cache_set(key: str, value: Any):
    if settings.CACHE_BACKEND == "memory":
        _cache.set(key, value)
```

### 6) `app/services/citydata_client.py` — API 프록시 호출

```python
# app/services/citydata_client.py
import httpx, asyncio
from typing import Dict, Any, Optional
from ..config import settings
from .cache import cache_get, cache_set

TIMEOUT = httpx.Timeout(5.0, connect=3.0)

async def fetch_citydata_by_code(code: str) -> Optional[Dict[str, Any]]:
    cache_key = f"citydata:{code}"
    cached = cache_get(cache_key)
    if cached: return cached

    url = f"{settings.CITYDATA_BASE_URL}/api/v1/citydata"
    params = {"code": code}
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(url, params=params)
        if r.status_code != 200:
            return None
        data = r.json()
        cache_set(cache_key, data)
        return data
```

### 7) `app/services/scoring.py` — 간단 스코어링 로직(가중치 기반)

```python
# app/services/scoring.py
from typing import Dict, Any, Optional, List
from ..config import settings

def normalize_congestion(level: Optional[str]) -> float:
    # 예: '여유' 1.0, '보통' 0.6, '혼잡' 0.2
    m = {"여유": 1.0, "보통": 0.6, "혼잡": 0.2}
    return m.get(level or "보통", 0.6)

def normalize_travel_time(mins: Optional[int]) -> float:
    if mins is None: return 0.5
    # 10분 이내 1.0 → 60분 0 근사
    v = max(0.0, 1.0 - (mins/60.0))
    return round(v, 3)

def event_fit(events: Optional[List[str]], interests: Optional[List[str]]) -> float:
    if not events or not interests: return 0.5
    ev = set([e.lower() for e in events])
    ints = set([i.lower() for i in interests])
    inter = len(ev & ints)
    return min(1.0, 0.4 + 0.2*inter)  # 간단한 가점

def score_item(congestion_level: Optional[str], travel_time_min: Optional[int],
               event_tags: Optional[List[str]], interests: Optional[List[str]]) -> float:
    s = (
        settings.WEIGHT_CONGESTION * normalize_congestion(congestion_level) +
        settings.WEIGHT_TRAVELTIME * normalize_travel_time(travel_time_min) +
        settings.WEIGHT_EVENTFIT   * event_fit(event_tags, interests)
    )
    return round(s, 4)
```

### 8) `app/routers/health.py` — 헬스/레디니스/버전

```python
# app/routers/health.py
from fastapi import APIRouter
import os

router = APIRouter(prefix="", tags=["health"])

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/ready")
async def ready():
    # 외부 의존성(예: Redis 또는 우리 API)에 대한 간단 체크 가능
    return {"status": "ready"}

@router.get("/version")
async def version():
    return {
        "service": "koreaaimap-reco",
        "revision": os.getenv("GIT_SHA", "local"),
    }
```

### 9) `app/routers/reco.py` — 추천 API

```python
# app/routers/reco.py
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from ..schemas import RecoInput, RecoResponse, RecoItem
from ..services.citydata_client import fetch_citydata_by_code
from ..services.scoring import score_item
from datetime import datetime, timezone

# 샘플: 우선 5개 POI로 데모(운영에서는 120개 테이블/DB에서 로드)
DEFAULT_POIS = [
    ("POI104","어린이대공원"),
    ("POI088","광화문광장"),
    ("POI071","압구정로데오거리"),
    ("POI105","여의도한강공원"),
    ("POI068","성수카페거리"),
]

router = APIRouter(prefix="/reco", tags=["reco"])

@router.get("/suggestions", response_model=RecoResponse)
async def suggestions(
    lat: float, lng: float,
    when: str = "now",
    interests: Optional[List[str]] = Query(default=None),
    limit: int = Query(10, ge=1, le=50),
):
    # 1) 후보 장소(운영: DB/캐시에서 120개 로드 및 가까운 순 샘플링)
    candidates = DEFAULT_POIS

    # 2) 각 후보에 대해 citydata 조회 & 간단 feature 추출
    items: List[RecoItem] = []
    for code, name in candidates:
        data = await fetch_citydata_by_code(code)
        if not data:
            continue
        # 아래는 예시: 실제 구조에 맞게 파싱(혼잡도/이동시간/행사태그)
        categories = data.get("categories") or data  # 프록시 구현에 따라
        congestion_level = (categories.get("population", {}) or {}).get("level") if isinstance(categories, dict) else None
        travel_time_min = (categories.get("traffic", {}) or {}).get("eta_min") if isinstance(categories, dict) else None
        event_tags = (categories.get("events", {}) or {}).get("tags") if isinstance(categories, dict) else None

        score = score_item(congestion_level, travel_time_min, event_tags, interests)
        items.append(RecoItem(
            place_code=code, place_name=name,
            score=score,
            distance_m=None,               # TODO: 실제 거리 계산 로직(하버사인)
            congestion_level=congestion_level,
            travel_time_min=travel_time_min,
            event_summary=None
        ))

    # 3) 스코어 내림차순 정렬 + 상위 N
    items = sorted(items, key=lambda x: x.score, reverse=True)[:limit]
    return RecoResponse(
        count=len(items),
        items=items,
        updated_at=datetime.now(timezone.utc).isoformat()
    )
```

### 10) `app/main.py` — FastAPI 앱 조립

```python
# app/main.py
import logging, uvicorn
from fastapi import FastAPI
from .config import settings
from .instrumentation import AccessLogMiddleware
from .routers import health, reco

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(
    title="koreaaimap-reco",
    version="1.0.0",
    docs_url="/reco/docs",
    openapi_url="/reco/openapi.json",
    redoc_url=None,
)

# 미들웨어/라우터
app.add_middleware(AccessLogMiddleware)
app.include_router(health.router)
app.include_router(reco.router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, workers=1)
```

---

## 컨테이너/배포

### `requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
httpx==0.27.2
pydantic==2.9.1
pydantic-settings==2.5.2
```


### `Dockerfile`

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

## 시스템 의존 패키지(필요 시 curl/ca-certificates 등)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY gunicorn_conf.py ./

EXPOSE 9000
# prod: gunicorn + uvicorn worker (ALB 타임아웃 고려해 keepalive 설정)
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
```

### `gunicorn_conf.py`

```python
bind = "0.0.0.0:9000"
workers = 2                   # Fargate 1vCPU/3GB 기준
worker_class = "uvicorn.workers.UvicornWorker"
timeout = 30
graceful_timeout = 30
keepalive = 5
accesslog = "-"
errorlog = "-"
```

### `.env.example`

```
ENV=prod
PORT=9000
CITYDATA_BASE_URL=https://api.koreaaimap.com
CACHE_BACKEND=memory
CACHE_TTL_SECONDS=60
WEIGHT_CONGESTION=0.5
WEIGHT_TRAVELTIME=0.3
WEIGHT_EVENTFIT=0.2
```

### `Makefile`

```makefile
IMAGE?=koreaaimap-reco:latest

run:
\tuvicorn app.main:app --host 0.0.0.0 --port 9000 --reload

build:
\tdocker build --platform linux/amd64 -t $(IMAGE) .

serve:
\tdocker run --rm -p 9000:9000 --env-file .env $(IMAGE)
```

---


