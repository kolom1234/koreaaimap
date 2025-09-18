# koreaaimap
세종대학교 2025-2 창의학기제 프로젝트. 서울시 도시 데이터 기반 사용자 맞춤형 AI 장소 추천 서비스

혼잡한 도시 생활 속에서 출퇴근, 관광, 데이트, 여가생활과 같이 개인의 목적에 맞게 적합한 장소와 이동 경로를 찾는 것은 중요한 문제입니다. 실제로 잡코리아의 2024년 설문조사에서는 서울 거주 직장인들의 46.8프로가 출근길에 스트레스를 받는다고 답변했으며 이중 55.9프로가 사람이 너무 많은 만원 버스와 지하철 때문에 스트레스를 받는다고 답변했습니다. 하지만 현재 장소와 경로 추천 관련 플랫폼들은 단편적인 정보나 정적인 정보를 바탕으로 장소와 경로를 추천해줘 사람들의 이동 스트레스를 줄이는 데 한계를 보이고 있으며 실시간 동적 요소를 반영한 상황인지형 개인 맞춤 추천 기능의 필요성이 증대되고 있는 상황입니다.

한편, 서울시에서는 시민 편의 증진을 목표로 다양한 도시 데이터(실시간 혼잡도, 장소별 인구수, 문화 행사 정보, 교통 정보 등)를 공공 API 형태로 적극 개방하고 있고 이는 상황인지형 개인 맞춤 추천 시스템에 활용할 수 있습니다.

이에 따라 본 프로젝트에서는 서울시의 다양한 도시 데이터를 융합하여 사용자 개개인의 목적과 취향을 정확히 반영한 맞춤형 장소 및 경로 추천 서비스를 개발하고 사람들의 이동 스트레스 문제를 해결하고자 합니다.

# 활용 데이터
## 서울 실시간 도시 데이터
<img width="1276" height="1104" alt="image" src="https://github.com/user-attachments/assets/24fdda21-f9dd-4641-ab16-2e8851d3fdf1" />


# RECO(FastAPI) 골격


## 1) RECO 서비스 목표/역할(요약)

* **포트/경로:** Fargate(포트 **9000**) + ALB 경로 규칙 **`/reco/* → tg-reco`**, 헬스 `GET /health=200` &#x20;
* **데이터 소스:** 프런트가 직접 서울시 API를 두드리지 않도록, \*\*내부 API 게이트웨이(Spring, `/api/v1/citydata`)\*\*를 1차 우선 사용. 키 보호/캐싱/로깅 일원화. (필요 시 백업 경로로 서울시 OpenAPI 직접 호출도 지원)&#x20;
* **캐시 정책(초기):** 인구 등 핵심은 **5분 단위 갱신**, 사용자는 15분 지연을 감안(통신사 보정/전수화) → **프록시/RECO 레이어 TTL 60초 권장**. 수집 배치 시엔 700ms 레이트 리밋 권장.  &#x20;
* **엔진:** 초기가동은 규칙기반(거리·혼잡도·날씨 가중)으로, 이후 ONNX/시계열 예측을 붙일 수 있게 **서비스 계층 분리**. (문서의 모델링 로드맵과 정합)&#x20;

---

## 2) 디렉터리 골격

```
reco/
  app/
    __init__.py
    main.py
    api/__init__.py
    api/routes.py
    core/__init__.py
    core/config.py
    models/__init__.py
    models/schemas.py
    services/__init__.py
    services/seoul_client.py
    services/redis_cache.py
    services/recommend.py
  requirements.txt
  gunicorn_conf.py
  Dockerfile
  README.md
```

**환경변수(초기값/권장):**

| 이름                  | 용도                 | 예시                             |
| ------------------- | ------------------ | ------------------------------ |
| `API_BASE_URL`      | 내부 게이트웨이 주소        | `https://api.koreaaimap.com`   |
| `SEOUL_GENERAL_KEY` | (백업) 서울시 OpenAPI 키 | `…`                            |
| `REDIS_URL`         | 캐시(선택)             | `rediss://:token@host:6379/0`  |
| `CACHE_TTL_SECONDS` | 기본 TTL             | `60`                           |
| `RATE_LIMIT_MS`     | 외부호출 간격            | `700`                          |

---

## 3) 최소 동작 FastAPI 코드

#### `app/core/config.py`

```python
from pydantic import BaseModel
import os

class Settings(BaseModel):
    api_base_url: str = os.getenv("API_BASE_URL", "https://api.koreaaimap.com")
    seoul_key: str | None = os.getenv("SEOUL_GENERAL_KEY")
    redis_url: str | None = os.getenv("REDIS_URL")
    cache_ttl: int = int(os.getenv("CACHE_TTL_SECONDS", "60"))
    rate_limit_ms: int = int(os.getenv("RATE_LIMIT_MS", "700"))
    service_name: str = "koreaaimap-reco"
    version: str = "0.1.0"

settings = Settings()
```

#### `app/models/schemas.py`

```python
from typing import List, Optional, Literal
from pydantic import BaseModel

class Place(BaseModel):
    code: str
    name: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None

class RecommendRequest(BaseModel):
    user_lat: float
    user_lng: float
    purpose: Literal["date","leisure","study","walk","shopping","family","any"] = "any"
    max_results: int = 5
    # 후보 POI 코드 목록(초기: 외부에서 주입; 추후 120장소 테이블로 대체)
    candidates: Optional[List[Place]] = None

class RecommendItem(BaseModel):
    place: Place
    score: float
    features: dict

class RecommendResponse(BaseModel):
    count: int
    items: List[RecommendItem]
    source: str = "Seoul Open Data via internal API"
```

#### `app/services/redis_cache.py`

```python
import time
from typing import Any, Optional
from . import typing as _t  # no-op to avoid linter issues
import json, threading

try:
    import redis
except ImportError:
    redis = None

class Cache:
    def __init__(self, url: Optional[str], default_ttl: int = 60):
        self.default_ttl = default_ttl
        if url and redis:
            self.client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2)
            self.local = None
        else:
            self.client = None
            self.local = {}
            self.lock = threading.Lock()

    def get(self, key: str) -> Optional[dict]:
        if self.client:
            raw = self.client.get(key)
            return json.loads(raw) if raw else None
        with self.lock:
            v = self.local.get(key)
            if not v: return None
            if v["exp"] < time.time():
                self.local.pop(key, None); return None
            return v["val"]

    def set(self, key: str, value: dict, ttl: Optional[int] = None):
        ttl = ttl or self.default_ttl
        if self.client:
            self.client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
        else:
            with self.lock:
                self.local[key] = {"val": value, "exp": time.time()+ttl}
```

#### `app/services/seoul_client.py`

```python
import httpx, time
from ..core.config import settings

class SeoulClient:
    def __init__(self):
        self.api_base = settings.api_base_url.rstrip("/")
        self.key = settings.seoul_key
        self.rate_ms = settings.rate_limit_ms
        self._last = 0.0
        self.http = httpx.Client(timeout=8, headers={"User-Agent":"koreaaimap-reco/0.1"})

    def _rl(self):
        now = time.time()
        wait = (self._last + self.rate_ms/1000.0) - now
        if wait > 0: time.sleep(wait)
        self._last = time.time()

    def citydata_via_internal(self, *, code: str | None = None, place: str | None = None):
        """권장 경로: 내부 Spring API 프록시 사용"""
        assert (code or place) and not (code and place)
        params = {"code": code} if code else {"place": place}
        url = f"{self.api_base}/api/v1/citydata"
        self._rl()
        r = self.http.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def citydata_direct(self, *, code: str):
        """백업 경로: 서울시 OpenAPI 직접 호출(키 필요, 1회 1장소)"""
        if not self.key:
            raise RuntimeError("SEOUL_GENERAL_KEY is not set.")
        url = f"http://openapi.seoul.go.kr:8088/{self.key}/json/citydata/1/5/{code}"
        self._rl()
        r = self.http.get(url)
        r.raise_for_status()
        return r.json()
```

#### `app/services/recommend.py`

```python
from math import radians, sin, cos, asin, sqrt
from typing import List, Dict
from ..models.schemas import Place, RecommendItem

def _haversine(lat1, lon1, lat2, lon2):
    R=6371.0
    dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
    a=sin(dlat/2)**2+cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2*R*asin(sqrt(a))  # km

def score_place(*, user_lat, user_lng, place: Place, citydata: dict, purpose: str) -> RecommendItem:
    # 간단한 특징 추출 (초기 규칙 기반)
    categories = citydata.get("categories", {})
    ppl = categories.get("population", {})
    crowd_level = ppl.get("congestionLevel") or ppl.get("level")  # (여유/보통/약간 붐빔/붐빔) -> 매핑
    level_map = {"여유":0.1, "보통":0.4, "약간 붐빔":0.7, "붐빔":1.0}
    crowd = level_map.get(str(crowd_level), 0.5)

    # 거리(가까울수록 가점)
    dist_km = None
    if place.lat and place.lng:
        dist_km = _haversine(user_lat, user_lng, place.lat, place.lng)
    dist_term = 0.0 if dist_km is None else max(0.0, 1.0 - min(dist_km/10.0, 1.0))  # 0~1

    # 목적 가중(예: 데이트/실내 선호 등은 이후 확장)
    purpose_bias = 0.0  # TODO: 날씨/상권 업종/실내·실외 신호 반영 예정

    # 낮을수록 좋은 혼잡도(crowd)는 음수 가중
    score = 0.60*dist_term + 0.35*(1.0 - crowd) + 0.05*purpose_bias
    feats = {"dist_km": dist_km, "crowd_raw": crowd_level, "crowd_norm": crowd, "dist_term": dist_term}
    return RecommendItem(place=place, score=round(score,4), features=feats)
```

#### `app/api/routes.py`

```python
from fastapi import APIRouter, HTTPException, Query
from ..core.config import settings
from ..services.seoul_client import SeoulClient
from ..services.redis_cache import Cache
from ..models.schemas import RecommendRequest, RecommendResponse, RecommendItem, Place

router = APIRouter()
client = SeoulClient()
cache = Cache(settings.redis_url, default_ttl=settings.cache_ttl)

@router.get("/health")
def health():
    return {"status":"ok","service":settings.service_name,"version":settings.version}

@router.get("/v1/ping")
def ping():
    return {"pong": True}

@router.get("/v1/citydata")
def citydata(code: str | None = Query(None), place: str | None = Query(None)):
    if not ((code or place) and not (code and place)):
        raise HTTPException(400, "Provide exactly one of code or place")
    key = f"citydata:{code or place}"
    cached = cache.get(key)
    if cached: return cached
    try:
        data = client.citydata_via_internal(code=code, place=place)
    except Exception:
        if code:
            data = client.citydata_direct(code=code)  # fallback
        else:
            raise
    cache.set(key, data)
    return data

@router.post("/v1/recommend", response_model=RecommendResponse)
def recommend(payload: RecommendRequest):
    # 후보가 없으면 일단 대표 샘플로 동작(POI104=어린이대공원)
    candidates = payload.candidates or [Place(code="POI104", name="어린이대공원")]
    items: list[RecommendItem] = []
    for p in candidates:
        try:
            data = citydata(code=p.code)  # 로컬 핸들러 호출(캐시 포함)
            # 위치 좌표가 있으면 점수 정확도↑ (초기는 옵션)
            # p.lat, p.lng 를 향후 120장소 메타테이블에서 채움
            from ..services.recommend import score_place
            items.append(score_place(user_lat=payload.user_lat, user_lng=payload.user_lng,
                                     place=p, citydata=data, purpose=payload.purpose))
        except Exception as e:
            # 한 항목 실패해도 나머지는 진행
            continue
    items.sort(key=lambda x: x.score, reverse=True)
    items = items[: payload.max_results]
    return RecommendResponse(count=len(items), items=items)
```

#### `app/main.py`

```python
from fastapi import FastAPI
from .api.routes import router

app = FastAPI(title="KoreaAIMap RECO", version="0.1.0")
app.include_router(router, prefix="/reco")
```

#### `requirements.txt`

```
fastapi==0.115.0
uvicorn[standard]==0.30.5
httpx==0.27.2
redis==5.0.8
```

#### `gunicorn_conf.py`

```python
workers = 2
worker_class = "uvicorn.workers.UvicornWorker"
bind = "0.0.0.0:9000"
timeout = 30
```

#### `Dockerfile`

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY gunicorn_conf.py .

EXPOSE 9000
CMD ["gunicorn", "-c", "gunicorn_conf.py", "app.main:app"]
```

---

## 4) ALB / ECS 연결값(운영 기준)

* **Target Group (tg-reco):** Port **9000**, Health check **`/health`**, 200 OK 기대.
* **Listener Rule:** HTTPS(443) **우선순위 1: `/reco/*` → tg-reco** (+ 정확일치 `/reco`도 추가 권장)&#x20;
* **Task Definition(예시):** `koreaaimap-reco:1` / Linux x86\_64 / awsvpc / **CPU 1024 / MEM 2048** / 컨테이너 포트 9000 / **로그: /ecs/koreaaimap-reco** / 헬스체크(CMD-SHELL):

  ```
  curl -f http://localhost:9000/reco/health || exit 1
  ```

  (Interval 30s / Timeout 5s / Retries 3 / StartPeriod 45s) &#x20;
* **Service:** `koreaaimap-reco-svc` / 프라이빗 2AZ / Public IP Off / SG(인바운드 9000 from alb-sg). &#x20;

> 주의사항: `/reco/*`만 있고 \*\*정확일치 `/reco`\*\*가 없으면 루트 접근에서 404가 날 수 있음. 

---

## 5) CI/CD (GitHub Actions) — `/.github/workflows/deploy-reco.yml`

```yaml
name: Deploy RECO (FastAPI)

on:
  push:
    paths:
      - "reco/**"
      - ".github/workflows/deploy-reco.yml"
    branches: [ "main" ]

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      id-token: write
      contents: read
    env:
      AWS_REGION: ap-northeast-2
      ECR_REPO: ${{ secrets.ECR_RECO_REPO }}   # e.g. 3304....dkr.ecr.ap-northeast-2.amazonaws.com/koreaaimap-reco
      CLUSTER: koreaaimap
      SERVICE: koreaaimap-reco-svc
      IMAGE_TAG: ${{ github.sha }}
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build & Push image
        working-directory: ./reco
        run: |
          docker build --platform linux/amd64 -t "$ECR_REPO:latest" -t "$ECR_REPO:$IMAGE_TAG" .
          docker push "$ECR_REPO:latest"
          docker push "$ECR_REPO:$IMAGE_TAG"

      - name: Update ECS Service (force new deployment)
        run: |
          aws ecs update-service \
            --cluster "$CLUSTER" \
            --service "$SERVICE" \
            --force-new-deployment \
            --region "$AWS_REGION"
```

> OIDC/레포 시크릿/역할 구조는 스캐폴드 가이드의 CI/CD 절과 동일하게 맞춰져있음. `AWS_ROLE_ARN`, `ECR_RECO_REPO`만 등록하면 됨.&#x20;

---

## 6) Postman & OpenAPI (RECO 추가)


#### Postman 아이템(추가 예)

* **`RECO — /reco/health`** → 200/`status=ok`
* **`RECO — /reco/v1/recommend`** (POST)
  Body(raw/JSON):

  ```json
  {
    "user_lat": 37.549,
    "user_lng": 127.074,
    "purpose": "date",
    "max_results": 3,
    "candidates": [{"code":"POI104","name":"어린이대공원"}]
  }
  ```

#### RECO OpenAPI 스니펫(병합용)

```yaml
paths:
  /reco/health:
    get:
      summary: Health
      responses: { '200': { description: OK } }

  /reco/v1/recommend:
    post:
      summary: Recommend places
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/RecommendRequest'
      responses:
        '200':
          description: OK
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/RecommendResponse'

components:
  schemas:
    Place:
      type: object
      properties:
        code: { type: string }
        name: { type: string }
        lat: { type: number }
        lng: { type: number }
    RecommendRequest:
      type: object
      required: [user_lat,user_lng]
      properties:
        user_lat: { type: number }
        user_lng: { type: number }
        purpose: { type: string, enum: [date,leisure,study,walk,shopping,family,any] }
        max_results: { type: integer, default: 5 }
        candidates:
          type: array
          items: { $ref: '#/components/schemas/Place' }
    RecommendResponse:
      type: object
      properties:
        count: { type: integer }
        items:
          type: array
          items:
            type: object
            properties:
              place: { $ref: '#/components/schemas/Place' }
              score: { type: number }
              features: { type: object }
        source: { type: string }
```


---

## 7) 운영 체크리스트(초기 런칭용)

1. **ECR**에 `koreaaimap-reco:latest` 푸시
2. **Task Definition** 등록(포트 9000 / 헬스커맨드 / 로그 그룹) → **Service 생성**(Private Subnets, Public IP Off)
3. **ALB 규칙** `/reco/* → tg-reco` + **정확일치 `/reco` 추가** → 헬스 200 확인 &#x20;
4. **DNS**에서 `reco.koreaaimap.com → ALB`(선택), 운영 시 Cloudflare **Full(Strict)** + SG를 CF IP로 축소(옵션 WAF) &#x20;
5. **캐시/레이트리밋**: TTL 60s, 호출 간격 700ms(배치/백엔드)  &#x20;
6. **출처·주요 주의표기**: 공공누리 출처표시 / 인구·상권 데이터의 처리·지연·주의사항 UI/문서 고정 표기  &#x20;

---

## 8) 로컬 스모크 & 운영 점검

```bash
# 로컬
uvicorn app.main:app --host 0.0.0.0 --port 9000
curl -s http://127.0.0.1:9000/reco/health
curl -s http://127.0.0.1:9000/reco/v1/citydata?code=POI104 | jq '.place,.updatedAt'
curl -s -X POST http://127.0.0.1:9000/reco/v1/recommend \
  -H "content-type: application/json" \
  -d '{"user_lat":37.55,"user_lng":127.07,"candidates":[{"code":"POI104","name":"어린이대공원"}]}'

# 운영 (ALB 뒤)
curl -I https://api.koreaaimap.com/reco/health
curl -s https://api.koreaaimap.com/reco/v1/citydata?code=POI104 | jq '.place,.updatedAt'
```

---

## 9) 이후 확장(간단 로드맵)

* **120장소 메타테이블**(코드/이름/좌표/카테고리) 주입 → `Place.lat/lng`로 거리 정확도↑.
* **상황인지 가중치**: 날씨(실내/실외), 시간대, 상권·교통 신호 반영. (매뉴얼 정의 따라 혼잡도·상권 범주 사용)&#x20;
* **예측/ML**: 12시간 예측 신호(Seq2Seq/Informer 등) 가중 반영 후 ONNX 서빙. (학습·성능 목표는 설계서와 일치)&#x20;

---

### 문서/설계와의 정합성 체크

* **아키텍처·경로 규칙·헬스**: ALB 443, `/reco/* → reco-service`, 헬스 `/health`. `/reco` 규칙 생각하기.&#x20;
* **내부 프록시 표준화/TTL 60s/키 보호**: 프런트는 내부 게이트웨이만 호출, 캐시/정규화 응답.&#x20;
* **수집 스크립트 베이스라인(0.7s 레이트리밋)**: 배치·수집 시 동일 규칙 고려.&#x20;
* **데이터 갱신·주의·혼잡도 정의**: 5분 단위 수집/약 15분 처리 지연, 혼잡도 산정/주의 표기.&#x20;

---

