// src/index.js

const DEFAULT_TTL = 2700;           // 엣지 캐시 TTL
const INFER_INPUT_MAX = 4000;     // /infer 입력 길이 제한

function corsHeaders(origin) {
	return {
		"Access-Control-Allow-Origin": origin || "*",
		"Access-Control-Allow-Methods": "GET,POST,OPTIONS", // POST 추가
		"Access-Control-Allow-Headers":
			"Content-Type, Authorization, X-Requested-With, X-Edge-Signature", // 서명 헤더 추가
		"Access-Control-Allow-Credentials": "true",
		"Vary": "Origin"
	};
}

// --- TOPIS XML helpers (put near top of file) ---
function pick(xml, tag) {
	const m = xml.match(new RegExp(`<${tag}>([^<]*)</${tag}>`, 'i'));
	return m ? m[1].trim() : null;
}
function toNum(v) {
	const n = Number(v);
	return Number.isFinite(n) ? n : null;
}
async function fetchTopisSpeedXML(linkId, key) {
	// TOPIS 실시간 소통: 링크 단건 조회(XML 고정 케이스 대응)
	// 서비스명은 TrafficInfo(실시간 링크 소통), /1/1/{linkId} 패턴
	const url = `http://openapi.seoul.go.kr:8088/${key}/xml/TrafficInfo/1/1/${encodeURIComponent(linkId)}`;
	const res = await fetch(url, { cf: { cacheTtl: 0 }, headers: { "User-Agent": "koreaaimap-edge/1.0" } });
	const text = await res.text();
	if (!res.ok) throw new Error(`TOPIS HTTP ${res.status}`);
	// 정상이면 <row> 하위에 LINK_ID, SPEED, IDX 등이 존재
	const speed = toNum(pick(text, 'SPEED'));
	const idx   = toNum(pick(text, 'IDX'));
	const lid   = pick(text, 'LINK_ID') || String(linkId);
	if (speed == null) throw new Error('no SPEED in response');
	return { linkId: lid, speed, idx };
}

// --------- citydata helpers + Edge Cache ----------
const EDGE_CACHE_TTL = 1800; // 30분 (Edge Cache, Cron 주기와 일치)
const KV_CACHE_TTL = 3600;   // 60분 (KV Store, 안정성 확보)

// 히트맵 접근용 간단한 토큰 생성
function generateHeatmapToken(code, timestamp) {
	const secret = "heatmap-secret-2025"; // 실제로는 환경 변수 사용 권장
	const data = `${code}-${timestamp}`;
	// 간단한 해시 (실제로는 crypto API 사용 권장)
	let hash = 0;
	for (let i = 0; i < data.length; i++) {
		hash = ((hash << 5) - hash) + data.charCodeAt(i);
		hash = hash & hash;
	}
	return Math.abs(hash).toString(36);
}

function verifyHeatmapToken(code, timestamp, token) {
	const expectedToken = generateHeatmapToken(code, timestamp);
	const now = Date.now();
	const tokenAge = now - parseInt(timestamp);
	// 토큰 유효 기간: 1시간
	return token === expectedToken && tokenAge < 3600000;
}

async function handleHeatmapProxy(req, env) {
	const url = new URL(req.url);
	const code = url.searchParams.get("code");
	const timestamp = url.searchParams.get("t");
	const token = url.searchParams.get("token");

	if (!code) {
		return new Response("Missing code parameter", { status: 400 });
	}

	// 토큰 검증 (선택적 - 개발 중에는 비활성화 가능)
	if (timestamp && token) {
		if (!verifyHeatmapToken(code, timestamp, token)) {
			return new Response("Invalid or expired token", { status: 403 });
		}
	}

	// 히트맵 파일 경로 생성
	const heatmapPath = `/heatmaps/${code}.png`;
	const heatmapReq = new Request(`http://assets.local${heatmapPath}`);

	try {
		const heatmapRes = await env.ASSETS.fetch(heatmapReq);
		if (!heatmapRes.ok) {
			return new Response("Heatmap not found", { status: 404 });
		}

		// 캐시 헤더 추가
		const response = new Response(heatmapRes.body, {
			status: 200,
			headers: {
				"Content-Type": "image/png",
				"Cache-Control": "public, max-age=86400", // 24시간 캐시
				"X-Heatmap-Code": code
			}
		});

		return response;
	} catch (err) {
		console.error(`[HEATMAP] Error loading ${code}:`, err);
		return new Response("Internal error", { status: 500 });
	}
}

const lc = s => String(s || "").trim().toLowerCase();

function deep(obj, ...keys) {
	let cur = obj;
	for (const k of keys) {
		if (!cur) return null;
		const K = Object.keys(cur).find(x => x.toLowerCase() === String(k).toLowerCase());
		cur = K ? cur[K] : null;
	}
	return cur;
}

// 서울 citydata 응답에서 우리가 쓰는 파트만 뽑아보는 보조 (참고용: UI는 원본 categories도 전달)
function extractCityStats(payload) {
	// ★ CITYDATA 래퍼가 있으면 그 안에서 탐색
	const citydata = payload.CITYDATA || payload.citydata || payload;

	// LIVE_PPLTN_STTS, FCST_PPLTN_STTS, ROAD_TRAFFIC_STTS 등 위치가 종종 섞여 있어서 느슨하게 탐색
	const liveRaw = deep(citydata, "LIVE_PPLTN_STTS") || deep(citydata, "live_ppltn_stts") || deep(citydata, "LivePpltnStts");
	const fcstRaw = deep(citydata, "FCST_PPLTN_STTS") || deep(citydata, "fcst_ppltn_stts") || deep(citydata, "FcstPpltnStts");
	const roadRaw = deep(citydata, "ROAD_TRAFFIC_STTS") || deep(citydata, "road_traffic_stts") || deep(citydata, "RoadTrafficStts");

	// ★ LIVE_PPLTN_STTS가 배열인 경우 첫 번째 요소 추출
	const live = Array.isArray(liveRaw) ? liveRaw[0] : liveRaw;
	const fcst = Array.isArray(fcstRaw) ? fcstRaw[0] : fcstRaw;
	const road = Array.isArray(roadRaw) ? roadRaw[0] : roadRaw;

	// 일부 키는 상/하위 대소문자 혼합이므로 다시 느슨하게 접근
	const level = live?.AREA_CONGEST_LVL ?? live?.AREA_CONGEST_MSG ?? live?.AREA_CONGEST_LVL_CD ?? null;
	const ppl   = live?.PPLTN_CNT ?? live?.TOT_PPLTN ?? live?.PPLTN_NUM ?? null;
	const idx   = live?.LIVE_PPLTN_INDEX ?? live?.AREA_PPLTN_INDEX ?? null;
	const time  = live?.LIVE_PPLTN_TIME ?? live?.LIVE_PPLTN_DT ?? live?.DATA_TIME ?? null;

	const roadSpd = road?.ROAD_TRAFFIC_SPD ?? null;

	return {
		level: level ?? null,
		population: (ppl != null ? Number(String(ppl).replace(/,/g, "")) : null),
		index: (idx != null ? Number(String(idx).replace(/,/g, "")) : null),
		roadSpeed: (roadSpd != null ? Number(String(roadSpd).replace(/,/g, "")) : null),
		dataTime: time ?? null,
		fcst: fcst || null,
	};
}

const CITYDATA_CACHE_BASE = "https://cache.koreaaimap/internal/citydata";

function cityCacheRequest(key) {
	return new Request(`${CITYDATA_CACHE_BASE}/${encodeURIComponent(key)}`, { method: "GET" });
}

async function cacheGetCity(env, key) {
	if (!key) return null;
	try {
		const value = await env.CITYDATA_KV.get(key, { type: "json" });
		if (value) {
			console.log(`[KV] ✓ HIT: ${key}`);
		} else {
			console.log(`[KV] ✗ MISS: ${key}`);
		}
		return value;
	} catch (err) {
		console.error(`[KV] ❌ READ ERROR: ${key}`, err.message);
		return null;
	}
}
async function cachePutCity(env, key, value) {
	if (!key) return;
	try {
		await env.CITYDATA_KV.put(
			key,
			JSON.stringify(value),
			{ expirationTtl: KV_CACHE_TTL }  // 60분 자동 삭제
		);
		console.log(`[KV] ✓ WRITE: ${key} (TTL: ${KV_CACHE_TTL}s)`);
	} catch (err) {
		console.error(`[KV] ❌ WRITE ERROR: ${key}`, err.message);
	}
}

// Edge Cache 헬퍼 함수
async function cacheGetEdge(key) {
	if (!key) return null;
	try {
		const cacheKey = new Request(`${CITYDATA_CACHE_BASE}/${encodeURIComponent(key)}`, { method: "GET" });
		const cached = await caches.default.match(cacheKey);
		if (cached) {
			const data = await cached.json();
			console.log(`[EDGE] ✓ HIT: ${key}`);
			return data;
		}
		console.log(`[EDGE] ✗ MISS: ${key}`);
		return null;
	} catch (err) {
		console.error(`[EDGE] ❌ READ ERROR: ${key}`, err.message);
		return null;
	}
}

async function cachePutEdge(key, value) {
	if (!key) return;
	try {
		const cacheKey = new Request(`${CITYDATA_CACHE_BASE}/${encodeURIComponent(key)}`, { method: "GET" });
		const response = new Response(JSON.stringify(value), {
			headers: {
				"Content-Type": "application/json; charset=utf-8",
				"Cache-Control": `public, max-age=${EDGE_CACHE_TTL}, s-maxage=${EDGE_CACHE_TTL}`
			}
		});
		await caches.default.put(cacheKey, response);
		console.log(`[EDGE] ✓ WRITE: ${key} (TTL: ${EDGE_CACHE_TTL}s)`);
	} catch (err) {
		console.error(`[EDGE] ❌ WRITE ERROR: ${key}`, err.message);
	}
}

// ---- CORS: single-origin echo + allowlist + Vary: Origin ----
const ALLOWLIST = [
	"https://www.koreaaimap.com",
	"https://koreaaimap.com",
	// 필요 시 프리뷰 도메인 추가:
	"https://koreaaimap.pages.dev",
	"http://localhost:5173",
	"http://127.0.0.1:5173"
];

function corsHeadersFor(req) {
	const origin = req.headers.get("Origin");
	if (origin && ALLOWLIST.includes(origin)) {
		return {
			"Access-Control-Allow-Origin": origin,
			"Access-Control-Allow-Methods": "GET,POST,OPTIONS",
			"Access-Control-Allow-Headers":
				"Content-Type, Authorization, X-Requested-With, X-Edge-Signature", // ← 보강
			// "Access-Control-Allow-Credentials": "true",
			"Vary": "Origin"
		};
	}
	return {};
}


function preflight(req) {
	return new Response(null, {
		status: 204,
		headers: {
			...corsHeadersFor(req),
			"Access-Control-Max-Age": "86400"
		}
	});
}

// 기존/업스트림 응답에서 CORS 관련 헤더를 지우고(중복 방지) 새 CORS 헤더를 붙임
function withCors(req, upstreamRes, extra = {}) {
	const h = new Headers(upstreamRes ? upstreamRes.headers : {});
	["Access-Control-Allow-Origin", "Access-Control-Allow-Methods",
		"Access-Control-Allow-Headers", "Access-Control-Allow-Credentials",
		"Vary"].forEach(k => h.delete(k));

	const ch = corsHeadersFor(req);
	for (const [k, v] of Object.entries(ch)) h.set(k, v);
	for (const [k, v] of Object.entries(extra)) h.set(k, v);

	return new Response(upstreamRes ? upstreamRes.body : null, {
		status: upstreamRes ? upstreamRes.status : 200,
		headers: h
	});
}


function isHexKey(s) {
	return typeof s === "string" && /^[0-9a-fA-F]+$/.test(s) && (s.length % 2 === 0);
}
function hexToBytes(hex) {
	const len = hex.length / 2;
	const out = new Uint8Array(len);
	for (let i = 0; i < len; i++) out[i] = parseInt(hex.substr(i * 2, 2), 16);
	return out;
}


function okJSON(obj, req, extra = {}) {
	const res = new Response(JSON.stringify(obj), {
		status: 200,
		headers: {
			"Content-Type": "application/json; charset=utf-8",
			// SWR: 15분 fresh, 24시간 stale-while-revalidate
			"Cache-Control": `public, max-age=${DEFAULT_TTL}, stale-while-revalidate=86400`
		}
	});
	return withCors(req, res, extra);
}

function errJSON(status, code, message, req, meta = {}) {
	const res = new Response(JSON.stringify({ status, error: { code, message }, meta }), {
		status,
		headers: { "Content-Type": "application/json; charset=utf-8" }
	});
	return withCors(req, res);
}


function buildCacheKey(req) {
	const u = new URL(req.url);
	const code = u.searchParams.get("code") || "";
	const place = u.searchParams.get("place") || "";
	return new Request(`${u.origin}${u.pathname}?code=${code}&place=${place}`, { method: "GET" });
}

// 핸들러 구현
async function handleRoute(req, env, origin) {
	const u = new URL(req.url);
	const from = u.searchParams.get("from"); // "lat,lng"
	const to   = u.searchParams.get("to");
	const alternatives = u.searchParams.get("alternatives") === "true";
	const alternativesCount = parseInt(u.searchParams.get("alternatives_count") || "3", 10);

	if (!from || !to) return json({ error: "missing from/to" }, 400, origin);

	const [fy, fx] = from.split(",").map(Number); // lat, lng
	const [ty, tx] = to.split(",").map(Number);
	if (![fy,fx,ty,tx].every(Number.isFinite)) return json({ error: "bad coords" }, 400, origin);

	// OSRM 데모(지속성 낮음) — 프로덕션은 자체 OSRM/유료 API 고려
	// alternatives 파라미터: true로 설정하면 대체 경로를 찾음
	let url = `https://router.project-osrm.org/route/v1/driving/${fx},${fy};${tx},${ty}?overview=full&geometries=geojson`;

	// OSRM alternatives 파라미터는 boolean 또는 숫자 지원
	// alternatives=true 또는 alternatives=N (N개의 대체 경로)
	if (alternatives) {
		url += `&alternatives=true`;
		// steps=true를 추가하면 더 상세한 경로 정보 제공
		url += `&steps=true`;
	}

	const r = await fetch(url, { headers: { "accept": "application/json" }, cf: { cacheTtl: 300, cacheEverything: true }});
	const data = await r.json();

	// OSRM이 충분한 경로를 반환하지 않으면, waypoint를 추가해서 다른 경로 생성
	if (data.routes && data.routes.length < 3 && data.routes.length > 0) {
		const neededRoutes = 3 - data.routes.length;
		const additionalRoutes = [];

		// 출발지와 도착지 사이의 중간 지점들을 계산
		const midLat = (fy + ty) / 2;
		const midLng = (fx + tx) / 2;

		// 여러 방향으로 오프셋을 주어 다른 경로 생성
		const offsets = [
			{ lat: 0.01, lng: 0.01 },   // 북동쪽 우회
			{ lat: -0.01, lng: 0.01 },  // 남동쪽 우회
			{ lat: 0.01, lng: -0.01 },  // 북서쪽 우회
			{ lat: -0.01, lng: -0.01 }, // 남서쪽 우회
		];

		for (let i = 0; i < Math.min(neededRoutes, offsets.length); i++) {
			const offset = offsets[i];
			const waypointLng = midLng + offset.lng;
			const waypointLat = midLat + offset.lat;

			// waypoint를 포함한 경로 요청
			const waypointUrl = `https://router.project-osrm.org/route/v1/driving/${fx},${fy};${waypointLng},${waypointLat};${tx},${ty}?overview=full&geometries=geojson`;

			try {
				const waypointRes = await fetch(waypointUrl, {
					headers: { "accept": "application/json" },
					cf: { cacheTtl: 300, cacheEverything: true }
				});
				const waypointData = await waypointRes.json();

				if (waypointData.routes && waypointData.routes.length > 0) {
					additionalRoutes.push(waypointData.routes[0]);
				}
			} catch (err) {
				console.error('Failed to fetch waypoint route:', err);
			}
		}

		// 추가 경로들을 메인 응답에 병합
		if (additionalRoutes.length > 0) {
			data.routes = [...data.routes, ...additionalRoutes];
		}
	}

	return json(data, r.status, origin);
}

// 네이버 Direction5 API 핸들러
async function handleNaverDirection5(req, env, origin) {
	const u = new URL(req.url);
	const from = u.searchParams.get("from"); // "lat,lng"
	const to   = u.searchParams.get("to");
	const option = u.searchParams.get("option") || "trafast"; // trafast, tracomfort, traoptimal

	if (!from || !to) return json({ error: "missing from/to" }, 400, origin);

	const [fy, fx] = from.split(",").map(Number); // lat, lng
	const [ty, tx] = to.split(",").map(Number);

	if (![fy,fx,ty,tx].every(Number.isFinite)) return json({ error: "bad coords" }, 400, origin);

	const KEY_ID = env.NCP_KEY_ID || env.NAVER_CLIENT_ID;
	const KEY    = env.NCP_KEY    || env.NAVER_CLIENT_SECRET;

	if (!KEY_ID || !KEY) {
		return json({ error: "NAVER credentials missing" }, 500, origin);
	}

	// 네이버 Direction5 API는 lng,lat 순서 사용
	const start = `${fx},${fy}`;
	const goal = `${tx},${ty}`;

	const apiUrl = `https://maps.apigw.ntruss.com/map-direction/v1/driving?start=${encodeURIComponent(start)}&goal=${encodeURIComponent(goal)}&option=${option}`;

	try {
		const r = await fetch(apiUrl, {
			headers: {
				"X-NCP-APIGW-API-KEY-ID": KEY_ID,
				"X-NCP-APIGW-API-KEY": KEY,
				"Accept": "application/json"
			},
			cf: { cacheTtl: 300, cacheEverything: true }
		});

		const data = await r.json();

		if (!r.ok) {
			console.error("Naver Direction5 API error:", data);
			return json({ error: "Naver Direction5 API error", details: data }, r.status, origin);
		}
		return json(data, r.status, origin);
	} catch (error) {
		console.error("Naver Direction5 fetch error:", error);
		return json({ error: "Failed to fetch Naver Direction5", details: error.message }, 500, origin);
	}
}

// 서울시 지하철역 정보 API 핸들러
async function handleSeoulSubwayStations(req, env, origin) {

	if (!env.SEOUL_GENERAL_KEY) {
		return json({ error: "SEOUL_GENERAL_KEY missing" }, 500, origin);
	}

	// 서울시 지하철역 좌표 정보 API (subwayStationMaster)
	// 최대 1000개까지 한 번에 요청 가능
	const apiUrl = `http://openapi.seoul.go.kr:8088/${env.SEOUL_GENERAL_KEY}/json/subwayStationMaster/1/1000/`;

	try {
		const r = await fetch(apiUrl, {
			cf: { cacheTtl: 3600, cacheEverything: true } // 1시간 캐시 (정적 데이터)
		});

		if (!r.ok) {
			const text = await r.text();
			console.error("Seoul Subway API error:", r.status, text);
			return json({ error: "Seoul Subway API error", status: r.status }, r.status, origin);
		}

		const data = await r.json();

		// API 응답 구조: { subwayStationMaster: { list_total_count, RESULT, row: [...] } }
		if (data.subwayStationMaster?.RESULT?.CODE !== "INFO-000") {
			console.error("Seoul Subway API error:", data.subwayStationMaster?.RESULT);
			return json({ error: "Seoul Subway API error", details: data.subwayStationMaster?.RESULT }, 400, origin);
		}

		const stations = data.subwayStationMaster?.row || [];

		return json({
			success: true,
			count: stations.length,
			stations: stations
		}, 200, origin);

	} catch (error) {
		console.error("Seoul Subway API fetch error:", error);
		return json({ error: "Failed to fetch Seoul Subway data", details: error.message }, 500, origin);
	}
}

// 서울시 문화행사 정보 API 핸들러
async function handleSeoulCulturalEvents(req, env, origin) {
	// CORS preflight 처리
	if (req.method === "OPTIONS") return preflight(req);

	if (!env.SEOUL_GENERAL_KEY) {
		return json({ error: "SEOUL_GENERAL_KEY missing" }, 500, origin);
	}

	const url = new URL(req.url);
	const start = url.searchParams.get("start") || "1";
	const end = url.searchParams.get("end") || "1000";

	// 서울시 문화행사 정보 API (culturalEventInfo)
	const apiUrl = `http://openapi.seoul.go.kr:8088/${env.SEOUL_GENERAL_KEY}/json/culturalEventInfo/${start}/${end}/`;

	try {
		const r = await fetch(apiUrl, {
			cf: { cacheTtl: 1800, cacheEverything: true } // 30분 캐시
		});

		if (!r.ok) {
			const text = await r.text();
			console.error("Seoul Cultural Events API error:", r.status, text);
			return json({ error: "Seoul Cultural Events API error", status: r.status }, r.status, origin);
		}

		const data = await r.json();

		// API 응답 구조: { culturalEventInfo: { list_total_count, RESULT, row: [...] } }
		if (data.culturalEventInfo?.RESULT?.CODE !== "INFO-000") {
			console.error("Seoul Cultural Events API error:", data.culturalEventInfo?.RESULT);
			return json({ error: "Seoul Cultural Events API error", details: data.culturalEventInfo?.RESULT }, 400, origin);
		}

		const events = data.culturalEventInfo?.row || [];

		return json({
			success: true,
			count: events.length,
			totalCount: data.culturalEventInfo?.list_total_count || events.length,
			events: events
		}, 200, origin);

	} catch (error) {
		console.error("Seoul Cultural Events API fetch error:", error);
		return json({ error: "Failed to fetch Seoul Cultural Events data", details: error.message }, 500, origin);
	}
}

// 버스 정류장 데이터 캐시 (전역)
let busStationsCache = null;
let busRoutesCache = null;
let busRouteGraphCache = null; // 노선별 정류장 순서 그래프
let busTransfersCache = null;  // 정류장별 환승 가능 노선

// 서울시 버스 정류장 정보 API 핸들러
async function handleSeoulBusStations(req, env, origin) {
	// CORS preflight 처리
	if (req.method === "OPTIONS") return preflight(req);

	const url = new URL(req.url);
	const lat = parseFloat(url.searchParams.get("lat"));
	const lng = parseFloat(url.searchParams.get("lng"));
	const radius = parseFloat(url.searchParams.get("radius")) || 300; // 기본 300m

	if (!lat || !lng || isNaN(lat) || isNaN(lng)) {
		return json({ error: "lat and lng are required" }, 400, origin);
	}

	console.log(`[BUS] Search request: lat=${lat}, lng=${lng}, radius=${radius}m`);

	// 실제 데이터 사용 시도
	try {
		// 그리드 인덱스 파일 로드
		if (!busStationsCache) {
			const dataReq = new Request("http://assets.local/data/bus-stations-gridded.json");
			const dataRes = await env.ASSETS.fetch(dataReq);

			if (dataRes.ok) {
				busStationsCache = await dataRes.json();
				console.log(`[BUS] Loaded gridded data: ${Object.keys(busStationsCache).length} grids`);
			} else {
				console.warn(`[BUS] Failed to load gridded data, using fallback`);
			}
		}

		// 버스 노선 매핑 파일 로드
		if (!busRoutesCache) {
			const routesReq = new Request("http://assets.local/data/bus-routes.json");
			const routesRes = await env.ASSETS.fetch(routesReq);

			if (routesRes.ok) {
				busRoutesCache = await routesRes.json();
				console.log(`[BUS] Loaded bus routes data: ${Object.keys(busRoutesCache).length} stations with route info`);
			} else {
				console.warn(`[BUS] Failed to load bus routes data`);
			}
		}

		if (busStationsCache) {
			// 그리드 기반 검색 (빠름!)
			const stations = searchStationsWithGrid(busStationsCache, lat, lng, radius);
			console.log(`[BUS] Found ${stations.length} stations using grid index`);

			// 각 정류장에 노선 정보 추가
			const stationsWithRoutes = stations.map(station => {
				const routeInfo = busRoutesCache && busRoutesCache[station.id];
				return {
					...station,
					routes: routeInfo ? routeInfo.routes : [],
					routeCount: routeInfo ? routeInfo.routes.length : 0
				};
			});

			return json({
				success: true,
				count: stationsWithRoutes.length,
				stations: stationsWithRoutes,
				searchCenter: { lat, lng },
				searchRadius: radius,
				method: "grid-index"
			}, 200, origin);
		}
	} catch (error) {
		console.error(`[BUS] Error loading real data:`, error);
	}

	// Fallback: 하드코딩 데이터
	console.log(`[BUS] Using fallback hardcoded data`);

	// 하드코딩된 서울 주요 버스 정류장 데이터 (샘플)
	// 실제 운영에서는 전체 DB를 사용하거나 다른 API 사용
	const allBusStations = [
		// 강남권
		{ name: "강남역.신논현역", lat: 37.497952, lng: 127.027619, id: "23280", arsId: "23-280" },
		{ name: "강남역사거리", lat: 37.498500, lng: 127.028100, id: "23281", arsId: "23-281" },
		{ name: "역삼역", lat: 37.500229, lng: 127.036536, id: "23410", arsId: "23-410" },
		{ name: "선릉역", lat: 37.504573, lng: 127.049080, id: "23500", arsId: "23-500" },
		{ name: "삼성역", lat: 37.508842, lng: 127.063747, id: "23600", arsId: "23-600" },
		{ name: "신사역", lat: 37.516224, lng: 127.020246, id: "23180", arsId: "23-180" },
		{ name: "논현역", lat: 37.510178, lng: 127.022223, id: "23200", arsId: "23-200" },
		{ name: "압구정역", lat: 37.527326, lng: 127.028227, id: "06001", arsId: "06-001" },

		// 서울역/시청권
		{ name: "서울역", lat: 37.554648, lng: 126.970880, id: "01001", arsId: "01-001" },
		{ name: "서울역광장", lat: 37.555000, lng: 126.971500, id: "01002", arsId: "01-002" },
		{ name: "시청역", lat: 37.564176, lng: 126.976830, id: "01115", arsId: "01-115" },
		{ name: "시청앞", lat: 37.565000, lng: 126.977500, id: "01116", arsId: "01-116" },

		// 광화문/종로권
		{ name: "광화문", lat: 37.571607, lng: 126.976889, id: "01141", arsId: "01-141" },
		{ name: "광화문사거리", lat: 37.572000, lng: 126.977500, id: "01142", arsId: "01-142" },
		{ name: "종로3가", lat: 37.571607, lng: 126.991806, id: "01230", arsId: "01-230" },
		{ name: "종각역", lat: 37.569722, lng: 126.982778, id: "01220", arsId: "01-220" },
		{ name: "명동입구", lat: 37.563692, lng: 126.982994, id: "01180", arsId: "01-180" },
		{ name: "명동역", lat: 37.561500, lng: 126.986500, id: "01181", arsId: "01-181" },

		// 청와대/삼청동/북촌권
		{ name: "경복궁", lat: 37.579617, lng: 126.977041, id: "01300", arsId: "01-300" },
		{ name: "경복궁역", lat: 37.575556, lng: 126.973056, id: "01301", arsId: "01-301" },
		{ name: "삼청동", lat: 37.586111, lng: 126.982222, id: "01350", arsId: "01-350" },
		{ name: "청와대", lat: 37.584000, lng: 126.976800, id: "01360", arsId: "01-360" },
		{ name: "청와대입구", lat: 37.583500, lng: 126.977500, id: "01361", arsId: "01-361" },
		{ name: "청와대사랑채", lat: 37.584500, lng: 126.976000, id: "01362", arsId: "01-362" },
		{ name: "북촌한옥마을", lat: 37.582222, lng: 126.985000, id: "01370", arsId: "01-370" },
		{ name: "안국역", lat: 37.571944, lng: 126.985833, id: "01250", arsId: "01-250" },
		{ name: "삼청터널", lat: 37.585000, lng: 126.978000, id: "01351", arsId: "01-351" },

		// 이태원/용산권
		{ name: "이태원역", lat: 37.534444, lng: 126.994444, id: "05001", arsId: "05-001" },
		{ name: "이태원로", lat: 37.535000, lng: 126.995000, id: "05002", arsId: "05-002" },
		{ name: "용산역", lat: 37.529722, lng: 126.964722, id: "05100", arsId: "05-100" },
		{ name: "용산역광장", lat: 37.530000, lng: 126.965000, id: "05101", arsId: "05-101" },

		// 남산/회현권
		{ name: "남산", lat: 37.551389, lng: 126.988056, id: "01400", arsId: "01-400" },
		{ name: "회현역", lat: 37.559167, lng: 126.978611, id: "01130", arsId: "01-130" },
		{ name: "숭례문", lat: 37.560556, lng: 126.975000, id: "01131", arsId: "01-131" },

		// 동대문/왕십리권
		{ name: "동대문", lat: 37.571270, lng: 127.009789, id: "02001", arsId: "02-001" },
		{ name: "동대문역사문화공원", lat: 37.565556, lng: 127.007778, id: "02010", arsId: "02-010" },
		{ name: "왕십리역", lat: 37.561536, lng: 127.037383, id: "04001", arsId: "04-001" },
		{ name: "왕십리역광장", lat: 37.562000, lng: 127.038000, id: "04002", arsId: "04-002" },

		// 건대/성수권
		{ name: "건대입구역", lat: 37.540462, lng: 127.070239, id: "04200", arsId: "04-200" },
		{ name: "건국대학교", lat: 37.541000, lng: 127.071000, id: "04201", arsId: "04-201" },
		{ name: "성수역", lat: 37.544581, lng: 127.055961, id: "04100", arsId: "04-100" },
		{ name: "뚝섬역", lat: 37.547153, lng: 127.047422, id: "04050", arsId: "04-050" },

		// 홍대/신촌권
		{ name: "홍대입구역", lat: 37.557192, lng: 126.925381, id: "18001", arsId: "18-001" },
		{ name: "홍대정문", lat: 37.556500, lng: 126.924500, id: "18002", arsId: "18-002" },
		{ name: "신촌역", lat: 37.554812, lng: 126.936893, id: "18100", arsId: "18-100" },
		{ name: "신촌오거리", lat: 37.555500, lng: 126.937500, id: "18101", arsId: "18-101" },

		// 잠실권
		{ name: "잠실역", lat: 37.513292, lng: 127.100017, id: "24001", arsId: "24-001" },
		{ name: "잠실새내역", lat: 37.512500, lng: 127.102500, id: "24010", arsId: "24-010" },
		{ name: "잠실광장", lat: 37.514000, lng: 127.101000, id: "24002", arsId: "24-002" },

		// 여의도권
		{ name: "여의도역", lat: 37.521944, lng: 126.924444, id: "07001", arsId: "07-001" },
		{ name: "국회의사당", lat: 37.532222, lng: 126.914722, id: "07100", arsId: "07-100" },
		{ name: "여의도공원", lat: 37.528333, lng: 126.925556, id: "07010", arsId: "07-010" },

		// 종합운동장/삼성동권
		{ name: "종합운동장", lat: 37.510833, lng: 127.073611, id: "24100", arsId: "24-100" },
		{ name: "삼성역", lat: 37.508611, lng: 127.063333, id: "23600", arsId: "23-600" },
		{ name: "선정릉역", lat: 37.504167, lng: 127.049167, id: "23501", arsId: "23-501" },

		// 대학로/혜화권
		{ name: "혜화역", lat: 37.582222, lng: 127.001944, id: "02100", arsId: "02-100" },
		{ name: "대학로", lat: 37.583333, lng: 127.003333, id: "02101", arsId: "02-101" },
		{ name: "마로니에공원", lat: 37.582500, lng: 127.002500, id: "02102", arsId: "02-102" },
	];

	console.log(`[BUS] Using hardcoded station database: ${allBusStations.length} stations`);

	// 주변 정류장 필터링
	const stations = [];

	allBusStations.forEach((station) => {
		const distance = calculateDistance(lat, lng, station.lat, station.lng);

		if (distance <= radius) {
			stations.push({
				id: station.id,
				arsId: station.arsId,
				name: station.name,
				lat: station.lat,
				lng: station.lng,
				distance: Math.round(distance)
			});
		}
	});

	console.log(`[BUS] Found ${stations.length} stations within ${radius}m`);

	// 거리순 정렬
	stations.sort((a, b) => a.distance - b.distance);

	return json({
		success: true,
		count: stations.length,
		stations: stations,
		searchCenter: { lat, lng },
		searchRadius: radius,
		totalStationsChecked: allBusStations.length,
		note: "Using sample data - replace with real API in production"
	}, 200, origin);
}

// 버스 도착정보 API 핸들러 (서울시 공공데이터포털)
async function handleSeoulBusArrival(req, env, origin) {
	// CORS preflight 처리
	if (req.method === "OPTIONS") return preflight(req);

	const url = new URL(req.url);
	const stId = url.searchParams.get("stId");
	const busRouteId = url.searchParams.get("busRouteId");
	const ord = url.searchParams.get("ord");

	// 파라미터 검증
	if (!stId || !busRouteId || !ord) {
		return json({
			success: false,
			error: "Required parameters: stId, busRouteId, ord"
		}, 400, origin);
	}

	try {
		// 서울시 버스도착정보조회 API 호출
		const apiKey = env.SEOUL_BUS_API_KEY;
		const apiUrl = `http://ws.bus.go.kr/api/rest/arrive/getArrInfoByRoute?serviceKey=${encodeURIComponent(apiKey)}&stId=${stId}&busRouteId=${busRouteId}&ord=${ord}&resultType=json`;

		console.log(`[BUS ARRIVAL API] Calling: stId=${stId}, busRouteId=${busRouteId}, ord=${ord}`);

		const response = await fetch(apiUrl);

		if (!response.ok) {
			console.error(`[BUS ARRIVAL API] HTTP Error: ${response.status}`);
			return json({
				success: false,
				error: `Seoul API returned ${response.status}`
			}, 500, origin);
		}

		const data = await response.json();

		// API 응답 구조 확인
		if (!data.msgBody) {
			console.error(`[BUS ARRIVAL API] Invalid response structure:`, data);
			return json({
				success: false,
				error: "Invalid API response structure",
				rawData: data
			}, 500, origin);
		}

		const msgBody = data.msgBody;
		const itemList = msgBody.itemList || [];

		// 첫 번째 아이템 추출 (보통 하나만 반환됨)
		const item = itemList[0] || null;

		if (!item) {
			// 도착 정보 없음
			return json({
				success: true,
				hasData: false,
				message: "No arrival information available"
			}, 200, origin);
		}

		// 응답 데이터 파싱 및 정리
		const arrivalInfo = {
			success: true,
			hasData: true,
			station: {
				stId: item.stId,
				stNm: item.stNm,
				arsId: item.arsId
			},
			route: {
				busRouteId: item.busRouteId || busRouteId,
				busRouteAbrv: item.busRouteAbrv,
				rtNm: item.rtNm,
				routeType: item.routeType,
				firstTm: item.firstTm,
				lastTm: item.lastTm,
				term: item.term,
				nextBus: item.nextBus,
				dir: item.dir
			},
			firstBus: {
				arrmsg: item.arrmsg1,
				plainNo: item.plainNo1,
				traTime: item.traTime1,
				busType: parseInt(item.busType1 || "0"),
				isLast: item.isLast1 === "1",
				isFull: item.full1 === "1",
				congestion: item.rerdie_Div1 === "4" ? parseInt(item.reride_Num1 || "0") : null,
				exps: item.exps1,
			vehId: item.vehId1
			},
			secondBus: item.vehId2 && item.vehId2 !== "0" ? {
				arrmsg: item.arrmsg2,
				plainNo: item.plainNo2,
				traTime: item.traTime2,
				busType: parseInt(item.busType2 || "0"),
				isLast: item.isLast2 === "1",
				isFull: item.full2 === "1",
				congestion: item.rerdie_Div2 === "4" ? parseInt(item.reride_Num2 || "0") : null,
				exps: item.exps2,
			vehId: item.vehId2
			} : null
		};

		console.log(`[BUS ARRIVAL API] Success:`, arrivalInfo);

		return json(arrivalInfo, 200, origin);

	} catch (error) {
		console.error(`[BUS ARRIVAL API] Error:`, error);
		return json({
			success: false,
			error: error.message
		}, 500, origin);
	}
}

// 버스 경로 탐색 API 핸들러
async function handleBusRouteFinder(req, env, origin) {
	// CORS preflight 처리
	if (req.method === "OPTIONS") return preflight(req);

	const url = new URL(req.url);
	const fromLat = parseFloat(url.searchParams.get("fromLat"));
	const fromLng = parseFloat(url.searchParams.get("fromLng"));
	const toLat = parseFloat(url.searchParams.get("toLat"));
	const toLng = parseFloat(url.searchParams.get("toLng"));
	const maxTransfers = parseInt(url.searchParams.get("maxTransfers") || "1", 10);

	if (!fromLat || !fromLng || !toLat || !toLng || isNaN(fromLat) || isNaN(fromLng) || isNaN(toLat) || isNaN(toLng)) {
		return json({ error: "fromLat, fromLng, toLat, toLng are required" }, 400, origin);
	}

	console.log(`[BUS-ROUTE] Finding route: (${fromLat},${fromLng}) → (${toLat},${toLng}), max transfers: ${maxTransfers}`);

	try {
		// 데이터 로드
		if (!busStationsCache) {
			const dataReq = new Request("http://assets.local/data/bus-stations-gridded.json");
			const dataRes = await env.ASSETS.fetch(dataReq);
			if (dataRes.ok) {
				busStationsCache = await dataRes.json();
				console.log(`[BUS-ROUTE] Loaded bus stations`);
			}
		}

		if (!busRouteGraphCache) {
			const graphReq = new Request("http://assets.local/data/bus-route-graph.json");
			const graphRes = await env.ASSETS.fetch(graphReq);
			if (graphRes.ok) {
				busRouteGraphCache = await graphRes.json();
				console.log(`[BUS-ROUTE] Loaded route graph: ${Object.keys(busRouteGraphCache).length} routes`);
			}
		}

		if (!busTransfersCache) {
			const transferReq = new Request("http://assets.local/data/bus-transfers.json");
			const transferRes = await env.ASSETS.fetch(transferReq);
			if (transferRes.ok) {
				busTransfersCache = await transferRes.json();
				console.log(`[BUS-ROUTE] Loaded transfer info: ${Object.keys(busTransfersCache).length} stations`);
			}
		}

		if (!busStationsCache || !busRouteGraphCache || !busTransfersCache) {
			throw new Error("Failed to load bus data");
		}

		// 1. 출발지/도착지 근처 정류장 찾기 (300m 이내)
		const fromStations = searchStationsWithGrid(busStationsCache, fromLat, fromLng, 300);
		const toStations = searchStationsWithGrid(busStationsCache, toLat, toLng, 300);

		console.log(`[BUS-ROUTE] Found ${fromStations.length} start stations, ${toStations.length} end stations`);

		if (fromStations.length === 0 || toStations.length === 0) {
			return json({
				success: false,
				error: "No stations found near start or end point",
				fromStations: fromStations.length,
				toStations: toStations.length
			}, 200, origin);
		}

		// 2. 경로 탐색 (BFS)
		const path = findBusPath(
			fromStations,
			toStations,
			busRouteGraphCache,
			busTransfersCache,
			maxTransfers
		);

		if (!path) {
			return json({
				success: false,
				message: "No route found",
				fromStations: fromStations.slice(0, 3).map(s => ({ name: s.name, distance: s.distance })),
				toStations: toStations.slice(0, 3).map(s => ({ name: s.name, distance: s.distance }))
			}, 200, origin);
		}

		console.log(`[BUS-ROUTE] Found path with ${path.segments.length} segment(s), ${path.transfers} transfer(s)`);

		return json({
			success: true,
			path: path,
			from: {
				lat: fromLat,
				lng: fromLng,
				nearestStation: fromStations[0]
			},
			to: {
				lat: toLat,
				lng: toLng,
				nearestStation: toStations[0]
			}
		}, 200, origin);

	} catch (error) {
		console.error(`[BUS-ROUTE] Error:`, error);
		return json({ error: "Failed to find bus route", details: error.message }, 500, origin);
	}
}

// 경로 재구성 함수: 인덱스 범위를 실제 정류장 배열로 변환
function reconstructPath(pathSegments, routeGraph) {
	return pathSegments.map(seg => {
		const routeData = routeGraph[seg.routeId];
		if (!routeData) return null;

		return {
			routeId: seg.routeId,
			routeName: seg.routeName,
			// 인덱스 범위로 정류장 배열 추출
			stations: routeData.stations.slice(seg.startIdx, seg.endIdx + 1)
		};
	}).filter(seg => seg !== null);
}

// 노선이 목적지 방향으로 가는지 확인하는 함수
function isRouteGoingTowardsDestination(routeData, startIdx, destLat, destLng, checkDistance = 500) {
	// 시작 정류장부터 다음 3-5개 정류장의 평균 위치가 목적지에 가까워지는지 확인
	if (startIdx >= routeData.stations.length - 1) return false;

	const startStation = routeData.stations[startIdx];
	const startDist = calculateDistance(startStation.lat, startStation.lng, destLat, destLng);

	// 다음 몇 개 정류장 확인 (최대 5개)
	const checkCount = Math.min(5, routeData.stations.length - startIdx - 1);
	let approachingCount = 0;

	for (let i = 1; i <= checkCount; i++) {
		const nextStation = routeData.stations[startIdx + i];
		const nextDist = calculateDistance(nextStation.lat, nextStation.lng, destLat, destLng);

		// 거리가 줄어드는지 확인
		if (nextDist < startDist - 100) {  // 100m 이상 가까워지면
			approachingCount++;
		}
	}

	// 절반 이상의 정류장이 목적지에 가까워지면 올바른 방향
	return approachingCount >= Math.ceil(checkCount / 2);
}

// BFS 기반 버스 경로 탐색 (인덱스 기반으로 수정)
function findBusPath(fromStations, toStations, routeGraph, transfers, maxTransfers) {
	// 도착 정류장 NODE_ID 집합
	const destNodeIds = new Set(toStations.map(s => s.id));

	// ✅ 목적지 대표 좌표 (가장 가까운 도착 정류장)
	const destStation = toStations[0];
	const destLat = destStation.lat;
	const destLng = destStation.lng;

	// BFS 큐: { nodeId, routeId, routeName, stationIndex, pathSegments, transferCount }
	const queue = [];

	// 방문 기록: "nodeId_routeId" -> true
	const visited = new Set();

	// 출발 정류장에서 출발하는 모든 버스 노선으로 초기화
	fromStations.forEach(startStation => {
		const transferInfo = transfers[startStation.id];
		if (!transferInfo || !transferInfo.routes) return;

		transferInfo.routes.forEach(route => {
			const routeData = routeGraph[route.routeId];
			if (!routeData) return;

			// 해당 노선에서 출발 정류장의 위치 찾기
			const stationIndex = routeData.stations.findIndex(s => s.nodeId === startStation.id);
			if (stationIndex === -1) return;

			// ✅ 방향성 체크: 이 노선이 목적지 방향으로 가는지 확인
			if (!isRouteGoingTowardsDestination(routeData, stationIndex, destLat, destLng)) {
				console.log(`[BUS-ROUTE-BFS] Skipping ${route.routeName} - wrong direction`);
				return;
			}

			queue.push({
				nodeId: startStation.id,
				routeId: route.routeId,
				routeName: route.routeName,
				stationIndex: stationIndex,
				// ✅ 핵심: stations 배열 대신 인덱스 범위만 저장
				pathSegments: [{
					routeId: route.routeId,
					routeName: route.routeName,
					startIdx: stationIndex,
					endIdx: stationIndex  // 시작과 끝이 같음
				}],
				transferCount: 0
			});

			visited.add(`${startStation.id}_${route.routeId}`);
		});
	});

	console.log(`[BUS-ROUTE-BFS] Starting BFS with ${queue.length} initial routes`);

	// BFS 탐색
	while (queue.length > 0) {
		const current = queue.shift();

		// 현재 노선 데이터
		const routeData = routeGraph[current.routeId];
		if (!routeData) continue;

		// 현재 노선을 따라 다음 정류장들 탐색
		for (let i = current.stationIndex + 1; i < routeData.stations.length; i++) {
			const station = routeData.stations[i];

			// 도착 정류장인지 확인
			if (destNodeIds.has(station.nodeId)) {
				console.log(`[BUS-ROUTE-BFS] Found path! Transfers: ${current.transferCount}`);

				// ✅ 도착지 발견: 마지막 구간의 endIdx를 현재 정류장으로 설정
				const finalSegments = current.pathSegments.map((seg, idx) => {
					if (idx === current.pathSegments.length - 1) {
						return { ...seg, endIdx: i };
					}
					return seg;
				});

				// ✅ 경로 재구성: 인덱스 → 실제 정류장 배열
				const reconstructedPath = reconstructPath(finalSegments, routeGraph);

				return {
					transfers: current.transferCount,
					segments: reconstructedPath,
					totalStations: reconstructedPath.reduce((sum, seg) => sum + seg.stations.length, 0)
				};
			}

			// 환승 가능한 경우 (최대 환승 횟수 이내)
			if (current.transferCount < maxTransfers) {
				const transferInfo = transfers[station.nodeId];
				if (transferInfo && transferInfo.routes) {
					transferInfo.routes.forEach(nextRoute => {
						// 같은 노선이면 스킵
						if (nextRoute.routeId === current.routeId) return;

						const key = `${station.nodeId}_${nextRoute.routeId}`;
						if (visited.has(key)) return;

						const nextRouteData = routeGraph[nextRoute.routeId];
						if (!nextRouteData) return;

						// 환승 후 노선에서 현재 정류장의 위치 찾기
						const nextStationIndex = nextRouteData.stations.findIndex(s => s.nodeId === station.nodeId);
						if (nextStationIndex === -1) return;

						// ✅ 환승 후 노선도 방향성 체크
						if (!isRouteGoingTowardsDestination(nextRouteData, nextStationIndex, destLat, destLng)) {
							console.log(`[BUS-ROUTE-BFS] Skipping transfer to ${nextRoute.routeName} - wrong direction`);
							return;
						}

						// ✅ 환승: 이전 구간의 endIdx를 환승 지점으로 설정
						const transferSegments = current.pathSegments.map((seg, idx) => {
							if (idx === current.pathSegments.length - 1) {
								return { ...seg, endIdx: i };
							}
							return seg;
						});

						// ✅ 새로운 경로 생성 (환승)
						queue.push({
							nodeId: station.nodeId,
							routeId: nextRoute.routeId,
							routeName: nextRoute.routeName,
							stationIndex: nextStationIndex,
							pathSegments: [
								...transferSegments,  // endIdx가 설정된 이전 구간들
								{
									routeId: nextRoute.routeId,
									routeName: nextRoute.routeName,
									startIdx: nextStationIndex,
									endIdx: nextStationIndex
								}
							],
							transferCount: current.transferCount + 1
						});

						visited.add(key);
					});
				}
			}
		}
	}

	console.log(`[BUS-ROUTE-BFS] No path found`);
	return null;
}

// Haversine 거리 계산 함수 (미터 단위)
function calculateDistance(lat1, lng1, lat2, lng2) {
	const R = 6371000; // 지구 반지름 (미터)
	const toRad = (deg) => deg * Math.PI / 180;

	const dLat = toRad(lat2 - lat1);
	const dLng = toRad(lng2 - lng1);

	const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
		Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) *
		Math.sin(dLng / 2) * Math.sin(dLng / 2);

	const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));

	return R * c;
}

// 그리드 기반 정류장 검색 (최적화)
function searchStationsWithGrid(griddedData, lat, lng, radius) {
	const stations = [];

	// 중심 그리드 계산
	const centerGridLat = Math.floor(lat * 100);
	const centerGridLng = Math.floor(lng * 100);

	// 반경을 그리드 단위로 변환 (0.01도 ≈ 1.1km)
	const gridRadius = Math.ceil(radius / 1000); // 보수적으로 계산

	// 주변 그리드 탐색
	for (let dLat = -gridRadius; dLat <= gridRadius; dLat++) {
		for (let dLng = -gridRadius; dLng <= gridRadius; dLng++) {
			const gridKey = `${centerGridLat + dLat}_${centerGridLng + dLng}`;
			const gridStations = griddedData[gridKey];

			if (!gridStations) continue;

			// 그리드 내 정류장 거리 계산
			gridStations.forEach(station => {
				const distance = calculateDistance(lat, lng, station.lat, station.lng);

				if (distance <= radius) {
					stations.push({
						...station,
						distance: Math.round(distance)
					});
				}
			});
		}
	}

	// 거리순 정렬
	stations.sort((a, b) => a.distance - b.distance);

	return stations;
}

function json(body, status=200, origin){
	const h = { "content-type":"application/json; charset=utf-8" };
	if (origin) { h["Access-Control-Allow-Origin"] = origin; h["Vary"] = "Origin"; }
	return new Response(JSON.stringify(body), { status, headers: h });
}


/* -------------------- citydata -------------------- */
const MAX_BATCH_SIZE = 120;

async function refreshAllCitydata(env, ctx) {
	console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
	console.log("[CRON] 🔄 Starting scheduled citydata refresh");
	console.log(`[CRON] ⏰ Trigger time: ${new Date().toISOString()}`);
	const startTime = Date.now();

	try {
		// places.json 로드
		const placesReq = new Request("http://assets.local/places.json");
		const placesRes = await env.ASSETS.fetch(placesReq);
		const placesData = await placesRes.json();
		const places = Array.isArray(placesData)
			? placesData
			: (placesData.places || []);

		console.log(`[CRON] 📊 Total places: ${places.length}`);

		// 🔧 수정: 배치 순환 로직 제거, 항상 전체 처리
		const batch = places.slice(0, MAX_BATCH_SIZE);  // 120개 전체

		console.log(`[CRON] 📦 Batch size: ${batch.length} places`);
		console.log(`[CRON] ⚙️  Rate limit: 100ms between requests`);
		console.log(`[CRON] 🔄 Max retries: 2 per place`);
		console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

		let successCount = 0;
		let failCount = 0;
		const DELAY_BETWEEN_REQUESTS = 100; // 100ms 대기

		// 배치 내 순차 처리
		for (let i = 0; i < batch.length; i++) {
			const place = batch[i];
			const searchParam = place.name;

			if (!searchParam) {
				console.error(`[CRON] ❌ Missing place name for ${place.code}:`, JSON.stringify(place));
				failCount++;
				continue;
			}

			// 🔧 추가: 재시도 로직
			let retries = 2;
			let success = false;
			const attempt_max = 3; // 초기 시도 + 2회 재시도

			while (retries > 0 && !success) {
				const attempt_num = attempt_max - retries;
				try {
					const r = await fetch(
						`http://openapi.seoul.go.kr:8088/${env.SEOUL_GENERAL_KEY}/json/citydata/1/5/${encodeURIComponent(searchParam)}`,
						{
							cf: { cacheTtl: 0 },
							headers: { "User-Agent": "koreaaimap-edge/1.0" }
						}
					);

					if (!r.ok) {
						throw new Error(`HTTP ${r.status}`);
					}

					const data = await r.json();
					const RESULT = (data.RESULT || data.result || {});

					if (!(RESULT.CODE === "INFO-000" || !RESULT.CODE)) {
						throw new Error(`Seoul API error: ${RESULT.CODE} - ${RESULT.MESSAGE || ''}`);
					}

					// 첫 3개 장소는 전체 응답 구조 로그 출력
					if (i < 3) {
						console.log(`[CRON] 🔍 Seoul API response structure for ${place.code}:`, {
							hasAreaNm: !!data.AreaNm,
							hasCITYDATA: !!data.CITYDATA,
							topLevelKeys: Object.keys(data).slice(0, 10),
							CITYDATA_keys: data.CITYDATA ? Object.keys(data.CITYDATA).slice(0, 10) : null
						});

						// ★ 실제 LIVE_PPLTN_STTS 구조 확인
						const citydata = data.CITYDATA || data.citydata || data;
						const liveRaw = deep(citydata, "LIVE_PPLTN_STTS");
						console.log(`[CRON] 🔍 LIVE_PPLTN_STTS for ${place.code}:`, {
							found: !!liveRaw,
							isArray: Array.isArray(liveRaw),
							length: Array.isArray(liveRaw) ? liveRaw.length : 'N/A',
							firstItem: Array.isArray(liveRaw) && liveRaw[0] ? Object.keys(liveRaw[0]).slice(0, 10) : liveRaw
						});
					}

					// 응답 데이터 검증
					if (!data.CITYDATA && !data.citydata) {
						console.warn(`[CRON] ⚠️ ${place.code}: No CITYDATA in response`);
					}

					const payload = {
						place: {
							code: place.code,
							name: data.AreaNm || data.AREA_NM || place.name
						},
						updatedAt: new Date().toISOString(),
						categories: data,
						stats: extractCityStats(data),
						source: "Seoul Open Data"
					};

					// 데이터 내용 검증
					if (!payload.stats.level && !payload.stats.population) {
						console.warn(`[CRON] ⚠️ ${place.code}: Empty stats - level: ${payload.stats.level}, pop: ${payload.stats.population}`);
					}

					// KV Store 저장 (Edge Cache는 각 지역에서 lazy loading)
					const codeKey = `citydata:code:${lc(payload.place.code)}`;
					const nameKey = payload.place.name ? `citydata:name:${lc(payload.place.name)}` : null;

					const cachePromises = [
						cachePutCity(env, codeKey, payload)
					];
					if (nameKey) {
						cachePromises.push(cachePutCity(env, nameKey, payload));
					}
					await Promise.all(cachePromises);

					successCount++;
					success = true; // ✅ 성공

					// 성공 로그
					if (attempt_num > 1) {
						console.log(`[CRON] ✓ ${i + 1}/${batch.length} ${place.code} (retry ${attempt_num - 1})`);
					}

					// 진행률 로깅 (10개마다)
					if ((i + 1) % 10 === 0 || i === batch.length - 1) {
						const progress = ((successCount / batch.length) * 100).toFixed(1);
						const elapsed = Math.round((Date.now() - startTime) / 1000);
						console.log(`[CRON] 📊 Progress: ${successCount}/${batch.length} (${progress}%) | ${elapsed}s elapsed`);
					}

				} catch (error) {
					retries--;
					const retry_msg = retries > 0 ? `${retries} retries left` : 'final attempt failed';
					console.error(`[CRON] ❌ ${place.code} (${place.name}): ${error.message} | ${retry_msg}`);

					if (retries > 0) {
						// Exponential backoff
						const backoff_delay = 200 * (3 - retries);
						await new Promise(r => setTimeout(r, backoff_delay));
					} else {
						// 최종 실패 시에만 카운트
						failCount++;
					}
				}
			}

			// 다음 요청 전 대기
			if (i < batch.length - 1) {
				await new Promise(r => setTimeout(r, DELAY_BETWEEN_REQUESTS));
			}
		}

		const successRate = (successCount / batch.length * 100).toFixed(1);
		const duration = Math.round((Date.now() - startTime) / 1000); // 실행 시간 (초)

		console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
		console.log(`[CRON] ✅ Batch refresh completed`);
		console.log(`[CRON] 📊 Success: ${successCount}/${batch.length} (${successRate}%)`);
		console.log(`[CRON] ❌ Failed: ${failCount}/${batch.length}`);
		console.log(`[CRON] ⏱️  Duration: ${duration}s`);
		console.log(`[CRON] 🔄 Avg: ${(duration / batch.length).toFixed(2)}s per place`);

		// Subrequest 카운팅 (서울시 API 1개 + KV 저장 2개)
		const apiCalls = successCount; // 서울시 API 호출
		const kvWrites = successCount * 2; // KV 저장 (code + name)
		const totalSubrequests = apiCalls + kvWrites;
		console.log(`[CRON] 📡 Seoul API calls: ${apiCalls}`);
		console.log(`[CRON] 💾 KV writes: ${kvWrites} (code + name keys, Edge Cache excluded)`);
		console.log(`[CRON] 🔢 Total subrequests: ${totalSubrequests}/1000 (Edge Cache lazy-loaded per region)`);

		// 캐시 검증 (샘플링)
		console.log(`[CRON] 🔍 Verifying cache (sampling 5 places)...`);
		for (let i = 0; i < Math.min(5, batch.length); i++) {
			const place = batch[i];
			const key = `citydata:code:${lc(place.code)}`;

			try {
				const cached = await cacheGetCity(env, key);
				if (cached) {
					console.log(`[CRON]    ✓ ${key} (${cached.updatedAt})`);
				} else {
					console.log(`[CRON]    ✗ ${key} NOT FOUND!`);
				}
			} catch (err) {
				console.log(`[CRON]    ⚠️ ${key} ERROR: ${err.message}`);
			}
		}

		// 실패율 경고
		if (failCount > batch.length * 0.1) {  // 10% 이상 실패 시 경고
			console.warn(`[CRON] ⚠️  HIGH FAILURE RATE: ${failCount}/${batch.length} places failed (>${(failCount/batch.length*100).toFixed(1)}%)`);
		}

		console.log(`[CRON] 🏁 Cron job completed at ${new Date().toISOString()}`);
		console.log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

	} catch (error) {
		console.error("[CRON] ❌ FATAL ERROR:", error.message);
		console.error("[CRON] Stack:", error.stack);
	}
}


// --- Naver Places Search (APIGW) ---
async function naverPlacesSearch(req, env) {
	const url = new URL(req.url);
	const q = (url.searchParams.get("query") || "").trim();
	const limit = Math.min(Number(url.searchParams.get("limit") || 10), 20);

	const baseHeaders = {
		"Content-Type": "application/json; charset=utf-8",
		// SWR: 5분 fresh, 24시간 stale-while-revalidate
		"Cache-Control": "public, max-age=300, s-maxage=300, stale-while-revalidate=86400"
	};

	if (!q) {
		const res = new Response(JSON.stringify({ items: [] }), { status: 200, headers: baseHeaders });
		res.headers.set("X-Edge-Cache", "BYPASS");
		return withCors(req, res);
	}

	const cacheKey = `https://cache.koreaaimap/internal/naver/search?query=${encodeURIComponent(q)}&limit=${limit}`;
	const cacheReq = new Request(cacheKey, { method: "GET" });
	const cached = await caches.default.match(cacheReq);
	if (cached) {
		return withCors(req, cached, { "X-Edge-Cache": "HIT" });
	}

	const KEY_ID = env.NCP_KEY_ID || env.NAVER_CLIENT_ID;
	const KEY    = env.NCP_KEY    || env.NAVER_CLIENT_SECRET;

	const strip = s => String(s || "").replace(/<[^>]+>/g, "");

	let items = [];
	let provider = "naver";
	let lastError = null;

	if (KEY_ID && KEY) {
		try {
			const api = `https://naveropenapi.apigw.ntruss.com/map-place/v1/search?query=${encodeURIComponent(q)}&display=${limit}`;
			const r = await fetch(api, {
				headers: {
					"X-NCP-APIGW-API-KEY-ID": KEY_ID,
					"X-NCP-APIGW-API-KEY": KEY,
					"Accept": "application/json"
				},
				cf: { cacheTtl: 60, cacheEverything: true },
			});

			if (r.ok) {
				const data = await r.json();
				const arr = data?.places || data?.items || [];
				items = arr.map((it, i) => {
					const lat = Number(it.y ?? it.lat);
					const lng = Number(it.x ?? it.lng);
					if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
					return {
						type: "external",
						provider: "naver",
						id: String(it.id ?? it.link ?? `naver-${i}`),
						name: strip(it.name ?? it.title ?? q),
						address: it.road_address ?? it.address ?? "",
						lat, lng
					};
				}).filter(Boolean);
			} else {
				lastError = new Error(`NAVER APIGW HTTP ${r.status}`);
			}
		} catch (err) {
			lastError = err;
		}
	} else {
		lastError = new Error("NAVER credentials missing");
	}

	if (!items.length) {
		try {
			const osmUrl = `https://nominatim.openstreetmap.org/search?format=json&addressdetails=1&limit=${limit}&q=${encodeURIComponent(q)}`;
			const osmRes = await fetch(osmUrl, {
				headers: {
					"User-Agent": "koreaaimap-edge/1.0 (+https://koreaaimap.com)",
					"Accept": "application/json",
				},
				cf: { cacheTtl: 60, cacheEverything: true },
			});
			if (osmRes.ok) {
				const osm = await osmRes.json();
				items = (Array.isArray(osm) ? osm : []).map((it) => {
					const lat = Number(it.lat), lng = Number(it.lon);
					if (!Number.isFinite(lat) || !Number.isFinite(lng)) return null;
					const display = String(it.display_name || "");
					const primary = display.split(",")[0]?.trim() || display || q;
					return {
						type: "external",
						provider: "osm",
						id: String(it.place_id || `${lat},${lng}`),
						name: primary,
						address: display,
						lat, lng
					};
				}).filter(Boolean);
				provider = "osm";
				lastError = null;
			} else {
				lastError = new Error(`OSM HTTP ${osmRes.status}`);
			}
		} catch (err) {
			lastError = err;
		}
	}

	if (!items.length && lastError) console.error("place search failed", lastError);
	const payload = { items, provider };
	const upstream = new Response(JSON.stringify(payload), { status: 200, headers: baseHeaders });
	const finalRes = withCors(req, upstream);
	try {
		await caches.default.put(cacheReq, finalRes.clone());
	} catch (err) {
		console.error('naver search cache put failed', err);
	}
	finalRes.headers.set("X-Edge-Cache", "MISS");
	return finalRes;
}



async function topisSpeedOne(env, linkId){
	const url = `http://openapi.seoul.go.kr:8088/${env.SEOUL_GENERAL_KEY}/json/TrafficInfo/1/1/${encodeURIComponent(linkId)}`;
	const r = await fetch(url, { cf:{ cacheTtl: 10 }, headers:{ "User-Agent":"koreaaimap-edge/1.0" }});
	if(!r.ok) throw new Error('TOPIS speed upstream error '+r.status);
	const j = await r.json();
	const row = j?.TrafficInfo?.row?.[0] || j?.trafficInfo?.row?.[0];
	if(!row) throw new Error('TOPIS no row');
	return {
		linkId: String(row.link_id || row.LINK_ID || linkId),
		speed: Number(row.prcs_spd || row.PRCSSPD || row.speed || 0),      // km/h
		travelTime: Number(row.prcs_trv_time || row.TRV_TIME || 0)        // sec
	};
}

async function trafficSpeed(req, env) {
	if (!env.SEOUL_GENERAL_KEY) return errJSON(500, "CONFIG_MISSING", "SEOUL_GENERAL_KEY missing", req);

	const u = new URL(req.url);
	const linkId = u.searchParams.get("linkId");
	if (!linkId) return errJSON(400, "BAD_REQUEST", "linkId is required", req);

	try {
		const data = await fetchTopisSpeedXML(linkId, env.SEOUL_GENERAL_KEY);
		// 캐시 10초만 (실시간 근사)
		const res = new Response(JSON.stringify({ ok: true, data }), {
			status: 200,
			headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "public, max-age=10" }
		});
		return withCors(req, res);
	} catch (e) {
		return errJSON(502, "TOPIS_ERROR", String(e), req);
	}
}

async function trafficSpeeds(req, env) {
	if (!env.SEOUL_GENERAL_KEY) return errJSON(500, "CONFIG_MISSING", "SEOUL_GENERAL_KEY missing", req);

	const u = new URL(req.url);
	const linkIds = (u.searchParams.get("linkIds") || "").split(",").map(s => s.trim()).filter(Boolean);
	if (linkIds.length === 0) return errJSON(400, "BAD_REQUEST", "linkIds is required", req);

	const results = await Promise.all(linkIds.map(async (lid) => {
		try {
			return await fetchTopisSpeedXML(lid, env.SEOUL_GENERAL_KEY);
		} catch (e) {
			return { linkId: lid, error: String(e) };
		}
	}));

	const res = new Response(JSON.stringify({ ok: true, data: results }), {
		status: 200,
		headers: { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "public, max-age=10" }
	});
	return withCors(req, res);
}



async function citydataProxy(req, env, ctx) {
	if (req.method === "OPTIONS") return preflight(req);
	if (req.method !== "GET") return errJSON(405, "METHOD_NOT_ALLOWED", "GET only", req);

	const url = new URL(req.url);
	const code  = url.searchParams.get("code");
	const place = url.searchParams.get("place");
	const forceRefresh = url.searchParams.has("_t"); // 🔧 추가: 강제 새로고침 감지

	console.log(`[API] 📥 Request: ${code || place} (${code ? 'code' : 'place'}${forceRefresh ? ', force refresh' : ''})`);

	if (!code && !place) return errJSON(400, "BAD_REQUEST", "use exactly one of ?code or ?place", req);
	if (code && place)   return errJSON(400, "BAD_REQUEST", "use either ?code or ?place (not both)", req);
	if (!env.SEOUL_GENERAL_KEY) return errJSON(500, "CONFIG_MISSING", "SEOUL_GENERAL_KEY missing", req);

	const kCode = code ? `citydata:code:${lc(code)}` : null;
	const kName = place ? `citydata:name:${lc(place)}` : null;
	const cacheKey = kCode || kName;

	// 🔧 추가: 강제 새로고침이 아닐 때만 캐시 확인
	if (!forceRefresh) {
		// 1순위: Edge Cache 조회
		const edgeCached = await cacheGetEdge(cacheKey);
		if (edgeCached) {
			// 데이터 검증 및 로그
			const hasData = edgeCached.stats && (edgeCached.stats.level || edgeCached.stats.population);
			console.log(`[API] 📊 Data check for ${cacheKey}:`, {
				hasStats: !!edgeCached.stats,
				level: edgeCached.stats?.level,
				population: edgeCached.stats?.population,
				placeName: edgeCached.place?.name
			});

			if (!hasData) {
				console.warn(`[API] ⚠️ Edge cache has EMPTY data for ${cacheKey}`);
			}

			return withCors(req, new Response(JSON.stringify({ ...edgeCached, cache: "edge" }), {
				status: 200,
				headers: {
					"Content-Type": "application/json; charset=utf-8",
					"Cache-Control": `public, max-age=${EDGE_CACHE_TTL}, stale-while-revalidate=86400`,
					"X-Edge-Cache": "HIT",
					"X-Cache-Layer": "edge"
				}
			}));
		}

		// 2순위: KV Store 조회
		const kvCached = await cacheGetCity(env, cacheKey);
		if (kvCached) {
			// KV에서 찾았으면 Edge Cache에도 저장 (warming)
			await cachePutEdge(cacheKey, kvCached);

			return withCors(req, new Response(JSON.stringify({ ...kvCached, cache: "kv" }), {
				status: 200,
				headers: {
					"Content-Type": "application/json; charset=utf-8",
					"Cache-Control": `public, max-age=${EDGE_CACHE_TTL}, stale-while-revalidate=86400`,
					"X-Edge-Cache": "MISS",
					"X-KV-Cache": "HIT",
					"X-Cache-Layer": "kv"
				}
			}));
		}
	}

	// 1) 업스트림 호출
	// ★★★ 중요: code가 POI로 시작하면 내부용 코드이므로 장소명으로 변환 필요 ★★★
	if (forceRefresh) {
		console.log(`[API] 🔄 Force refresh: calling Seoul API directly`);
	} else {
		console.log(`[API] 🌐 Cache miss, calling upstream API`);
	}

	const service = "citydata";
	const base = `http://openapi.seoul.go.kr:8088/${env.SEOUL_GENERAL_KEY}/json/${service}/1/5`;

	// POI 코드인 경우, places.json에서 실제 장소명을 찾아야 함
	let searchParam;
	if (code && code.startsWith('POI')) {
		// POI 코드는 서울시 API가 인식 못함 -> 장소명으로 검색
		// places.json 로드 필요
		try {
			const placesReq = new Request("http://assets.local/places.json");
			const placesRes = await env.ASSETS.fetch(placesReq);
			const placesData = await placesRes.json();
			const placeObj = Array.isArray(placesData)
				? placesData.find(p => p.code === code)
				: placesData.places?.find(p => p.code === code);

			if (placeObj && placeObj.name) {
				searchParam = placeObj.name;
				console.log(`[API] 🔄 POI code ${code} → name: ${searchParam}`);
			} else {
				console.error(`[API] ❌ POI code not found: ${code}`);
				return errJSON(404, "PLACE_NOT_FOUND", `Place with code ${code} not found`, req);
			}
		} catch (e) {
			console.error(`[API] ❌ Failed to load places.json:`, e.message);
			return errJSON(500, "PLACES_LOAD_ERROR", "Failed to load places.json", req);
		}
	} else {
		searchParam = code || place;
	}

	const target = `${base}/${encodeURIComponent(searchParam)}`;

	let upstream, data;
	const started = Date.now();
	try {
		console.log(`[API] 📡 Fetching: ${searchParam}`);
		upstream = await fetch(target, {
			cf: { cacheTtl: 0 },
			headers: { "User-Agent": "koreaaimap-edge/1.0" }
		});
		data = await upstream.json();
		const elapsed = Date.now() - started;
		console.log(`[API] ✓ Upstream responded in ${elapsed}ms`);
	} catch (e) {
		const elapsed = Date.now() - started;
		console.error(`[API] ❌ Upstream error after ${elapsed}ms:`, e.message);
		return errJSON(502, "UPSTREAM_FETCH_ERROR", "failed to fetch Seoul OpenAPI", req, { error: String(e) });
	}

	const RESULT = (data.RESULT || data.result || {});
	const ok = upstream.ok && (RESULT.CODE === "INFO-000" || !RESULT.CODE);
	if (!ok) {
		return errJSON(502, "UPSTREAM_RESULT_NOT_OK", "Seoul RESULT not OK", req, {
			upstreamStatus: upstream.status,
			upstreamResult: RESULT,
			searchParam
		});
	}

	const payload = {
		place: {
			code: code || (data.AreaCd || data.AREA_CD) || null,
			name: (data.AreaNm || data.AREA_NM) || place || null
		},
		updatedAt: new Date().toISOString(),
		categories: data,
		stats: extractCityStats(data),
		source: "Seoul Open Data"
	};

	// 2) Edge Cache + KV Store 저장 (code/name 양쪽 키 모두)
	const key1 = payload.place.code ? `citydata:code:${lc(payload.place.code)}` : null;
	const key2 = payload.place.name ? `citydata:name:${lc(payload.place.name)}` : null;

	// 🔧 수정: 강제 새로고침 시에는 Edge Cache만 업데이트
	if (forceRefresh) {
		// 강제 새로고침: Edge Cache만 업데이트 (KV Store는 Cron이 관리)
		if (key1) ctx.waitUntil(cachePutEdge(key1, payload));
		if (key2) ctx.waitUntil(cachePutEdge(key2, payload));
		console.log(`[PROXY] 🔄 Force refresh: Updated Edge Cache only for ${payload.place.code}`);
	} else {
		// 일반 흐름: Edge Cache + KV Store 동시 저장
		if (key1) {
			ctx.waitUntil(Promise.all([
				cachePutEdge(key1, payload),
				cachePutCity(env, key1, payload)
			]));
		}
		if (key2) {
			ctx.waitUntil(Promise.all([
				cachePutEdge(key2, payload),
				cachePutCity(env, key2, payload)
			]));
		}
	}

	return withCors(req, new Response(JSON.stringify(payload), {
		status: 200,
		headers: {
			"Content-Type": "application/json; charset=utf-8",
			"Cache-Control": `public, max-age=${EDGE_CACHE_TTL}, stale-while-revalidate=86400`,
			"X-Elapsed-ms": String(Date.now() - started),
			"X-Source": "seoul-citydata",
			"X-Edge-Cache": "MISS",
			"X-KV-Cache": "MISS",
			"X-Cache-Layer": "upstream"
		}
	}));
}

async function citydataBulk(req, env) {
	if (req.method === "OPTIONS") return preflight(req);
	if (req.method !== "GET") return errJSON(405, "METHOD_NOT_ALLOWED", "GET only", req);
	if (!env.SEOUL_GENERAL_KEY) return errJSON(500, "CONFIG_MISSING", "SEOUL_GENERAL_KEY missing", req);

	const u = new URL(req.url);
	const codes = (u.searchParams.get("codes") || "")
		.split(",").map(s => s.trim()).filter(Boolean);
	const forceRefresh = u.searchParams.has("_t"); // 🔧 추가: 강제 새로고침 감지

	console.log(`[BULK] 📥 Request: ${codes.length} places${forceRefresh ? ' (force refresh)' : ''}`);

	if (!codes.length) return errJSON(400, "BAD_REQUEST", "codes required", req);

	// places.json 로드
	let placesData;
	try {
		const placesReq = new Request("http://assets.local/places.json");
		const placesRes = await env.ASSETS.fetch(placesReq);
		placesData = await placesRes.json();
	} catch (e) {
		return errJSON(500, "PLACES_LOAD_ERROR", "Failed to load places.json", req);
	}

	// 1) Edge Cache 우선 조회 → KV Store 조회 → Upstream (force refresh 시 건너뛰기)
	const out = [];
	const misses = [];
	let edgeHits = 0;
	let kvHits = 0;

	if (forceRefresh) {
		// 🔧 추가: 강제 새로고침 시 캐시 우회
		console.log(`[BULK] 🔄 Force refresh: bypassing all caches, fetching ${codes.length} items from Seoul API`);
		misses.push(...codes);
	} else {
		// 캐시 확인
		for (const c of codes) {
			const key = `citydata:code:${lc(c)}`;

			// 1순위: Edge Cache
			let v = await cacheGetEdge(key);
			if (v) {
				// 첫 3개만 데이터 검증 로그
				if (edgeHits < 3) {
					console.log(`[BULK] 📊 Sample data for ${key}:`, {
						level: v.stats?.level,
						population: v.stats?.population,
						name: v.place?.name
					});
				}
				out.push(v);
				edgeHits++;
				continue;
			}

			// 2순위: KV Store
			v = await cacheGetCity(env, key);
			if (v) {
				out.push(v);
				kvHits++;
				// KV에서 찾았으면 Edge Cache에도 저장 (warming)
				await cachePutEdge(key, v);
				continue;
			}

			// 3순위: Upstream API 호출 필요
			misses.push(c);
		}
	}

	const edgeStatus = misses.length === 0 ? "HIT" : (misses.length === codes.length ? "MISS" : "PARTIAL");
	const hitRate = ((edgeHits + kvHits) / codes.length * 100).toFixed(1);

	console.log(`[BULK] 📊 Cache results:`);
	console.log(`[BULK]    Total: ${codes.length}`);
	console.log(`[BULK]    Edge hits: ${edgeHits}`);
	console.log(`[BULK]    KV hits: ${kvHits}`);
	console.log(`[BULK]    Misses: ${misses.length}`);
	console.log(`[BULK]    Hit rate: ${hitRate}%`);


	// src/index.js - citydataBulk 함수 내부 수정
	if (misses.length > 0) {
		// 🔧 수정: 5ms → 100ms (Seoul API Rate Limit 준수)
		const DELAY_BETWEEN_REQUESTS = 5;

		console.log(`[BULK] 🌐 Fetching ${misses.length} missing items from upstream...`);

		for (let idx = 0; idx < misses.length; idx++) {
			const c = misses[idx];

			// 🔧 추가: 재시도 로직
			let retries = 2;
			let success = false;

			while (retries > 0 && !success) {
				try {
					// POI 코드 → 장소명 변환
					let searchParam = c;
					if (c.startsWith('POI')) {
						const placeObj = Array.isArray(placesData)
							? placesData.find(p => p.code === c)
							: placesData.places?.find(p => p.code === c);
						if (placeObj && placeObj.name) {
							searchParam = placeObj.name;
						} else {
							console.error(`Place not found: ${c}`);
							break; // 재시도 불필요
						}
					}

					const r = await fetch(
						`http://openapi.seoul.go.kr:8088/${env.SEOUL_GENERAL_KEY}/json/citydata/1/5/${encodeURIComponent(searchParam)}`,
						{
							cf: { cacheTtl: 0 },
							headers: { "User-Agent": "koreaaimap-edge/1.0" }
						}
					);

					if (!r.ok) {
						throw new Error(`HTTP ${r.status}`);
					}

					const data = await r.json();
					const RESULT = (data.RESULT || data.result || {});

					if (!(RESULT.CODE === "INFO-000" || !RESULT.CODE)) {
						throw new Error(`Seoul API error: ${RESULT.CODE}`);
					}

					const payload = {
						place: {
							code: c,
							name: data.AreaNm || data.AREA_NM || searchParam,
						},
						updatedAt: new Date().toISOString(),
						categories: data,
						stats: extractCityStats(data),
						source: "Seoul Open Data"
					};

					// 🔧 수정: 강제 새로고침 시에는 Edge Cache만 업데이트
					const codeKey = `citydata:code:${lc(c)}`;
					const nameKey = payload.place.name ? `citydata:name:${lc(payload.place.name)}` : null;

					if (forceRefresh) {
						// 강제 새로고침: Edge Cache만 업데이트 (KV Store는 Cron이 관리)
						const edgeCachePromises = [cachePutEdge(codeKey, payload)];
						if (nameKey) {
							edgeCachePromises.push(cachePutEdge(nameKey, payload));
						}
						await Promise.all(edgeCachePromises);
					} else {
						// 일반 흐름: KV Store만 저장 (Edge Cache는 각 지역에서 lazy loading)
						const cachePromises = [cachePutCity(env, codeKey, payload)];
						if (nameKey) {
							cachePromises.push(cachePutCity(env, nameKey, payload));
						}
						await Promise.all(cachePromises);
					}

					out.push(payload);
					success = true; // ✅ 성공

					if ((idx + 1) % 10 === 0 || idx === misses.length - 1) {
						console.log(`[BULK] 📊 Progress: ${out.length}/${codes.length} total (${idx + 1}/${misses.length} upstream)`);
					}

				} catch (error) {
					retries--;
					const retry_msg = retries > 0 ? `${retries} retries left` : 'final attempt failed';
					console.error(`[BULK] ❌ ${c}: ${error.message} | ${retry_msg}`);

					if (retries > 0) {
						// 🔧 추가: Exponential backoff
						await new Promise(r => setTimeout(r, 200 * (3 - retries)));
					}
				}
			}

			// 다음 요청 전 대기
			if (idx < misses.length - 1) {
				await new Promise(r => setTimeout(r, DELAY_BETWEEN_REQUESTS));
			}
		}

		if (forceRefresh) {
			console.log(`[BULK] ✅ Completed: ${out.length}/${codes.length} items (Force refresh: Edge Cache updated only)`);
		} else {
			console.log(`[BULK] ✅ Completed: ${out.length}/${codes.length} items`);
		}
	} else {
		console.log(`[BULK] ✅ All ${codes.length} items served from cache`);
	}

	return withCors(req, new Response(JSON.stringify({ ok: true, data: out }), {
		status: 200,
		headers: {
			"Content-Type": "application/json; charset=utf-8",
			"Cache-Control": `public, max-age=${EDGE_CACHE_TTL}, stale-while-revalidate=86400`,
			"X-Edge-Cache": edgeStatus,
			"X-Edge-Cache-Hits": String(edgeHits),
			"X-KV-Cache-Hits": String(kvHits),
			"X-Upstream-Calls": String(misses.length),
			"X-Total": String(codes.length),
			"X-Success": String(out.length),
		}
	}));
}




/* -------------------- infer (API GW → Lambda) -------------------- */

function bad(reason, origin) {
	return new Response(JSON.stringify({ ok: false, reason }), {
		status: 400,
		headers: { "Content-Type": "application/json", ...corsHeaders(origin) }
	});
}

// HMAC-SHA256(hex) 서명 유틸(Workers SubtleCrypto)
// HMAC-SHA256 → hex digest
async function hmacHex(keyStr, msg) {
	const enc = new TextEncoder();
	const rawKey = isHexKey(keyStr) ? hexToBytes(keyStr) : enc.encode(keyStr); // ★ 핵심
	const cryptoKey = await crypto.subtle.importKey(
		"raw",
		rawKey,
		{ name: "HMAC", hash: "SHA-256" },
		false,
		["sign"]
	);
	const sig = await crypto.subtle.sign("HMAC", cryptoKey, enc.encode(msg));
	return [...new Uint8Array(sig)].map(b => b.toString(16).padStart(2, "0")).join("");
}


async function inferProxy(req, env) {
	if (req.method === "OPTIONS") return preflight(req);
	if (req.method !== "POST") return errJSON(405, "METHOD_NOT_ALLOWED", "POST only", req);
	if (!env.API_GW_BASE)      return errJSON(500, "CONFIG_MISSING", "API_GW_BASE missing", req);
	let body;
	try { body = await req.json(); } catch { return errJSON(400, "BAD_REQUEST", "invalid JSON", req); }
	// NOTE: 더 이상 body.input 강제 안 함 (route_rank 등 임의의 JSON 허용)

	console.log("🔍 [inferProxy] Received body:", JSON.stringify(body));
	console.log("🔍 [inferProxy] API_GW_BASE:", env.API_GW_BASE);

	const payload = JSON.stringify(body);
	const headers = { "Content-Type": "application/json" };

	if (env.INFER_SIGNING_KEY) {
		try {
			headers["X-Edge-Signature"] = await hmacHex(env.INFER_SIGNING_KEY, payload);
		} catch {}
	}

	let up;
	try {
		// Lambda Function URL은 루트 경로 사용, API Gateway는 /infer 경로 사용
		const targetUrl = env.API_GW_BASE.includes('lambda-url') ? env.API_GW_BASE : `${env.API_GW_BASE}/infer`;
		console.log("📤 [inferProxy] Sending to Lambda:", targetUrl);
		up = await fetch(targetUrl, {
			method: "POST", headers, body: payload, cf: { cacheTtl: 0 }
		});
		console.log("📥 [inferProxy] Lambda response status:", up.status);
	} catch (e) {
		console.error("❌ [inferProxy] Lambda fetch error:", String(e));
		return errJSON(502, "LAMBDA_UPSTREAM_ERROR", "Failed to reach Lambda/API Gateway", req, { error: String(e) });
	}

	// 업스트림 응답을 withCors로 래핑
	const responseText = await up.text();
	console.log("📥 [inferProxy] Lambda response body:", responseText);

	const res = new Response(responseText, {
		status: up.status,
		headers: { "Content-Type": up.headers.get("Content-Type") || "application/json" }
	});
	return withCors(req, res);
}


/* -------------------- router -------------------- */

export default {
	async fetch(req, env, ctx) {

		const url = new URL(req.url);
		const p = url.pathname;

		// 0) CORS Preflight
		if (req.method === "OPTIONS") return preflight(req);

		// 0.5) Cron 수동 트리거 (POST 요청)
		if (p === "/__trigger-cron" && req.method === "POST") {
			ctx.waitUntil(refreshAllCitydata(env, ctx));
			return okJSON({ status: "triggered", message: "Cron job started in background" }, req);
		}

		// 1) API 라우팅을 "먼저" 처리
		if (p === "/health") return okJSON({ status: "ok", service: "koreaaimap-edge" }, req);
		if (p === "/reco/v1/citydata") return citydataProxy(req, env, ctx);
		if (p === "/reco/v1/citydata/bulk") return citydataBulk(req, env);
		if (p === "/traffic/speed") return trafficSpeed(req, env);
		if (p === "/traffic/speeds") return trafficSpeeds(req, env);
		if (p === "/naver/places/search" && req.method === "GET") {
			return naverPlacesSearch(req, env);
		}
		if (p === "/route") return handleRoute(req, env, req.headers.get("Origin") || "");
		if (p === "/naver/direction5") return handleNaverDirection5(req, env, req.headers.get("Origin") || "");
		if (p === "/seoul/subway/stations") return handleSeoulSubwayStations(req, env, req.headers.get("Origin") || "");
		if (p === "/seoul/cultural-events") return handleSeoulCulturalEvents(req, env, req.headers.get("Origin") || "");
		if (p === "/seoul/bus-stations") return handleSeoulBusStations(req, env, req.headers.get("Origin") || "");
		if (p === "/seoul/bus-arrival") return handleSeoulBusArrival(req, env, req.headers.get("Origin") || "");
		if (p === "/seoul/bus-route") return handleBusRouteFinder(req, env, req.headers.get("Origin") || "");
		if (p === "/api/heatmap") return handleHeatmapProxy(req, env);

		if (p.startsWith("/infer")) return inferProxy(req, env);

		// 히트맵 파일 보호: 직접 접근 차단
		if (p.startsWith("/heatmaps/")) {
			const referer = req.headers.get("Referer") || "";
			const origin = req.headers.get("Origin") || "";

			// 자신의 도메인에서만 접근 허용
			const allowedHosts = [
				"orange-brook-ac24.kolom1357.workers.dev",
				"koreaaimap.com",
				"localhost",
				"127.0.0.1"
			];

			const isAllowed = allowedHosts.some(host =>
				referer.includes(host) || origin.includes(host)
			);

			if (!isAllowed) {
				return new Response("Forbidden", { status: 403 });
			}
		}

		// 2) 그 외는 정적 자산으로 위임 (SPA 폴백은 wrangler.toml이 처리)
		if ((req.method === "GET" || req.method === "HEAD") && env.ASSETS) {
			return env.ASSETS.fetch(req);
		}

		return withCors(req, new Response("Not Found", { status: 404 }));
	},
	async scheduled(event, env, ctx) {
		ctx.waitUntil(refreshAllCitydata(env,ctx));
	}
}
