import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { enrichTransitPaths } from "./utils/enrichTransit";
import { haversine } from "./utils/haversine";
import Controls from "./components/Controls.jsx";
import TopList from "./components/TopList.jsx";
import SearchBar from "./components/SearchBar.jsx";
import PlacePicker from "./components/PlacePicker.jsx";
import AirQuality from "./components/AirQuality.jsx";
import AdvancedFilters from "./components/AdvancedFilters.jsx";
import { loadNaverMaps, waitForGeocoder } from "./utils/naverReady.js";
import { API_BASE } from "./utils/api.js";
import "./styles/ui.css";
import { infer } from "./utils/api";
import { loadSubwayStations, findNearestStations, findSubwayPath, getPathSegments, SUBWAY_LINE_COLORS } from "./utils/subwayRoute.js";
import gonggongnuriLogo from "./assets/gonggongnuri.png";
import { generateLinksFromRoute } from './utils/generateLinksFromRoute';

const NAVER_KEY_ID = String(import.meta.env.VITE_NAVER_KEY_ID ?? "").trim();
const NAVER_CUSTOM_URL = (import.meta.env.VITE_NAVER_MAPS_URL || "").trim();

const HEAT_GRADIENT = {
	0.0: "rgba(59,130,246,0)",
	0.2: "rgba(37,99,235,0.18)",
	0.4: "rgba(14,165,233,0.38)",
	0.6: "rgba(99,102,241,0.62)",
	0.8: "rgba(236,72,153,0.82)",
	1.0: "rgba(239,68,68,0.95)",
};


const GEO_CACHE_KEY = "GEO_CACHE_V2";
const GEOCODE_DELAY_MS = 20;
let __lastGeoAt = 0;
async function throttleGeocode() {
	const now = Date.now();
	const wait = Math.max(0, __lastGeoAt + GEOCODE_DELAY_MS - now);
	if (wait) await new Promise(r => setTimeout(r, wait));
	__lastGeoAt = Date.now();
}

function loadGeoCache() {
	try {
		return JSON.parse(localStorage.getItem(GEO_CACHE_KEY) || "{}");
	} catch {
		return {};
	}
}

function saveGeoCache(cache) {
	try {
		// LRU 정책: 최대 100개까지만 유지
		let cacheToSave = cache;
		const entries = Object.entries(cache);

		if (entries.length > 100) {
			console.warn(`[GeoCache] Cache size (${entries.length}) exceeds limit, keeping latest 100 entries`);
			// 최근 100개만 유지 (마지막 100개)
			const recent = entries.slice(-100);
			cacheToSave = Object.fromEntries(recent);
		}

		localStorage.setItem(GEO_CACHE_KEY, JSON.stringify(cacheToSave));
	} catch (e) {
		// QuotaExceededError 처리
		if (e.name === 'QuotaExceededError') {
			console.error('[GeoCache] LocalStorage quota exceeded, clearing old entries');
			try {
				// 절반만 유지
				const entries = Object.entries(cache);
				const half = entries.slice(-50);
				localStorage.setItem(GEO_CACHE_KEY, JSON.stringify(Object.fromEntries(half)));
			} catch (retryErr) {
				console.error('[GeoCache] Failed to save even after clearing:', retryErr);
			}
		} else {
			console.error('[GeoCache] Failed to save cache:', e);
		}
	}
}


async function fetchJSON(url, { timeout = 300000, retries = 2 } = {}) {
	let lastError;

	for (let attempt = 0; attempt < retries; attempt++) {
		const controller = new AbortController();
		const timeoutId = setTimeout(() => {
			controller.abort();
		}, timeout);

		try {
			const res = await fetch(url, {
				signal: controller.signal
			});

			clearTimeout(timeoutId);

			if (!res.ok) {
				throw new Error(`HTTP ${res.status}`);
			}

			return res.json();

		} catch (error) {
			clearTimeout(timeoutId);
			lastError = error;

			if (error.name === 'AbortError') {
				console.warn(`Request timeout (${timeout}ms) for ${url}, attempt ${attempt + 1}/${retries}`);
			} else {
				console.error(`Fetch failed for ${url}, attempt ${attempt + 1}/${retries}:`, error.message);
			}

			// 마지막 시도가 아니면 재시도 전 대기 (exponential backoff)
			if (attempt < retries - 1) {
				await new Promise(r => setTimeout(r, 300 * Math.pow(2, attempt)));
			}
		}
	}

	throw lastError || new Error(`Failed to fetch ${url} after ${retries} attempts`);
}

function numberOrNull(value) {
	if (value === null || value === undefined || value === '') return null;
	const cleaned = String(value).replace(/,/g, '').trim();
	const num = Number(cleaned);
	return Number.isFinite(num) ? num : null;
}


function deepFind(obj, ...keys) {
	if (!obj || typeof obj !== 'object') return null;

	// Stack 기반 BFS로 전체 트리 탐색
	const stack = [obj];
	const visited = new Set();

	while (stack.length > 0) {
		const current = stack.pop();

		if (!current || typeof current !== 'object' || visited.has(current)) {
			continue;
		}
		visited.add(current);

		// 현재 객체에서 키 찾기
		for (const key of keys) {
			const lowerKey = key.toLowerCase();
			for (const objKey of Object.keys(current)) {
				if (objKey.toLowerCase() === lowerKey) {
					let result = current[objKey];
					if (Array.isArray(result) && result.length > 0) {
						result = result[0];
					}
					return result;
				}
			}
		}

		// 하위 객체를 스택에 추가
		for (const value of Object.values(current)) {
			if (value && typeof value === 'object') {
				stack.push(value);
			}
		}
	}

	return null;
}


function parseKoreanDate(value) {
	if (!value) return null;

	if (value instanceof Date) {
		return Number.isNaN(value.getTime()) ? null : value;
	}

	if (typeof value === "string") {
		const trimmed = value.trim();

		// YYYYMMDDHHMM 형식 (예: "202510311430")
		if (/^\d{12}$/.test(trimmed)) {
			const y = Number(trimmed.slice(0, 4));
			const m = Number(trimmed.slice(4, 6)) - 1;
			const d = Number(trimmed.slice(6, 8));
			const hh = Number(trimmed.slice(8, 10));
			const mm = Number(trimmed.slice(10, 12));
			const date = new Date(y, m, d, hh, mm);
			return Number.isNaN(date.getTime()) ? null : date;
		}

		// ISO 형식
		const date = new Date(trimmed);
		return Number.isNaN(date.getTime()) ? null : date;
	}

	return null;
}

function createEmptyStats() {
	return {
		index: null,
		level: null,
		message: '데이터 없음',
		population: null,
		dataTime: null,
		timestamp: null,
		roadSpeed: null,
		fcstLevel: null,
		fcstTime: null,
	};
}

/**
 * 업종 중분류를 9개 카테고리로 매핑
 */
function mapBusinessCategory(business) {
	if (!business || !business.RSB_MID_CTGR) return '기타';

	const mid = String(business.RSB_MID_CTGR).toLowerCase();
	const large = String(business.RSB_LRG_CTGR || '').toLowerCase();

	// 유흥
	if (mid.includes('유흥') || mid.includes('주점') || large.includes('유흥')) {
		return '유흥';
	}

	// 스포츠/문화/레저
	if (mid.includes('스포츠') || mid.includes('문화') || mid.includes('레저') ||
		mid.includes('헬스') || mid.includes('체육') || mid.includes('영화') ||
		mid.includes('공연') || mid.includes('오락')) {
		return '스포츠/문화/레저';
	}

	// 한식
	if (mid.includes('한식') || mid.includes('한정식') || mid.includes('찌개') ||
		mid.includes('국밥') || mid.includes('분식')) {
		return '한식';
	}

	// 일식/중식/양식
	if (mid.includes('일식') || mid.includes('중식') || mid.includes('양식') ||
		mid.includes('돈까스') || mid.includes('초밥') || mid.includes('라면') ||
		mid.includes('스테이크') || mid.includes('파스타')) {
		return '일식/중식/양식';
	}

	// 제과/커피/패스트푸드
	if (mid.includes('제과') || mid.includes('커피') || mid.includes('카페') ||
		mid.includes('베이커리') || mid.includes('패스트푸드') || mid.includes('햄버거') ||
		mid.includes('피자') || mid.includes('치킨') || mid.includes('디저트')) {
		return '제과/커피/패스트푸드';
	}

	// 편의점
	if (mid.includes('편의점')) {
		return '편의점';
	}

	// 할인점/슈퍼마켓
	if (mid.includes('할인점') || mid.includes('슈퍼마켓') || mid.includes('마트') ||
		mid.includes('대형마트')) {
		return '할인점/슈퍼마켓';
	}

	// 기타요식 (음식 관련이지만 위 카테고리에 포함되지 않음)
	if (large.includes('음식') || large.includes('식당') || large.includes('요식') ||
		mid.includes('음식') || mid.includes('식당')) {
		return '기타요식';
	}

	return '기타';
}

/**
 * 성별 비율 카테고리 판별
 */
function getGenderRatioCategory(maleRate, femaleRate) {
	if (!maleRate && !femaleRate) return null;

	const male = parseFloat(maleRate) || 0;
	const female = parseFloat(femaleRate) || 0;

	if (male >= 60) return '6:4~10:0';
	if (male >= 50) return '5:5~6:4';
	if (male >= 40) return '4:6~5:5';
	return '0:10~4:6';
}

/**
 * 연령대 비율 카테고리 판별 (여러 개 반환 가능)
 */
function getAgeGroupCategories(demographics) {
	if (!demographics || !demographics.age) return [];

	const categories = [];
	const age = demographics.age;

	if (age.age0 >= 4) categories.push('10세 미만(4% 이상)');
	if (age.age10 >= 9) categories.push('10대(9% 이상)');
	if (age.age20 >= 20) categories.push('20대(20% 이상)');
	if (age.age30 >= 20) categories.push('30대(20% 이상)');
	if (age.age40 >= 20) categories.push('40대(20% 이상)');
	if (age.age50 >= 18) categories.push('50대(18% 이상)');
	if (age.age60 >= 14) categories.push('60대(14% 이상)');
	if (age.age70 >= 10) categories.push('70세 이상(10% 이상)');

	// 모든 연령대가 조건을 만족하지 않으면 '기타'
	if (categories.length === 0) {
		categories.push('기타');
	}

	return categories;
}

/**
 * 상주·비상주 비율 카테고리 판별
 */
function getResidencyRatioCategory(residentRate, nonResidentRate) {
	if (!residentRate && !nonResidentRate) return null;

	const resident = parseFloat(residentRate) || 0;
	const nonResident = parseFloat(nonResidentRate) || 0;

	if (resident >= 60) return '6:4~10:0';
	if (resident >= 50) return '5:5~6:4';
	if (resident >= 40) return '4:6~5:5';
	return '0:10~4:6';
}


function extractLiveStats(categories) {
	if (!categories || typeof categories !== 'object') {
		return createEmptyStats();
	}

	const citydata = categories.CITYDATA || categories.citydata || categories;

	// LIVE_PPLTN_STTS 찾기
	let live = deepFind(
		citydata,  // ← categories 대신 citydata 사용
		'LIVE_PPLTN_STTS',
		'live_ppltn_stts',
		'LivePpltnStts'
	);

	if (Array.isArray(live) && live.length > 0) {
		live = live[0];
	}

	// FCST_PPLTN_STTS 찾기
	let fcst = deepFind(
		citydata,  // ← categories 대신 citydata 사용
		'FCST_PPLTN_STTS',
		'fcst_ppltn_stts',
		'FcstPpltnStts'
	);

	if (Array.isArray(fcst) && fcst.length > 0) {
		fcst = fcst[0];
	}

	// ROAD_TRAFFIC_STTS 찾기
	let road = deepFind(
		citydata,  // ← categories 대신 citydata 사용
		'ROAD_TRAFFIC_STTS',
		'road_traffic_stts',
		'RoadTrafficStts'
	);

	if (Array.isArray(road) && road.length > 0) {
		road = road[0];
	}

	// live 데이터가 없거나 객체가 아니면 빈 stats 반환
	if (!live || typeof live !== 'object') {
		console.warn('No valid LIVE_PPLTN_STTS found in categories:', categories);
		return createEmptyStats();
	}

	// 인구 수 추출 (여러 필드명 시도)
	const population = numberOrNull(
		live.PPLTN_CNT ??
		live.LIVE_PPLTN_CNT ??
		live.TOT_PPLTN ??
		live.PPLTN_NUM ??
		live.AREA_PPLTN_MIN ??
		live.AREA_PPLTN_MAX
	);

	// 혼잡도 레벨
	const level =
		live.AREA_CONGEST_LVL ||
		live.AREA_CONGEST_MSG ||
		live.AREA_CONGEST_LVL_CD ||
		live.area_congest_lvl ||
		live.area_congest_msg ||
		null;

	// 혼잡도 메시지
	const message =
		live.AREA_CONGEST_MSG ||
		live.area_congest_msg ||
		level ||
		null;

	// 혼잡도 지수
	const index = numberOrNull(
		live.LIVE_PPLTN_INDEX ??
		live.AREA_PPLTN_INDEX ??
		live.live_ppltn_index
	);

	// 데이터 시각
	const dataTime =
		live.LIVE_PPLTN_TIME ||
		live.LIVE_PPLTN_DT ||
		live.DATA_TIME ||
		live.PPLTN_TIME ||
		live.live_ppltn_time ||
		null;

	const timestamp = parseKoreanDate(dataTime);

	// 도로 속도
	const roadSpeed = road?.ROAD_TRAFFIC_SPD
		? numberOrNull(road.ROAD_TRAFFIC_SPD)
		: null;

	// 예보 데이터
	const fcstLevel = fcst?.AREA_CONGEST_LVL || fcst?.AREA_CONGEST_MSG || null;
	const fcstTime = fcst?.FCST_PPLTN_TIME || fcst?.FCST_PPLTN_DT || null;

	const events = extractEvents(categories);
	const accidents = extractAccidents(categories);
	const commerce = extractCommerce(categories);
	const weather = extractWeather(categories);
	const demographics = extractDemographics(categories);
	const forecast = extractForecast(categories);

	return {
		index,
		level,
		message,
		population,
		dataTime,
		timestamp,
		roadSpeed,
		fcstLevel,
		fcstTime,
		events,
		accidents,
		commerce,
		weather,
		demographics,
		forecast,
	};
}

/**
 * 인구 구성 비율 추출
 */
function extractDemographics(categories) {
	const citydata = categories?.CITYDATA || categories?.citydata || categories;
	const live = citydata?.LIVE_PPLTN_STTS?.[0] || {};

	return {
		male: parseFloat(live.MALE_PPLTN_RATE) || 0,
		female: parseFloat(live.FEMALE_PPLTN_RATE) || 0,
		age: {
			age0: parseFloat(live.PPLTN_RATE_0) || 0,
			age10: parseFloat(live.PPLTN_RATE_10) || 0,
			age20: parseFloat(live.PPLTN_RATE_20) || 0,
			age30: parseFloat(live.PPLTN_RATE_30) || 0,
			age40: parseFloat(live.PPLTN_RATE_40) || 0,
			age50: parseFloat(live.PPLTN_RATE_50) || 0,
			age60: parseFloat(live.PPLTN_RATE_60) || 0,
			age70: parseFloat(live.PPLTN_RATE_70) || 0,
		},
		resident: parseFloat(live.RESNT_PPLTN_RATE) || 0,
		nonResident: parseFloat(live.NON_RESNT_PPLTN_RATE) || 0,
	};
}

/**
 * 혼잡도 예측 정보 추출
 */
function extractForecast(categories) {
	const citydata = categories?.CITYDATA || categories?.citydata || categories;
	const live = citydata?.LIVE_PPLTN_STTS?.[0] || {};
	const fcstArray = live?.FCST_PPLTN || [];

	if (!Array.isArray(fcstArray)) return [];

	return fcstArray.map(f => ({
		time: f.FCST_TIME,
		level: f.FCST_CONGEST_LVL,
		min: parseInt(f.FCST_PPLTN_MIN) || 0,
		max: parseInt(f.FCST_PPLTN_MAX) || 0,
	}));
}

function extractEvents(categories) {
	const citydata = categories?.CITYDATA || categories?.citydata || categories;
	const events = citydata?.EVENT_STTS || [];
	if (!Array.isArray(events)) return [];

	return events.slice(0, 10).map(event => ({
		// 이름
		name: event.EVENT_NM,
		TITLE: event.EVENT_NM,
		// 기간
		period: event.EVENT_PERIOD,
		DATE: event.EVENT_PERIOD,
		EVENT_PERIOD: event.EVENT_PERIOD,
		// 장소
		place: event.EVENT_PLACE,
		PLACE: event.EVENT_PLACE,
		EVENT_PLACE: event.EVENT_PLACE,
		// 이미지
		THUMBNAIL: event.THUMBNAIL,
		MAIN_IMG: event.THUMBNAIL,
		IMGURL: event.THUMBNAIL,
		// URL
		URL: event.URL,
		HMPG_ADDR: event.URL,
		// 좌표
		EVENT_X: event.EVENT_X,
		EVENT_Y: event.EVENT_Y,
		// 유료/무료
		PAY_YN: event.PAY_YN,
		USE_FEE: event.PAY_YN === 'Y' ? '유료' : event.PAY_YN === 'N' ? '무료' : null,
		// 상세설명
		EVENT_ETC_DETAIL: event.EVENT_ETC_DETAIL
	}));
}

function extractAccidents(categories) {
	const citydata = categories?.CITYDATA || categories?.citydata || categories;
	const accidents = citydata?.ACDNT_CNTRL_STTS || [];
	if (!Array.isArray(accidents)) return [];

	return accidents.map(acc => ({
		type: acc.ACDNT_TYPE || acc.ACDNT_DTYPE || '사고',
		location: acc.ACDNT_INFO || '위치 미상',
		startTime: acc.ACDNT_OCCR_DT || acc.ACDNT_TIME
	}));
}

/**
 * 실시간 상권 현황 추출
 */
function extractCommerce(categories) {
	const citydata = categories?.CITYDATA || categories?.citydata || categories;
	const commerce = citydata?.LIVE_CMRCL_STTS || {};

	return {
		level: commerce.AREA_CMRCL_LVL || null,
		paymentCount: commerce.AREA_SH_PAYMENT_CNT,
		paymentMin: commerce.AREA_SH_PAYMENT_AMT_MIN,
		paymentMax: commerce.AREA_SH_PAYMENT_AMT_MAX,
		time: commerce.CMRCL_TIME,
		// 업종별 상세 정보
		businesses: commerce.CMRCL_RSB || [],
		// 성별 비율
		maleRate: commerce.CMRCL_MALE_RATE,
		femaleRate: commerce.CMRCL_FEMALE_RATE,
		// 연령대별 비율
		age10Rate: commerce.CMRCL_10_RATE,
		age20Rate: commerce.CMRCL_20_RATE,
		age30Rate: commerce.CMRCL_30_RATE,
		age40Rate: commerce.CMRCL_40_RATE,
		age50Rate: commerce.CMRCL_50_RATE,
		age60Rate: commerce.CMRCL_60_RATE,
		// 개인/법인 비율
		personalRate: commerce.CMRCL_PERSONAL_RATE,
		corporationRate: commerce.CMRCL_CORPORATION_RATE
	};
}

/**
 * 날씨 정보 추출
 */
function extractWeather(categories) {
	const citydata = categories?.CITYDATA || categories?.citydata || categories;
	const weatherArray = citydata?.WEATHER_STTS || [];
	const weather = Array.isArray(weatherArray) && weatherArray.length > 0
		? weatherArray[0]
		: {};

	return {
		temp: weather.TEMP,
		sky: weather.PCP_MSG,
		rainProbability: weather.RAIN_CHANCE,
		// 미세먼지 상세 정보 추출
		pm25: numberOrNull(weather.PM25), // 초미세먼지 농도
		pm25Index: weather.PM25_INDEX, // 초미세먼지 지표 (좋음/보통/나쁨 등)
		pm10: numberOrNull(weather.PM10), // 미세먼지 농도
		pm10Index: weather.PM10_INDEX, // 미세먼지 지표
		// 레거시 호환성
		fineDust: weather.PM10 || weather.PM10_INDEX,
		time: weather.WEATHER_TIME
	};
}

function congestionBias(level) {
	if (!level) return 0;
	const text = String(level).toLowerCase();

	if (/여유|원활|한산|원활함/.test(text)) return -0.08;
	if (/보통|보합|보통임/.test(text)) return 0.04;
	if (/혼잡|주의/.test(text)) return 0.14;
	if (/매우|심각|붐빔|붐빔|혼잡도 높음/.test(text)) return 0.22;

	return 0;
}

// 이용료 문자열에서 금액 추출
function parseFeeString(feeStr) {
	if (!feeStr || feeStr.trim() === '') return 0;
	const text = String(feeStr).trim().toLowerCase();

	// 무료 판별
	if (/무료|free|없음/.test(text)) return 0;

	const numbers = [];

	// "1만원", "3만원" 등 한글 단위 (예: "1인 1만원")
	const koreanMatch = text.match(/(\d+)\s*만\s*원?/g);
	if (koreanMatch) {
		koreanMatch.forEach(m => {
			const num = parseInt(m.match(/\d+/)[0]);
			numbers.push(num * 10000);
		});
	}

	// 일반 숫자 (쉼표 포함, 예: "10,000원", "30000원")
	const numMatch = text.match(/(\d{1,3}(,\d{3})*|\d+)\s*원?/g);
	if (numMatch) {
		numMatch.forEach(m => {
			const num = parseInt(m.replace(/,/g, '').match(/\d+/)[0]);
			// 100보다 큰 값만 유효한 금액으로 간주
			if (num > 100) numbers.push(num);
		});
	}

	// 추출된 숫자가 없으면 -1 (기타)
	if (numbers.length === 0) return -1;

	// 여러 금액이 있으면 최소값 반환 (예: "30,000원 / 50,000원" → 30000)
	return Math.min(...numbers);
}

// 이용료 카테고리 분류
function categorizeFee(fee) {
	if (fee === 0) return 'free';
	if (fee === -1) return 'other';
	if (fee <= 5000) return '1-5000';
	if (fee <= 10000) return '5001-10000';
	if (fee <= 20000) return '10001-20000';
	if (fee <= 30000) return '20001-30000';
	return '30000+';
}

function formatRelative(date) {
	if (!date) return "-";

	const now = Date.now();
	const diff = now - date.getTime();

	if (diff < 0) return "방금 전"; // 시간이 미래인 경우
	if (diff < 60_000) return "방금 전";
	if (diff < 3_600_000) return `${Math.round(diff / 60_000)}분 전`;
	if (diff < 86_400_000) return `${Math.round(diff / 3_600_000)}시간 전`;

	// 24시간 이상 지났으면 날짜 표시
	return date.toLocaleString("ko-KR", {
		month: 'short',
		day: 'numeric',
		hour: "2-digit",
		minute: "2-digit"
	});
}

function formatPopulation(value) {
	if (!value || value <= 0) return "정보 없음";
	return `${Math.round(value).toLocaleString()}명`;
}

function buildHeatWeight(item, maxPopulation, maxIndex) {
	const popWeight = maxPopulation > 0
		? (item.stats.population || 0) / maxPopulation
		: 0;

	const indexWeight = maxIndex > 0
		? (item.stats.index || 0) / maxIndex
		: 0;

	// 혼잡도 바이어스 추가
	const bias = congestionBias(item.stats.level);

	// 가중 평균 (인구 60%, 지수 30%, 바이어스 10%)
	const baseWeight = (popWeight * 0.6) + (indexWeight * 0.3);
	const finalWeight = Math.max(0, Math.min(1, baseWeight + bias));

	return finalWeight;
}

// 자주 실패하는 이름 보정 사전
const NAME_FIX = {
	'동대문 관광특구': '동대문',
	'명동 관광특구': '명동',
	'종로·청계 관광특구': '청계천',
	'홍대 관광특구': '홍대입구역',
	'잠실 관광특구': '잠실 롯데타워',
	'광화문·덕수궁': '광화문',
	'창덕궁·종묘': '창덕궁',
	'서울식물원·마곡나루역': '서울식물원',
	'신논현역·논현역': '신논현역',
	'오목교역·목동운동장': '오목교역',
	'덕수궁길·정동길': '덕수궁',
	'성수카페거리': '성수동 카페거리',
	'송리단길·호수단길': '송리단길',
	'북창동 먹자골목': '북창동 먹자골목 중구',
	'청담동 명품거리': '청담동 명품거리',
	'강서한강공원': '강서 한강공원',
	'서리풀공원·몽마르뜨공원': '서리풀공원',
	'서울숲공원': '서울숲',
	'고척돔': '고척스카이돔',
	'광장(전통)시장': '광장시장',
	'홍대입구역(2호선)': '홍대입구역',
	'잠실롯데타워 일대': '롯데월드타워',
};
async function waitForNonZeroSize(el, ms = 3000) {
	const t0 = Date.now();
	while (el && (el.offsetWidth === 0 || el.offsetHeight === 0)) {
		if (Date.now() - t0 > ms) throw new Error('map element has no size');
		await new Promise(r => requestAnimationFrame(r));
	}
}


function normalizeName(s) {
	return s
		.normalize('NFKC')
		.replace(/[·•]/g, ' ')
		.replace(/\(.+?\)/g, '')
		.replace(/\s*일대$/,'')        // '일대'만 제거
		.replace(/역(\s*\(.+?\))?$/, '역' )   // '역'은 유지(지오코딩 키워드)
		.replace(/\s+/g, ' ')
		.trim();
}

async function geocodeWithNaver(raw) {
	//await waitForGeocoder();
	const svc = window.naver.maps.Service;

	const name0 = String(raw || '').trim();
	const base = normalizeName(NAME_FIX[name0] || name0);


	// 1) 지역→장소(우선)  2) 장소→지역(후순위)
	const generated = [base];
	const candidates = [base, `서울 ${base}`, `${base} 서울`];

	const tried = new Set();
	for (const q of candidates) {
		if (tried.has(q)) continue;
		tried.add(q);
		await throttleGeocode();

		const [status, resp] = await new Promise(r => {
			try {
				svc.geocode({ query: q }, (st, res) => r([st, res])); }
			catch { r([null, null]); }
		});
		if (status === svc.Status.OK) {
			const i = resp?.result?.items?.[0];
			if (i?.point?.x && i?.point?.y) return { lat: +i.point.y, lng: +i.point.x };
			const v2 = resp?.v2?.addresses?.[0];
			if (v2?.x && v2?.y) return { lat: +v2.y, lng: +v2.x };
		}
		await new Promise(r => setTimeout(r, 80));
	}
	throw new Error('지오코딩 결과 없음');
}



// 🔧 수정: forceRefresh 파라미터 추가
async function fetchCitydataBatch(places, geoCache, forceRefresh = false) {
	// 1) 코드만 추출 + 안전한 chunking
	const codes = places.map(p => String(p.code || "")).filter(Boolean);
	const CHUNK = 10;
	const chunks = [];
	for (let i = 0; i < codes.length; i += CHUNK) chunks.push(codes.slice(i, i + CHUNK));

	// 2) bulk 호출 시도 (Promise.allSettled로 변경)
	async function callBulk(oneChunk) {
		// 🔧 추가: forceRefresh 시 타임스탬프로 캐시 무효화
		const cacheBuster = forceRefresh ? `&_t=${Date.now()}` : '';
		const url = `${API_BASE}/reco/v1/citydata/bulk?codes=${encodeURIComponent(oneChunk.join(","))}${cacheBuster}`;

		const j = await fetchJSON(url);
		return Array.isArray(j?.data) ? j.data : [];
	}

	let rows = [];
	try {
		// ✅ 수정: Promise.all → Promise.allSettled
		const results = await Promise.allSettled(chunks.map(callBulk));

		// 성공한 chunk만 추출
		const blocks = results
			.filter(r => r.status === 'fulfilled')
			.map(r => r.value);

		// 실패한 chunk 로깅
		results.forEach((r, i) => {
			if (r.status === 'rejected') {
				console.error(`Chunk ${i} failed (codes: ${chunks[i].slice(0, 3).join(',')}, ...):`, r.reason?.message || r.reason);
			}
		});

		rows = blocks.flat();

		// ✅ 추가: 부분 성공 로깅
		const successRate = (results.filter(r => r.status === 'fulfilled').length / results.length * 100).toFixed(1);
		console.log(`Bulk fetch: ${successRate}% success (${rows.length}/${codes.length} items)`);

	} catch (e) {
		// bulk 미구현/오류 시 → 단건 폴백
		console.warn("bulk 호출 실패, 단건 폴백으로 진행:", e?.message || e);

		const results = await Promise.allSettled(places.map(async (p) => {
			try {
				// 🔧 수정: 단건도 타임스탬프만 사용
				const cacheBuster = forceRefresh ? `&_t=${Date.now()}` : '';
				const single = await fetchJSON(`${API_BASE}/reco/v1/citydata?code=${encodeURIComponent(p.code)}${cacheBuster}`);
				return {
					place: {
						code: String(single.AreaCd || single.AREA_CD || p.code || ""),
						name: String(single.AreaNm || single.AREA_NM || p.name || ""),
					},
					categories: single,
					updatedAt: single?.updatedAt || new Date().toISOString(),
				};
			} catch (err) {
				console.error(`Failed to fetch ${p.code} (${p.name}):`, err.message);
				return null;
			}
		}));

		rows = results
			.filter(r => r.status === 'fulfilled')
			.map(r => r.value)
			.filter(Boolean);

		const failedCount = results.filter(r => r.status === 'rejected' || !r.value).length;
		console.warn(`Fallback fetch: ${failedCount}/${places.length} items failed`);
	}

	// 3) code → row 매핑
	const rowByCode = new Map(rows.map(r => [String(r?.place?.code || ""), r]));

	// 4) places의 순서를 보존하며 결과 구성 + 좌표 보강(캐시 우선)
	const out = [];
	for (const p of places) {
		const key = String(p.code || "");
		const row = rowByCode.get(key) || null;

		let stats;
		if (row) {
			stats = extractLiveStats(row.categories || {});
			if (!stats?.message) stats.message = null;
		} else {
			// ✅ 개선: 더 친절한 메시지
			stats = {
				index: null,
				level: "데이터없음",
				message: "현재 이 지역의 실시간 데이터를 사용할 수 없습니다.",
				population: null,
				dataTime: null,
				timestamp: null,
				roadSpeed: null,
				fcstLevel: null,
				fcstTime: null,
			};
		}

		// 좌표: place.coord → geoCache → 지오코딩
		let coord = p.coord || geoCache[p.name];
		if (!coord || !Number.isFinite(coord.lat) || !Number.isFinite(coord.lng)) {
			try {
				coord = await geocodeWithNaver(p.name);
				geoCache[p.name] = coord;
				saveGeoCache(geoCache);
				await new Promise(r => setTimeout(r, 100));
			} catch (err) {
				console.warn(`Geocoding failed for ${p.name}:`, err.message);
				// 좌표 실패 시 null 유지
			}
		}

		out.push({
			place: p,
			coord: coord || null,
			stats,
			cityData: stats,
			raw: row || null,
			updatedAt: row?.updatedAt || null,
		});
	}

	return out;
}


export default function App() {
	const mapRef = useRef(null);
	const heatRef = useRef(null);
	const overlaysRef = useRef([]);
	const geoCacheRef = useRef(loadGeoCache());
	const datasetRef = useRef([]);
	const placesRef = useRef([]);
	const focusMarkerRef = useRef(null);
	const focusCircleRef = useRef(null);
	const routePathsRef = useRef([]);
	const routeMarkersRef = useRef({ start: null, end: null });
	const placeMarkersRef = useRef([]);
	const culturalEventMarkersRef = useRef([]);
	const culturalEventInfoWindowsRef = useRef([]);
	const busStationMarkersRef = useRef([]);
	const busStationInfoWindowsRef = useRef([]);
	const busRoutePolylinesRef = useRef([]); // 버스 경로 polylines
	const pickModeRef = useRef(null);
	const initialCategoriesRef = useRef(new Set());
	const userOverrideRef = useRef(false);
	const [radius, setRadius] = useState(28);
	const [opacity, setOpacity] = useState(0.78);
	const [mapReady, setMapReady] = useState(false);
	const [loading, setLoading] = useState(true);
	const [status, setStatus] = useState("지도 초기화 중…");
	const [places, setPlaces] = useState([]);
	const [dataset, setDataset] = useState([]);
	const [categories, setCategories] = useState([]);
	const [activeCats, setActiveCats] = useState(new Set());
	const [topItems, setTopItems] = useState([]);
	const [showHeatmap, setShowHeatmap] = useState(true);
	const [summary, setSummary] = useState({ count: 0, totalPopulation: 0, busiest: null, updatedAt: null, });
	const [focusItem, setFocusItem] = useState(null);
	const [routeLabels, setRouteLabels] = useState({ from: "", to: "" });
	const [routes, setRoutes] = useState([]);
	const [routeSource, setRouteSource] = useState(null); // 'osrm', 'naver', 'subway' 등
	const [selectedRouteIndex, setSelectedRouteIndex] = useState(0);
	const [transitList, setTransitList] = useState([]);
	const [aiRecIndex, setAiRecIndex] = useState(null);
	const [aiScores, setAiScores] = useState([]);
	const [culturalEvents, setCulturalEvents] = useState([]);
	const [showDestinationList, setShowDestinationList] = useState(false);
	const [destinationListLimit, setDestinationListLimit] = useState(50);
	const [destinationSearchQuery, setDestinationSearchQuery] = useState('');
	const [showTravelPanel, setShowTravelPanel] = useState(false);
	const [showCulturalMarkers, setShowCulturalMarkers] = useState(false);
	const [filterNearRoute, setFilterNearRoute] = useState(false);
	const [filterFeeCategories, setFilterFeeCategories] = useState(new Set(['all']));
	const [showEventsCommercePanel, setShowEventsCommercePanel] = useState(false);
	const focusInfoWindowRef = useRef(null);

	// 고급 필터 상태
	const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
	const [activeBusinessTypes, setActiveBusinessTypes] = useState(new Set());
	const [activeGenderRatios, setActiveGenderRatios] = useState(new Set());
	const [activeAgeGroups, setActiveAgeGroups] = useState(new Set());
	const [activeResidencyRatios, setActiveResidencyRatios] = useState(new Set());

	// 패널 열고 닫기 상태 (모바일/태블릿에서는 기본적으로 닫힘)
	const [leftPanelOpen, setLeftPanelOpen] = useState(window.innerWidth >= 1024);
	const [rightPanelOpen, setRightPanelOpen] = useState(window.innerWidth >= 1024);


	const suggestionItems = useMemo(() => (
		places.map((p) => ({
			type: "internal",
			name: p.name,
			code: p.code,
			category: p.category,
			coord: p.coord || null,
			lat: p.coord?.lat ?? null,
			lng: p.coord?.lng ?? null,
		}))
	), [places]);

	// 필터링된 문화행사 목록
	const filteredCulturalEvents = useMemo(() => {
		if (!culturalEvents || culturalEvents.length === 0) return [];

		const filtered = culturalEvents.filter(event => {
			// 1. 경로 주변 필터 (경로가 있을 때만 적용)
			if (filterNearRoute && routePathsRef.current.length > 0) {
				const lat = parseFloat(event.LAT);
				const lng = parseFloat(event.LOT);
				if (!lat || !lng || isNaN(lat) || isNaN(lng)) return false;

				// 모든 경로에 대해 확인
				let nearAnyRoute = false;
				let validPolylineCount = 0;

				for (const polyline of routePathsRef.current) {
					try {
						// polyline인지 확인 (getPath 메서드가 있는지)
						if (typeof polyline.getPath !== 'function') {
							console.warn('Invalid polyline object in routePathsRef:', polyline);
							continue;
						}

						const path = polyline.getPath();
						if (path && path.length >= 2) {
							validPolylineCount++;
							if (isNearRoute({ lat, lng }, path, 1500)) {
								nearAnyRoute = true;
								break;
							}
						}
					} catch (e) {
						console.error('Error getting path from polyline:', e, polyline);
						continue;
					}
				}

				if (validPolylineCount === 0) {
					console.warn('No valid polylines found in routePathsRef');
				}

				if (!nearAnyRoute) return false;
			}

			// 2. 이용료 필터
			if (!filterFeeCategories.has('all')) {
				const feeStr = event.USE_FEE || '';
				const parsedFee = parseFeeString(feeStr);
				const category = categorizeFee(parsedFee);

				if (!filterFeeCategories.has(category)) return false;
			}

			return true;
		});

		// 디버깅 로그
		if (filterNearRoute && routePathsRef.current.length > 0) {
			console.log('[필터링] 경로 주변 필터 활성화');
			console.log('[필터링] 전체 문화행사:', culturalEvents.length);
			console.log('[필터링] 경로 polyline 개수:', routePathsRef.current.length);
			console.log('[필터링] 필터링 결과:', filtered.length);
		}

		return filtered;
	}, [culturalEvents, filterNearRoute, filterFeeCategories]);

	// 검색어로 필터링된 문화행사 목록
	const searchedCulturalEvents = useMemo(() => {
		if (!destinationSearchQuery.trim()) return filteredCulturalEvents;

		const query = destinationSearchQuery.toLowerCase().trim();
		return filteredCulturalEvents.filter(event => {
			const title = (event.TITLE || '').toLowerCase();
			const place = (event.PLACE || '').toLowerCase();
			const codename = (event.CODENAME || '').toLowerCase();
			const orgName = (event.ORG_NAME || '').toLowerCase();

			return title.includes(query) ||
				place.includes(query) ||
				codename.includes(query) ||
				orgName.includes(query);
		});
	}, [filteredCulturalEvents, destinationSearchQuery]);

	function clearPlaceMarkers() {
		placeMarkersRef.current.forEach(m => {
			try { m.setMap(null); } catch {}
		});
		placeMarkersRef.current = [];
	}

	function computeWeight(item, maxPopulation, maxIndex) {
		const popWeight = maxPopulation > 0
			? (item.stats.population || 0) / maxPopulation
			: 0;

		const indexWeight = maxIndex > 0
			? (item.stats.index || 0) / maxIndex
			: 0;

		// 혼잡도 바이어스 추가
		const bias = congestionBias(item.stats.level);

		// 가중 평균 (인구 60%, 지수 30%, 바이어스 10%)
		const baseWeight = (popWeight * 0.6) + (indexWeight * 0.3);
		const finalWeight = Math.max(0, Math.min(1, baseWeight + bias));

		return finalWeight;
	}
	// 혼잡 레벨을 색상으로 매핑
	const levelToColor = (level) => {
		const txt = String(level || "").trim();
		if (/약간\s*붐빔/.test(txt)) return "#f97316"; // 주황
		if (/붐빔/.test(txt))        return "#ef4444"; // 빨강
		if (/보통/.test(txt))        return "#f59e0b"; // 노랑
		if (/여유/.test(txt))        return "#22c55e"; // 녹색
		if (/한산|원활/.test(txt))   return "#38bdf8"; // 하늘색
		return "#94a3b8"; // 기본(데이터없음/미매핑)
	};

// 색상을 받아 HTML 마커 콘텐츠 생성
	const buildColorPin = (color, title = "") => `
  <div title="${title}"
       style="
        position:relative;
        width:18px;height:18px;border-radius:80%;
        background:${color};
        border:2px solid #0b1220;
        box-shadow:0 6px 14px rgba(2,6,23,.45), inset 0 0 0 2px rgba(255,255,255,.12);
       ">
    <div style="
      position:absolute; left:4.5px; bottom:-6px;
      width:0;height:0;
      border-left:3px solid transparent;
      border-right:3px solid transparent;
      border-top: 6px solid ${color}; /* noinspection CssInvalidPropertyValue */
    "></div>
  </div>
`;


	const getLevelOpacity = useCallback((level) => {
		const opacityMap = {
			'붐빔': 0.8,
			'약간 붐빔': 0.75,
			'보통': 0.7,
			'여유': 0.65,
			'한산': 0.6,
			'데이터없음': 0.5
		};
		return opacityMap[level] || 0.5;
	}, []);

	const highlightPlace = useCallback((item, { panTo = true } = {}) => {
		console.log('🔍 [highlightPlace] item:', item);
		console.log('🔍 [highlightPlace] item.stats:', item.stats);
		console.log('🔍 [highlightPlace] item.cityData:', item.cityData);


		if (!item || !mapRef.current || !item.coord) return;
		const naver = window.naver;
		const latlng = new naver.maps.LatLng(item.coord.lat, item.coord.lng);
		if (panTo) {
			mapRef.current.panTo(latlng, { duration: 400 });
		}
		if (focusCircleRef.current) focusCircleRef.current.setMap(null);
		focusCircleRef.current = new naver.maps.Circle({
			map: mapRef.current,
			center: latlng,
			radius: 140,
			strokeColor: "#60a5fa",
			strokeOpacity: 0.7,
			strokeWeight: 2,
			fillColor: "#3b82f6",
			fillOpacity: 0.18,
		});
		if (focusMarkerRef.current) focusMarkerRef.current.setMap(null);
		focusMarkerRef.current = new naver.maps.Marker({
			map: mapRef.current,
			position: latlng,
			icon: {
				content: '<div class="focus-pin"></div>',
				anchor: new naver.maps.Point(10, 30),
			},
		});

		// Close previous InfoWindow if exists
		if (focusInfoWindowRef.current) {
			focusInfoWindowRef.current.close();
		}

		// Create InfoWindow content
		const infoWindowId = `focus-info-${item.place?.code || Date.now()}`;
		const population = item.stats?.population ?
			(item.stats.population >= 1000 ?
				`${(item.stats.population / 1000).toFixed(1)}천명` :
				`${item.stats.population}명`) :
			'정보 없음';

		const roadSpeed = item.stats?.roadSpeed ? `${Math.round(item.stats.roadSpeed)} km/h` : null;

		const timestamp = item.stats?.timestamp ?
			(() => {
				const now = new Date();
				const ts = new Date(item.stats.timestamp);
				const diff = Math.floor((now - ts) / 1000 / 60);
				if (diff < 1) return '방금 전';
				if (diff < 60) return `${diff}분 전`;
				const hours = Math.floor(diff / 60);
				if (hours < 24) return `${hours}시간 전`;
				return `${Math.floor(hours / 24)}일 전`;
			})() : '-';

		const infoContent = `
			<div id="${infoWindowId}" style="
				position: relative;
				padding: 17px;
				min-width: 272px;
				max-width: 340px;
				font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
				background: white;
				border-radius: 14px;
				box-shadow: 0 7px 27px rgba(0,0,0,0.15);
			">
				<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
					<span style="
						display: inline-block;
						padding: 3px 10px;
						background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
						color: white;
						border-radius: 10px;
						font-size: 10px;
						font-weight: 600;
					">${item.place?.category || '장소'}</span>
					<button onclick="window.closeFocusInfoWindow()" style="
						border: none;
						background: #f3f4f6;
						color: #6b7280;
						border-radius: 7px;
						padding: 3px 10px;
						cursor: pointer;
						font-size: 10px;
						font-weight: 600;
					">닫기</button>
				</div>

				<h3 style="
					margin: 0 0 14px 0;
					font-size: 17px;
					font-weight: 700;
					color: #111827;
				">${item.place?.name || '알 수 없음'}</h3>

				<div style="
					display: grid;
					grid-template-columns: repeat(2, 1fr);
					gap: 10px;
					margin-bottom: 14px;
				">
					<div>
						<div style="font-size: 10px; color: #6b7280; margin-bottom: 3px;">실시간 혼잡도</div>
						<div style="font-size: 14px; font-weight: 700; color: #111827;">${item.stats?.level || '정보 없음'}</div>
					</div>
					<div>
						<div style="font-size: 10px; color: #6b7280; margin-bottom: 3px;">추정 유동인구</div>
						<div style="font-size: 14px; font-weight: 700; color: #111827;">${population}</div>
					</div>
					<div>
						<div style="font-size: 10px; color: #6b7280; margin-bottom: 3px;">갱신</div>
						<div style="font-size: 12px; font-weight: 600; color: #111827;">${timestamp}</div>
					</div>
					${roadSpeed ? `
					<div>
						<div style="font-size: 10px; color: #6b7280; margin-bottom: 3px;">주변 도로 속도</div>
						<div style="font-size: 12px; font-weight: 600; color: #111827;">${roadSpeed}</div>
					</div>
					` : ''}
				</div>

				${item.stats?.message ? `
				<p style="
					margin: 0 0 14px 0;
					padding: 10px;
					background: #f0f9ff;
					border-left: 3px solid #3b82f6;
					border-radius: 5px;
					font-size: 11px;
					color: #1e40af;
				">${item.stats.message}</p>
				` : ''}

				<div style="
					display: grid;
					grid-template-columns: repeat(3, 1fr);
					gap: 7px;
				">
					<button onclick="window.handleFocusRouteSelection('from', '${item.place?.name}', '${item.place?.code}')" style="
						border: 1px solid #e5e7eb;
						background: white;
						color: #374151;
						border-radius: 7px;
						padding: 8px 7px;
						cursor: pointer;
						font-size: 10px;
						font-weight: 600;
						transition: all 0.2s;
					" onmouseover="this.style.background='#f9fafb'" onmouseout="this.style.background='white'">출발지 지정</button>

					<button onclick="window.handleFocusRouteSelection('to', '${item.place?.name}', '${item.place?.code}')" style="
						border: 1px solid #e5e7eb;
						background: white;
						color: #374151;
						border-radius: 7px;
						padding: 8px 7px;
						cursor: pointer;
						font-size: 10px;
						font-weight: 600;
						transition: all 0.2s;
					" onmouseover="this.style.background='#f9fafb'" onmouseout="this.style.background='white'">도착지 지정</button>

					<button onclick="window.openEventsCommercePanel('${item.place?.code}')" style="
						border: 1px solid #8b5cf6;
						background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
						color: white;
						border-radius: 7px;
						padding: 8px 7px;
						cursor: pointer;
						font-size: 10px;
						font-weight: 600;
						transition: all 0.2s;
						box-shadow: 0 2px 7px rgba(139, 92, 246, 0.3);
					" onmouseover="this.style.transform='translateY(-1px)'; this.style.boxShadow='0 3px 10px rgba(139, 92, 246, 0.4)'" onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 2px 7px rgba(139, 92, 246, 0.3)'">행사·상권</button>
				</div>
			</div>
		`;

		const infoWindow = new naver.maps.InfoWindow({
			content: infoContent,
			backgroundColor: 'transparent',
			borderWidth: 0,
			disableAnchor: false,
			anchorSkew: true,
			pixelOffset: new naver.maps.Point(0, -8)
		});

		infoWindow.open(mapRef.current, focusMarkerRef.current);
		focusInfoWindowRef.current = infoWindow;

		setFocusItem(item);
		console.log('✅ [highlightPlace] InfoWindow 생성 완료');
	}, []);

	const updatePlaceMarkers = useCallback(() => {
		const naver = window.naver;
		const map = mapRef.current;
		if (!naver || !map) return;

		// 기존 마커 싹 정리
		clearPlaceMarkers();

		// Heatmap과 동일한 카테고리 필터 로직 재사용
		const fallbackSet = (!userOverrideRef.current && initialCategoriesRef.current.size)
			? initialCategoriesRef.current
			: null;
		const baseSet = activeCats.size ? activeCats : (fallbackSet || new Set());
		if (!baseSet.size) return;

		// 좌표 가진 항목만 + 고급 필터 적용
		const nodes = dataset.filter(item => {
			const cat = item.place?.category;
			// 1. 기본 필터
			if (!cat || !baseSet.has(cat) || !item.coord?.lat || !item.coord?.lng) {
				return false;
			}

			// 2. 업종 필터
			if (activeBusinessTypes.size > 0) {
				const businesses = item.cityData?.commerce?.businesses || [];

				// "기타"가 선택되었고 데이터가 없으면 통과
				if (businesses.length === 0) {
					if (activeBusinessTypes.has('기타')) {
						// "기타" 선택 시 데이터 없는 경우도 포함
						return true;
					}
					return false;
				}

				const hasMatchingBusiness = businesses.some(business => {
					const category = mapBusinessCategory(business);
					return activeBusinessTypes.has(category);
				});
				if (!hasMatchingBusiness) return false;
			}

			// 3. 성별 비율 필터
			if (activeGenderRatios.size > 0) {
				const demographics = item.cityData?.demographics;
				if (!demographics) return false;
				const genderCategory = getGenderRatioCategory(demographics.male, demographics.female);
				if (!genderCategory || !activeGenderRatios.has(genderCategory)) return false;
			}

			// 4. 연령대 비율 필터
			if (activeAgeGroups.size > 0) {
				const demographics = item.cityData?.demographics;

				// "기타"가 선택되었고 데이터가 없으면 통과
				if (!demographics || !demographics.age) {
					if (activeAgeGroups.has('기타')) {
						// "기타" 선택 시 데이터 없는 경우도 포함
						return true;
					}
					return false;
				}

				const ageCategories = getAgeGroupCategories(demographics);
				const hasMatchingAge = ageCategories.some(ageCategory => activeAgeGroups.has(ageCategory));
				if (!hasMatchingAge) return false;
			}

			// 5. 상주·비상주 비율 필터
			if (activeResidencyRatios.size > 0) {
				const demographics = item.cityData?.demographics;
				if (!demographics) return false;
				const residencyCategory = getResidencyRatioCategory(demographics.resident, demographics.nonResident);
				if (!residencyCategory || !activeResidencyRatios.has(residencyCategory)) return false;
			}

			return true;
		});

		// 마커 생성 + 클릭 시 상자 열기
		nodes.forEach(item => {
			const level = item.stats?.level || "";
			const color = levelToColor(level);
			const marker = new naver.maps.Marker({
				map,
				position: new naver.maps.LatLng(item.coord.lat, item.coord.lng),
				title: `${item.place?.name || ""} · ${level || "정보없음"}`,
				icon: {
					// HtmlIcon: content/size/anchor 지정
					content: buildColorPin(color, `${item.place?.name || ""} · ${level || "정보없음"}`),
					size: new naver.maps.Size(24, 30),         // 핀+꼬리 여유 포함
					anchor: new naver.maps.Point(12, 24),      // 하단 중앙이 좌표에 닿도록
				},
				zIndex: 10
			});
			naver.maps.Event.addListener(marker, 'click', () => {
				highlightPlace(item, { panTo: true });
			});
			placeMarkersRef.current.push(marker);
		});
	}, [dataset, activeCats, highlightPlace, activeBusinessTypes, activeGenderRatios, activeAgeGroups, activeResidencyRatios]);




	const updateHeatmap = useCallback(() => {
		const map = mapRef.current;
		const heat = heatRef.current;
		const naver = window.naver;
		if (!map || !heat || !naver) return;


		const fallbackSet = (!userOverrideRef.current && initialCategoriesRef.current.size)
			? initialCategoriesRef.current
			: null;
		const baseSet = activeCats.size ? activeCats : (fallbackSet || new Set());

		if (!baseSet.size) {
			setTopItems([]);
			setSummary({ count: 0, totalPopulation: 0, busiest: null, updatedAt: null });
			// 그리고 히트맵 레이어는 안전하게 비워서 내림
			heat.setData([]);
			heat.redraw();
			overlaysRef.current.forEach(o => { try { o.setMap(null); } catch {} });
			overlaysRef.current = [];
			heat.setMap(null);
			return;
		}

		const filtered = dataset.filter(item => {
			const cat = item.place?.category;
			// 1. 기본 카테고리 필터
			if (!cat || !baseSet.has(cat) || !item.stats?.level || item.stats.level === '데이터없음') {
				return false;
			}

			// 2. 업종 필터 (activeBusinessTypes가 비어있지 않을 때만 적용)
			if (activeBusinessTypes.size > 0) {
				const businesses = item.cityData?.commerce?.businesses || [];

				// "기타"가 선택되었고 데이터가 없으면 통과
				if (businesses.length === 0) {
					if (activeBusinessTypes.has('기타')) {
						// "기타" 선택 시 데이터 없는 경우도 포함
						return true;
					}
					return false;
				}

				// 업종 데이터를 카테고리로 매핑해서 하나라도 일치하는지 확인
				const hasMatchingBusiness = businesses.some(business => {
					const category = mapBusinessCategory(business);
					return activeBusinessTypes.has(category);
				});
				if (!hasMatchingBusiness) {
					return false;
				}
			}

			// 3. 성별 비율 필터
			if (activeGenderRatios.size > 0) {
				const demographics = item.cityData?.demographics;
				if (!demographics) return false;
				const genderCategory = getGenderRatioCategory(demographics.male, demographics.female);
				if (!genderCategory || !activeGenderRatios.has(genderCategory)) {
					return false;
				}
			}

			// 4. 연령대 비율 필터
			if (activeAgeGroups.size > 0) {
				const demographics = item.cityData?.demographics;

				// "기타"가 선택되었고 데이터가 없으면 통과
				if (!demographics || !demographics.age) {
					if (activeAgeGroups.has('기타')) {
						// "기타" 선택 시 데이터 없는 경우도 포함
						return true;
					}
					return false;
				}

				const ageCategories = getAgeGroupCategories(demographics);
				// 하나라도 일치하면 통과
				const hasMatchingAge = ageCategories.some(ageCategory => activeAgeGroups.has(ageCategory));
				if (!hasMatchingAge) {
					return false;
				}
			}

			// 5. 상주·비상주 비율 필터
			if (activeResidencyRatios.size > 0) {
				const demographics = item.cityData?.demographics;
				if (!demographics) return false;
				const residencyCategory = getResidencyRatioCategory(demographics.resident, demographics.nonResident);
				if (!residencyCategory || !activeResidencyRatios.has(residencyCategory)) {
					return false;
				}
			}

			return true;
		});

		const withBounds = filtered.filter(item =>
			item.place?.bounds && item.place?.heatmapImage
		);
		const withoutBounds = filtered.filter(item =>
			!item.place?.bounds || !item.place?.heatmapImage
		);

		console.log('🔍 필터링 결과:', {
			전체: filtered.length,
			bounds있음: withBounds.length,
			bounds없음: withoutBounds.length
		});

		const totalPopulation = filtered.reduce((sum, item) => sum + (item.stats?.population || 0), 0);
		const busiest = filtered.reduce((max, item) => {
			if (!max || (item.stats?.population || 0) > (max.stats?.population || 0)) {
				return item;
			}
			return max;
		}, null);

		const latestUpdate = filtered.reduce((latest, item) => {
			const itemTime = item.stats?.timestamp || item.updatedAt;
			if (!latest || (itemTime && itemTime > latest)) {
				return itemTime;
			}
			return latest;
		}, null);

		setSummary({
			count: filtered.length,
			totalPopulation,
			busiest,
			updatedAt: latestUpdate
		});

		setTopItems(filtered.slice(0, 10));
		heat.setOptions({ radius, opacity });
		if (!showHeatmap) {
			// 지도에서 레이어만 내림 (데이터/TopList 유지)
			heat.setData([]);
			heat.redraw();
			heat.setMap(null);                            // HeatMap 제거, 올바른 사용
			overlaysRef.current.forEach(o => { try { o.setMap(null); } catch {} }); // GroundOverlay 제거
			overlaysRef.current = [];
			return;
		}

		heat.setMap(map);
		overlaysRef.current.forEach(o => { try { o.setMap(null); } catch {} });
		overlaysRef.current = [];

		withBounds.forEach(item => {
			try {
				const { bounds, heatmapImage } = item.place;
				const level = item.stats.level;

				const latLngBounds = new naver.maps.LatLngBounds(
					new naver.maps.LatLng(bounds.minY, bounds.minX),
					new naver.maps.LatLng(bounds.maxY, bounds.maxX)
				);

				const overlayOpacity = getLevelOpacity(level);

				// 히트맵 URL을 동적 API로 변환 (보안 강화)
				const originalUrl = heatmapImage.url;
				const code = item.place.code;
				const protectedUrl = `${API_BASE}/api/heatmap?code=${code}`;

				const groundOverlay = new naver.maps.GroundOverlay(
					protectedUrl, latLngBounds, { opacity: overlayOpacity, clickable: true }
				);

				naver.maps.Event.addListener(groundOverlay, 'click', () => {
					highlightPlace(item, { panTo: true });
				});

				groundOverlay.setMap(map);
				overlaysRef.current.push(groundOverlay);

				console.log('✅ GroundOverlay 생성:', item.place.name);
			} catch (e) {
				console.error('❌ GroundOverlay 생성 실패:', item.place.name, e);
			}
		});

		if (withoutBounds.length > 0) {
			const maxPopulation = Math.max(...filtered.map(x => x.stats.population || 0), 1);
			const maxIndex = Math.max(...filtered.map(x => x.stats.index || 0), 1);

			const points = withoutBounds
				.filter(item => item.coord?.lat && item.coord?.lng)
				.map(item => {
					const w = Math.max(0.1, Math.min(1, computeWeight(item, maxPopulation, maxIndex)));
					// 좌표 순서 혼동 방지를 위해 WeightedLocation 클래스를 사용 (lat, lng, weight)
					return new naver.maps.visualization.WeightedLocation(item.coord.lat, item.coord.lng, w);
				});
			heat.setData(points);                   // setData 후
			heat.redraw();
		} else  {
			heat.setData([]); heat.redraw();
		}
	}, [dataset, activeCats, showHeatmap, radius, opacity, getLevelOpacity, activeBusinessTypes, activeGenderRatios, activeAgeGroups, activeResidencyRatios]);




	const ensureSelectionCoordinate = useCallback(async (selection) => {
		const name = typeof selection === "string" ? selection : selection?.name;
		if (!name) throw new Error("장소 이름이 필요합니다.");
		if (typeof selection === "object" && selection?.coord && Number.isFinite(selection.coord.lat) && Number.isFinite(selection.coord.lng)) {
			return selection.coord;
		}

		const cached = geoCacheRef.current[name];
		if (cached && Number.isFinite(cached.lat) && Number.isFinite(cached.lng)) {
			return cached;
		}
		// 이전 실패 흔적 제거
		if (cached) {
			delete geoCacheRef.current[name];
		}
		const byCode = typeof selection === "object" && selection?.code
			? placesRef.current.find((p) => p.code === selection.code)
			: null;
		const fromPlace = byCode || placesRef.current.find((p) => p.name === name);
		if (fromPlace?.coord && Number.isFinite(fromPlace.coord.lat) && Number.isFinite(fromPlace.coord.lng)) {
			geoCacheRef.current[name] = fromPlace.coord;
			saveGeoCache(geoCacheRef.current);
			return fromPlace.coord;
		}
		let geocoded;
		try {
			geocoded = await geocodeWithNaver(name);
		} catch {
			// 최종 안전망: OSM
			const url = `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(name)}`;
			const r = await fetch(url);
			if (r.ok) {
				const j = await r.json();
				const it = j?.[0];
				if (it) geocoded = { lat: Number(it.lat), lng: Number(it.lon) };
			}
			if (!geocoded) throw new Error("지오코딩 결과 없음");
		}
		geoCacheRef.current[name] = geocoded;
		saveGeoCache(geoCacheRef.current);
		return geocoded;
	}, []);

	const adoptRoutePoint = useCallback(async (type, selection) => {
		try {
			const coord = await ensureSelectionCoordinate(selection);
			if (type === "from") {
				routeMarkersRef.current.start?.setMap(null);
				routeMarkersRef.current.start = new window.naver.maps.Marker({
					map: mapRef.current,
					position: new window.naver.maps.LatLng(coord.lat, coord.lng),
					icon: {
						content: '<div class="route-marker route-start">출발</div>',
						anchor: new window.naver.maps.Point(20, 20),
					},
				});
				setRouteLabels((prev) => ({ ...prev, from: selection?.name || selection }));
			} else {
				routeMarkersRef.current.end?.setMap(null);
				routeMarkersRef.current.end = new window.naver.maps.Marker({
					map: mapRef.current,
					position: new window.naver.maps.LatLng(coord.lat, coord.lng),
					icon: {
						content: '<div class="route-marker route-end">도착</div>',
						anchor: new window.naver.maps.Point(20, 20),
					},
				});
				setRouteLabels((prev) => ({ ...prev, to: selection?.name || selection }));
			}
			return coord;
		} catch (err) {
			setStatus(err.message || "지오코딩 실패");
			throw err;
		}
	}, [ensureSelectionCoordinate]);

	// Setup window functions for InfoWindow button handlers
	useEffect(() => {
		window.closeFocusInfoWindow = () => {
			if (focusInfoWindowRef.current) {
				focusInfoWindowRef.current.close();
			}
			if (focusMarkerRef.current) {
				focusMarkerRef.current.setMap(null);
			}
			if (focusCircleRef.current) {
				focusCircleRef.current.setMap(null);
			}
			setFocusItem(null);
		};

		window.handleFocusRouteSelection = async (type, name, code) => {
			const selection = { name, code };
			try {
				await adoptRoutePoint(type, selection);
			} catch (err) {
				console.error('Route selection error:', err);
			}
		};

		window.openEventsCommercePanel = (code) => {
			setShowEventsCommercePanel(true);
		};

		return () => {
			delete window.closeFocusInfoWindow;
			delete window.handleFocusRouteSelection;
			delete window.openEventsCommercePanel;
		};
	}, [adoptRoutePoint]);

	const drawRoute = useCallback(async () => {
		setStatus("차량 경로를 계산 중입니다…");
		try {
			const fromCoord = routeMarkersRef.current.start?.getPosition();
			const toCoord = routeMarkersRef.current.end?.getPosition();
			if (!fromCoord || !toCoord) {
				throw new Error("출발지와 도착지를 모두 선택하세요.");
			}
			const from = `${fromCoord.y},${fromCoord.x}`;
			const to = `${toCoord.y},${toCoord.x}`;

			// Request multiple route alternatives (최대 5개 요청)
			const res = await fetch(`${API_BASE}/route?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}&alternatives=true&alternatives_count=5`);
			if (!res.ok) throw new Error("경로 API 오류");
			const json = await res.json();

			if (!json?.routes || !Array.isArray(json.routes) || json.routes.length === 0) {
				throw new Error("추천 경로를 찾을 수 없습니다");
			}

			console.log(`API returned ${json.routes.length} routes`);

			// Clear existing route polylines
			routePathsRef.current.forEach(polyline => {
				try { polyline.setMap(null); } catch {}
			});
			routePathsRef.current = [];

			let allRoutes = [...json.routes];

			// API가 5개 미만의 경로를 반환하면 추가 경로 생성
			if (allRoutes.length < 5) {
				try {
					// 약간 다른 출발지/도착지로 추가 경로 요청 (더 큰 offset과 다양한 방향)
					const alternativeOffsets = [
						{ fromLatOffset: 0.005, fromLngOffset: 0, toLatOffset: 0, toLngOffset: 0, label: '출발 북쪽' },
						{ fromLatOffset: -0.005, fromLngOffset: 0, toLatOffset: 0, toLngOffset: 0, label: '출발 남쪽' },
						{ fromLatOffset: 0, fromLngOffset: 0.005, toLatOffset: 0, toLngOffset: 0, label: '출발 동쪽' },
						{ fromLatOffset: 0, fromLngOffset: -0.005, toLatOffset: 0, toLngOffset: 0, label: '출발 서쪽' },
						{ fromLatOffset: 0, fromLngOffset: 0, toLatOffset: 0.005, toLngOffset: 0, label: '도착 북쪽' },
						{ fromLatOffset: 0, fromLngOffset: 0, toLatOffset: -0.005, toLngOffset: 0, label: '도착 남쪽' },
						{ fromLatOffset: 0.003, fromLngOffset: 0.003, toLatOffset: 0, toLngOffset: 0, label: '출발 북동쪽' },
						{ fromLatOffset: -0.003, fromLngOffset: -0.003, toLatOffset: 0, toLngOffset: 0, label: '출발 남서쪽' },
					];

					for (let i = 0; i < alternativeOffsets.length && allRoutes.length < 5; i++) {
						const offset = alternativeOffsets[i];
						const altFrom = `${fromCoord.y + offset.fromLatOffset},${fromCoord.x + offset.fromLngOffset}`;
						const altTo = `${toCoord.y + offset.toLatOffset},${toCoord.x + offset.toLngOffset}`;

						console.log(`Trying alternative route ${i + 1} (${offset.label})`);

						const altRes = await fetch(
							`${API_BASE}/route?from=${encodeURIComponent(altFrom)}&to=${encodeURIComponent(altTo)}&alternatives=true`
						);

						if (altRes.ok) {
							const altJson = await altRes.json();

							// alternatives=true이므로 여러 경로가 올 수 있음
							const newRoutes = altJson?.routes || [];

							for (const newRoute of newRoutes) {
								if (allRoutes.length >= 5) break;

								// 새 경로가 기존 경로와 충분히 다른지 확인 (거리 차이로 판단)
								const isDifferent = allRoutes.every(existingRoute => {
									const distanceDiff = Math.abs(newRoute.distance - existingRoute.distance);
									const relativeDiff = distanceDiff / existingRoute.distance;
									return relativeDiff > 0.02; // 2% 이상 차이나야 함 (더 완화)
								});

								if (isDifferent) {
									allRoutes.push(newRoute);
									console.log(`✓ Added alternative route from ${offset.label}: distance=${newRoute.distance}m, duration=${newRoute.duration}s`);
								} else {
									console.log(`✗ Skipped route from ${offset.label} (too similar: ${newRoute.distance}m)`);
								}
							}
						} else {
							console.log(`Alternative route ${i + 1} API call failed: ${altRes.status}`);
						}

						// 너무 많은 API 호출 방지를 위한 짧은 대기
						if (i < alternativeOffsets.length - 1 && allRoutes.length < 5) {
							await new Promise(resolve => setTimeout(resolve, 50));
						}
					}
				} catch (err) {
					console.warn('Failed to generate additional routes:', err);
				}
			}

			console.log(`Total routes after augmentation: ${allRoutes.length}`);

			// Process up to 5 routes
			const routeColors = ["#38bdf8", "#f59e0b", "#a78bfa", "#10b981", "#ef4444"]; // Blue, Orange, Purple, Green, Red
			const processedRoutes = allRoutes.slice(0, 5).map((route, idx) => {
				const coords = route?.geometry?.coordinates;
				if (!Array.isArray(coords) || !coords.length) return null;

				const path = coords.map(([lng, lat]) => new window.naver.maps.LatLng(lat, lng));
				const isSelected = idx === 0;

				const polyline = new window.naver.maps.Polyline({
					map: mapRef.current,
					path,
					strokeColor: routeColors[idx],
					strokeWeight: isSelected ? 6 : 4,
					strokeOpacity: isSelected ? 0.9 : 0.6,
					zIndex: isSelected ? 100 : 50 - idx,
				});

				// Add click event to select this route
				window.naver.maps.Event.addListener(polyline, 'click', () => {
					setSelectedRouteIndex(idx);
				});

				routePathsRef.current.push(polyline);

				return {
					index: idx,
					distance: route.distance,
					duration: route.duration,
					path,
					polyline,
					color: routeColors[idx]
				};
			}).filter(Boolean);

			if (processedRoutes.length === 0) {
				throw new Error("추천 경로를 찾을 수 없습니다");
			}

			// Fit bounds to show all routes
			const allPoints = processedRoutes.flatMap(r => r.path);
			const bounds = allPoints.reduce(
				(b, point) => (b.extend(point), b),
				new window.naver.maps.LatLngBounds(allPoints[0], allPoints[0])
			);
			mapRef.current.fitBounds(bounds, { top: 120, bottom: 140, left: 80, right: 80 });

			setRoutes(processedRoutes);
			setRouteSource('osrm'); // OSRM 경로임을 표시
			setSelectedRouteIndex(0);
			setStatus(`${processedRoutes.length}개의 차량 경로를 불러왔습니다.`);
		} catch (err) {
			console.error(err);
			setStatus(err.message || "경로 계산 실패");
			setRoutes([]);
			setSelectedRouteIndex(0);
		}
	}, []);

	// 네이버 Direction5 API를 이용한 차량 경로 찾기
	const drawNaverDirection5 = useCallback(async () => {
		setStatus("네이버 Direction5로 차량 경로를 계산 중입니다…");
		try {
			const fromCoord = routeMarkersRef.current.start?.getPosition();
			const toCoord = routeMarkersRef.current.end?.getPosition();
			if (!fromCoord || !toCoord) {
				throw new Error("출발지와 도착지를 모두 선택하세요.");
			}
			const from = `${fromCoord.y},${fromCoord.x}`;
			const to = `${toCoord.y},${toCoord.x}`;

			// Clear existing route polylines
			routePathsRef.current.forEach(polyline => {
				try { polyline.setMap(null); } catch {}
			});
			routePathsRef.current = [];

			// 네이버 Direction5 API는 3가지 옵션을 제공: trafast (실시간 빠른길), tracomfort (실시간 편한길), traoptimal (실시간 최적)
			const options = ['trafast', 'tracomfort', 'traoptimal'];
			const optionNames = {
				'trafast': '빠른길',
				'tracomfort': '편한길',
				'traoptimal': '최적경로'
			};
			const routeColors = ["#38bdf8", "#f59e0b", "#a78bfa"]; // Blue, Orange, Purple

			const processedRoutes = [];

			let lastError = null;

			for (let i = 0; i < options.length; i++) {
				const option = options[i];

				try {
					const url = `${API_BASE}/naver/direction5?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}&option=${option}`;
					console.log(`Fetching ${option} route from:`, url);
					const res = await fetch(url);

					// 더 나은 오류 처리
					if (!res.ok) {
						const errorText = await res.text();
						console.error(`HTTP ${res.status} for ${option}:`, errorText);
						lastError = new Error(`API returned ${res.status}`);
						continue;
					}

					const contentType = res.headers.get('content-type');
					if (!contentType || !contentType.includes('application/json')) {
						const text = await res.text();
						console.error(`Invalid content-type for ${option}:`, contentType, text.substring(0, 200));
						lastError = new Error(`API endpoint not found. 새로운 기능이 아직 배포되지 않았습니다.`);
						continue;
					}

					const json = await res.json();
					console.log(`${option} response:`, json);
					if (json.error) {
						console.error(`API error for ${option}:`, json.error);
						lastError = new Error(json.error);
						continue;
					}

					if (json.code !== 0 || !json.route) {
						console.warn(`Invalid route data for ${option}:`, json);
						continue;
					}
					// 네이버 Direction5는 옵션별로 경로를 반환 (예: route.trafast[0])
					const routeData = json.route[option]?.[0];
					if (!routeData) {
						console.warn(`No route data for ${option}`);
						continue;
					}
					const path = routeData.path.map(([lng, lat]) => new window.naver.maps.LatLng(lat, lng));
					const isSelected = i === 0;
					const polyline = new window.naver.maps.Polyline({
						map: mapRef.current,
						path,
						strokeColor: routeColors[i],
						strokeWeight: isSelected ? 6 : 4,
						strokeOpacity: isSelected ? 0.9 : 0.6,
						zIndex: isSelected ? 100 : 50 - i,
					});

					// Add click event to select this route
					window.naver.maps.Event.addListener(polyline, 'click', () => {
						setSelectedRouteIndex(i);
					});
					routePathsRef.current.push(polyline);
					processedRoutes.push({
						index: i,
						distance: routeData.summary.distance,
						duration: Math.round(routeData.summary.duration / 1000), // 밀리초 -> 초
						path,
						polyline,
						color: routeColors[i],
						optionName: optionNames[option]
					});
				} catch (err) {
					console.error(`Failed to fetch ${option} route:`, err);
					lastError = err;
				}
			}

			if (processedRoutes.length === 0) {
				throw lastError || new Error("추천 경로를 찾을 수 없습니다. 네이버 Direction5 API 엔드포인트가 배포되지 않았거나 환경 변수가 설정되지 않았을 수 있습니다.");
			}


			// Fit bounds to show all routes
			const allPoints = processedRoutes.flatMap(r => r.path);
			const bounds = allPoints.reduce(
				(b, point) => (b.extend(point), b),
				new window.naver.maps.LatLngBounds(allPoints[0], allPoints[0])
			);
			mapRef.current.fitBounds(bounds, { top: 120, bottom: 140, left: 80, right: 80 });
			setRoutes(processedRoutes);
			setRouteSource('naver'); // 네이버 경로임을 표시 (AI 추천 없음)
			setSelectedRouteIndex(0);
			setStatus(`${processedRoutes.length}개의 네이버 차량 경로를 불러왔습니다.`);
		} catch (err) {
			console.error('drawNaverDirection5 error:', err);
			setSelectedRouteIndex(0);
		}
	}, []);

	// 경로 특성을 분석하여 사용자 맞춤 추천 생성
	const analyzeRouteCharacteristics = useCallback((route, allRoutes, index) => {
		const distances = allRoutes.map(r => r.distance);
		const durations = allRoutes.map(r => r.duration);
		const speeds = allRoutes.map(r => (r.distance / r.duration * 3.6)); // km/h

		const minDistance = Math.min(...distances);
		const minDuration = Math.min(...durations);
		const maxSpeed = Math.max(...speeds);

		const currentSpeed = route.distance / route.duration * 3.6;
		const distanceRank = distances.filter(d => d < route.distance).length + 1;
		const durationRank = durations.filter(d => d < route.duration).length + 1;

		let recommendation = "";
		let userType = "";
		let icon = "";

		// 최단 거리 경로
		if (route.distance === minDistance) {
			icon = "🎯";
			userType = "연료비 절약 선호자";
			recommendation = `가장 짧은 거리(${(route.distance / 1000).toFixed(1)}km)를 자랑하는 경로입니다. 연료비를 아끼고 싶거나 단순한 경로를 선호하는 분께 추천합니다.`;
		}
		// 최단 시간 경로
		else if (route.duration === minDuration) {
			icon = "⚡";
			userType = "빠른 이동 선호자";
			recommendation = `가장 빠른 경로(${Math.round(route.duration / 60)}분)입니다. 시간이 촉박하거나 빠른 이동을 최우선으로 하는 분께 적합합니다.`;
		}
		// 균형잡힌 경로 (거리도 시간도 중간)
		else if (distanceRank <= 3 && durationRank <= 3) {
			icon = "⚖️";
			userType = "균형 추구자";
			recommendation = `거리와 시간의 균형이 잘 잡힌 경로입니다. 안정적이고 예측 가능한 이동을 원하는 분께 추천합니다.`;
		}
		// 고속 경로 (평균 속도가 높음)
		else if (currentSpeed >= maxSpeed * 0.9) {
			icon = "🏎️";
			userType = "고속도로 선호자";
			recommendation = `평균 속도 ${currentSpeed.toFixed(0)}km/h로 원활한 주행이 가능한 경로입니다. 고속도로나 넓은 도로를 선호하는 분께 적합합니다.`;
		}
		// 여유로운 경로 (시간은 좀 걸리지만 안정적)
		else if (durationRank >= 4) {
			icon = "🌿";
			userType = "여유로운 드라이브 선호자";
			recommendation = `시간은 조금 더 걸리지만 한적하고 안정적인 경로입니다. 여유로운 드라이브를 즐기거나 복잡한 교통을 피하고 싶은 분께 추천합니다.`;
		}
		// 대안 경로
		else {
			icon = "🔄";
			userType = "대안 탐색자";
			recommendation = `주요 경로의 대안으로 활용 가능한 경로입니다. 교통 상황에 따라 선택할 수 있는 백업 옵션입니다.`;
		}

		return {
			icon,
			userType,
			recommendation,
			stats: {
				distanceRank,
				durationRank,
				speed: currentSpeed.toFixed(0)
			}
		};
	}, []);

	// 후보 경로들을 Lambda로 보내 점수/이유를 받아오는 함수
	const rankRoutesWithAI = useCallback(async (rs) => {
		if (!Array.isArray(rs) || rs.length === 0) {
			setAiRecIndex(null);
			setAiScores([]);
			return;
		}

		try {
			console.log("🤖 AI 모델 추론 시작...");

			// 현재 시간 정보
			const now = new Date();
			const hour = now.getHours();
			const dayOfWeek = now.getDay();
			const isWeekend = dayOfWeek === 0 || dayOfWeek === 6;
			const isRushHour = (hour >= 7 && hour <= 9) || (hour >= 17 && hour <= 19);

			// 날짜 형식: YYYYMMDD
			const stnd_dt = now.toISOString().split('T')[0].replace(/-/g, '');
			const time_cd = hour + 1;

			// 🔥 각 경로에 링크 데이터 생성 (핵심 변경!)
			const routesWithLinks = rs.map((route, i) => {
				const links = generateLinksFromRoute(route);

				console.log(`📍 경로 ${i + 1}: ${links.length}개 링크 생성`);
				console.log(`   거리: ${(route.distance/1000).toFixed(1)}km, 시간: ${Math.round(route.duration/60)}분`);
				console.log(`   평균속도: ${((route.distance/1000)/(route.duration/3600)).toFixed(1)}km/h`);

				return {
					id: i,
					distance: route.distance,
					duration: route.duration,
					links: links,  // 🔥 AI 모델용 링크 데이터
					optionName: route.optionName || `경로 ${i + 1}`
				};
			});

			// Lambda payload 구성
			const payload = {
				mode: "rank_routes",
				stnd_dt: stnd_dt,      // 🔥 날짜 추가
				time_cd: time_cd,       // 🔥 시간 추가
				context: {
					currentTime: now.toISOString(),
					hour: hour,
					isWeekend: isWeekend,
					isRushHour: isRushHour,
					dayOfWeek: dayOfWeek
				},
				routes: routesWithLinks  // 🔥 링크 포함
			};

			console.log("🚀 Lambda AI에 전송:", {
				날짜: stnd_dt,
				시간: time_cd,
				경로수: routesWithLinks.length,
				총링크수: routesWithLinks.reduce((sum, r) => sum + r.links.length, 0)
			});

			// 🤖 Lambda AI 모델 호출
			const ai = await infer(payload);
			console.log("✅ AI 응답:", ai);

			// 추천 경로 ID
			const bestId = Number(ai?.recommendation?.id ?? ai?.best?.id);
			console.log("🏆 AI 추천 경로:", bestId);

			setAiRecIndex(Number.isFinite(bestId) ? bestId : null);

			// 경로별 분석 결과
			const scores = (ai.routes || ai.scores || []).map(s => {
				if (s.pred_time_sec) {
					console.log(`   경로 ${s.id}: AI 예상 ${Math.round(s.pred_time_sec/60)}분`);
				}
				if (s.avg_speed) {
					console.log(`      평균 ${s.avg_speed.toFixed(1)}km/h`);
				}
				if (s.slow10 > 0) {
					console.log(`      ⚠️ 느린 구간 ${s.slow10}개 (< 10km/h)`);
				}
				if (s.slow20 > 0) {
					console.log(`      ⚠️ 주의 구간 ${s.slow20}개 (< 20km/h)`);
				}

				return {
					id: Number(s.id),
					reason: s.reason || "",
					pred_time_sec: s.pred_time_sec,
					avg_speed: s.avg_speed,
					slow10: s.slow10,
					slow20: s.slow20
				};
			});

			setAiScores(scores);

			// AI 추천 경로 자동 선택
			if (Number.isFinite(bestId)) {
				setSelectedRouteIndex(bestId);
			}

			console.log("✅ AI 경로 분석 완료!");

		} catch (e) {
			console.error("❌ AI 분석 실패:", e);
			console.error("상세:", e.message);
			setAiRecIndex(null);
			setAiScores([]);
		}
	}, [setSelectedRouteIndex]);

// OSRM 경로일 때만 자동으로 AI 랭킹 요청
	useEffect(() => {
		if (routes && routes.length > 0 && routeSource === 'osrm') {
			rankRoutesWithAI(routes);
		} else {
			setAiRecIndex(null);
			setAiScores([]);
		}
	}, [routes, routeSource, rankRoutesWithAI]);

	// 문화행사 마커 제거 함수
	function clearCulturalEventMarkers() {
		culturalEventMarkersRef.current.forEach(m => {
			try { m.setMap(null); } catch {}
		});
		culturalEventMarkersRef.current = [];

		culturalEventInfoWindowsRef.current.forEach(iw => {
			try { iw.close(); } catch {}
		});
		culturalEventInfoWindowsRef.current = [];

		// 전역 InfoWindow 객체 정리
		if (window.culturalInfoWindows) {
			window.culturalInfoWindows = {};
		}
	}

	// 문화행사 데이터 로드 함수
	const loadCulturalEvents = useCallback(async () => {
		if (culturalEvents.length > 0) return; // 이미 로드된 경우 스킵

		try {
			setStatus("문화행사 정보를 불러오는 중입니다…");
			const culturalRes = await fetch(`${API_BASE}/seoul/cultural-events?start=1&end=1000`);
			if (!culturalRes.ok) throw new Error("문화행사 API 오류");
			const culturalData = await culturalRes.json();

			const events = culturalData.events || [];
			setCulturalEvents(events);
			setStatus(`문화행사 ${events.length}개를 불러왔습니다.`);
			return events;
		} catch (err) {
			console.error("문화행사 로드 실패:", err);
			setStatus("문화행사 정보를 불러오지 못했습니다.");
			return [];
		}
	}, [culturalEvents.length]);

	// 문화행사 마커 생성 함수
	const createCulturalMarkers = useCallback((events) => {
		if (!mapRef.current || !window.naver || !events || events.length === 0) return;

		// 기존 마커 제거
		clearCulturalEventMarkers();

		// 전역 InfoWindow 저장소 초기화
		if (!window.culturalInfoWindows) {
			window.culturalInfoWindows = {};
		}

		events.forEach((event, eventIndex) => {
			const lat = parseFloat(event.LAT);
			const lng = parseFloat(event.LOT);

			if (!lat || !lng || isNaN(lat) || isNaN(lng)) return;

			// 마커 생성
			const markerContent = `
				<div style="
					position: relative;
					width: 20px;
					height: 20px;
					border-radius: 50%;
					background: linear-gradient(135deg, #ec4899, #8b5cf6);
					border: 2px solid #fff;
					box-shadow: 0 3px 10px rgba(236, 72, 153, 0.5);
					cursor: pointer;
				">
					<div style="
						position: absolute;
						top: 50%;
						left: 50%;
						transform: translate(-50%, -50%);
						color: white;
						font-size: 12px;
						font-weight: bold;
					">🎭</div>
				</div>
			`;

			const marker = new window.naver.maps.Marker({
				position: new window.naver.maps.LatLng(lat, lng),
				map: showCulturalMarkers ? mapRef.current : null,
				icon: {
					content: markerContent,
					anchor: new window.naver.maps.Point(10, 10)
				},
				zIndex: 200
			});

			// 정보창 생성
			const imgUrl = event.MAIN_IMG || event.THUMBNAIL || event.IMGURL || '';
			const infoWindowId = `cultural-info-${eventIndex}`;

			const infoWindow = new window.naver.maps.InfoWindow({
				content: `
					<div id="${infoWindowId}" style="
						position: relative;
						padding: 14px;
						min-width: 238px;
						max-width: 340px;
						font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
						background: white;
						border-radius: 10px;
						box-shadow: 0 3px 17px rgba(0,0,0,0.15);
					">
						<button
							onclick="window.culturalInfoWindows['${infoWindowId}']?.close()"
							style="
								position: absolute;
								top: 10px;
								right: 10px;
								width: 24px;
								height: 24px;
								border: none;
								background: #f3f4f6;
								color: #6b7280;
								border-radius: 50%;
								cursor: pointer;
								font-size: 15px;
								font-weight: bold;
								display: flex;
								align-items: center;
								justify-content: center;
								transition: all 0.2s;
								padding: 0;
								line-height: 1;
								z-index: 10;
							"
							onmouseover="this.style.background='#e5e7eb'; this.style.color='#374151';"
							onmouseout="this.style.background='#f3f4f6'; this.style.color='#6b7280';"
						>×</button>
						${imgUrl ? `
							<img src="${imgUrl}"
								style="width: 100%; height: 136px; object-fit: cover; border-radius: 7px; margin-bottom: 10px;"
								onerror="this.style.display='none'"
							/>
						` : ''}
						<h3 style="
							margin: 0 0 7px 0;
							padding-right: 24px;
							font-size: 14px;
							font-weight: 700;
							color: #1f2937;
							line-height: 1.4;
						">${event.TITLE || '제목 없음'}</h3>
						<div style="
							display: flex;
							flex-direction: column;
							gap: 7px;
							font-size: 11px;
							color: #6b7280;
							margin-bottom: 10px;
						">
							${event.CODENAME ? `
								<div style="display: flex; align-items: center; gap: 5px;">
									<span style="font-weight: 600; color: #374151;">🎨 카테고리</span>
									<span>${event.CODENAME}</span>
								</div>
							` : ''}
							${event.DATE ? `
								<div style="display: flex; align-items: center; gap: 5px;">
									<span style="font-weight: 600; color: #374151;">📅 기간</span>
									<span>${event.DATE}</span>
								</div>
							` : ''}
							${event.PLACE ? `
								<div style="display: flex; align-items: center; gap: 5px;">
									<span style="font-weight: 600; color: #374151;">📍 장소</span>
									<span>${event.PLACE}</span>
								</div>
							` : ''}
							${event.ORG_NAME ? `
								<div style="display: flex; align-items: center; gap: 5px;">
									<span style="font-weight: 600; color: #374151;">🏢 주최</span>
									<span>${event.ORG_NAME}</span>
								</div>
							` : ''}
							${event.USE_TRGT ? `
								<div style="display: flex; align-items: center; gap: 5px;">
									<span style="font-weight: 600; color: #374151;">👥 대상</span>
									<span>${event.USE_TRGT}</span>
								</div>
							` : ''}
							${event.USE_FEE ? `
								<div style="display: flex; align-items: center; gap: 5px;">
									<span style="font-weight: 600; color: #374151;">💰 이용료</span>
									<span>${event.USE_FEE}</span>
								</div>
							` : ''}
						</div>
						${event.ORG_LINK ? `
							<a href="${event.ORG_LINK}" target="_blank" rel="noopener noreferrer"
								style="
									display: inline-block;
									padding: 7px 14px;
									background: #3b82f6;
									color: white;
									text-decoration: none;
									border-radius: 5px;
									font-size: 11px;
									font-weight: 600;
									transition: background 0.2s;
								"
								onmouseover="this.style.background='#2563eb'"
								onmouseout="this.style.background='#3b82f6'"
							>자세히 보기 →</a>
						` : ''}
					</div>
				`,
				borderWidth: 0,
				disableAnchor: true,
				backgroundColor: 'transparent',
				pixelOffset: new window.naver.maps.Point(0, -10)
			});

			// InfoWindow를 전역 객체에 저장
			window.culturalInfoWindows[infoWindowId] = infoWindow;

			// 마커 클릭 이벤트
			window.naver.maps.Event.addListener(marker, 'click', () => {
				// 다른 InfoWindow 닫기
				culturalEventInfoWindowsRef.current.forEach(iw => {
					try { iw.close(); } catch {}
				});
				infoWindow.open(mapRef.current, marker);
			});

			culturalEventMarkersRef.current.push(marker);
			culturalEventInfoWindowsRef.current.push(infoWindow);
		});

		console.log(`문화행사 마커 ${culturalEventMarkersRef.current.length}개 생성 완료`);
	}, [showCulturalMarkers]);

	// 버스 정류장 마커 제거 함수
	function clearBusStationMarkers() {
		busStationMarkersRef.current.forEach(m => {
			try { m.setMap(null); } catch {}
		});
		busStationMarkersRef.current = [];

		busStationInfoWindowsRef.current.forEach(iw => {
			try { iw.close(); } catch {}
		});
		busStationInfoWindowsRef.current = [];

		// 버스 경로 polylines 제거
		busRoutePolylinesRef.current.forEach(p => {
			try { p.setMap(null); } catch {}
		});
		busRoutePolylinesRef.current = [];

		// 전역 InfoWindow 객체 정리
		if (window.busStationInfoWindows) {
			window.busStationInfoWindows = {};
		}
	}

	// 버스 도착 정보 API 호출 헬퍼
	async function fetchBusArrival(stId, busRouteId, ord) {
		try {
			const url = `${API_BASE}/seoul/bus-arrival?stId=${stId}&busRouteId=${busRouteId}&ord=${ord}`;
			console.log('[FETCH BUS ARRIVAL] Calling:', url);

			const response = await fetch(url);

			if (!response.ok) {
				console.error(`[BUS ARRIVAL] API Error: ${response.status}`);
				return null;
			}

			const data = await response.json();
			console.log('[FETCH BUS ARRIVAL] Response:', data);

			if (!data.success || !data.hasData) {
				console.warn('[FETCH BUS ARRIVAL] No data:', data);
				return null;
			}

			return data;
		} catch (error) {
			console.error(`[BUS ARRIVAL] Fetch error:`, error);
			return null;
		}
	}

	// 버스 도착 정보 HTML 생성
	function createBusArrivalHTML(route, arrivalData) {
		const busTypes = ['🚌 일반버스', '♿ 저상버스', '🚍 굴절버스'];

		if (!arrivalData || !arrivalData.firstBus) {
			// 도착 정보 없음
			return `
				<div style="background:linear-gradient(135deg,#f1f5f9,#e2e8f0);border:1px solid #cbd5e1;border-radius:10px;padding:16px;margin-bottom:8px;text-align:center">
					<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;padding-bottom:8px;border-bottom:2px dashed #cbd5e1">
						<span style="padding:6px 12px;background:linear-gradient(135deg,#64748b,#475569);color:white;border-radius:8px;font-size:16px;font-weight:800;box-shadow:0 2px 6px rgba(100,116,139,0.3)">${route.routeName}</span>
						<span style="padding:3px 8px;background:#f1f5f9;color:#64748b;border-radius:6px;font-size:10px;font-weight:600">#${route.sequence}</span>
					</div>
					<p style="margin:0;font-size:13px;color:#64748b;font-weight:600">운행 정보 없음</p>
				</div>
			`;
		}

		const {firstBus, secondBus} = arrivalData;

		// 혼잡도 뱃지
		const getCongestionBadge = (congestion) => {
			if (congestion === null || congestion === 0) return '';
			if (congestion === 3) return `<span style="padding:4px 8px;background:#dcfce7;border:1px solid #86efac;border-radius:6px;font-size:10px;font-weight:700;color:#16a34a">✓ 여유</span>`;
			if (congestion === 4) return `<span style="padding:4px 8px;background:#fef3c7;border:1px solid #fcd34d;border-radius:6px;font-size:10px;font-weight:700;color:#d97706">보통</span>`;
			if (congestion === 5) return `<span style="padding:4px 8px;background:#fee2e2;border:1px solid #fca5a5;border-radius:6px;font-size:10px;font-weight:700;color:#dc2626">혼잡</span>`;
			return '';
		};

		return `
			<div style="background:linear-gradient(135deg,#ffffff,#fafafa);border:1px solid #e5e7eb;border-radius:10px;padding:12px;margin-bottom:8px;transition:all 0.2s;cursor:pointer"
				onmouseover="this.style.transform='translateX(4px)';this.style.boxShadow='0 4px 12px rgba(59,130,246,0.15)';this.style.borderColor='#3b82f6'"
				onmouseout="this.style.transform='translateX(0)';this.style.boxShadow='none';this.style.borderColor='#e5e7eb'">
				<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;padding-bottom:10px;border-bottom:2px dashed #e5e7eb">
					<div style="display:flex;align-items:center;gap:8px">
						<span style="padding:6px 12px;background:linear-gradient(135deg,#3b82f6,#2563eb);color:white;border-radius:8px;font-size:16px;font-weight:800;box-shadow:0 2px 6px rgba(59,130,246,0.3)">${route.routeName}</span>
						<span style="padding:3px 8px;background:#f1f5f9;color:#64748b;border-radius:6px;font-size:10px;font-weight:600">#${route.sequence}</span>
					</div>
				</div>
				<div style="background:linear-gradient(135deg,#ecfeff,#cffafe);border:1px solid #a5f3fc;border-radius:8px;padding:10px;margin-bottom:8px">
					<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
						<span style="padding:4px 8px;background:linear-gradient(135deg,#06b6d4,#0891b2);color:white;border-radius:6px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">1st BUS</span>
						<span style="font-size:16px;font-weight:800;color:#0e7490;text-shadow:0 1px 2px rgba(0,0,0,0.1)">${firstBus.arrmsg || '정보 없음'}</span>
					</div>
					<div style="display:flex;gap:6px;flex-wrap:wrap">
						<span style="padding:4px 8px;background:white;border:1px solid #67e8f9;border-radius:6px;font-size:10px;font-weight:600;color:#0e7490;display:inline-flex;align-items:center;gap:4px">${busTypes[firstBus.busType] || '🚌 일반버스'}</span>
						${firstBus.isFull ? '<span style="padding:4px 8px;background:#fee2e2;border:1px solid #fca5a5;border-radius:6px;font-size:10px;font-weight:700;color:#dc2626">🚫 만차</span>' : getCongestionBadge(firstBus.congestion)}
						${firstBus.isLast ? '<span style="padding:4px 8px;background:#fef3c7;border:1px solid #fcd34d;border-radius:6px;font-size:10px;font-weight:700;color:#d97706">🌙 막차</span>' : ''}
					</div>
				</div>
				${secondBus ? `
					<div style="background:linear-gradient(135deg,#f5f3ff,#ede9fe);border:1px solid #d8b4fe;border-radius:8px;padding:10px">
						<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
							<span style="padding:4px 8px;background:linear-gradient(135deg,#a855f7,#9333ea);color:white;border-radius:6px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px">2nd BUS</span>
							<span style="font-size:14px;font-weight:800;color:#7e22ce;text-shadow:0 1px 2px rgba(0,0,0,0.1)">${secondBus.arrmsg || '정보 없음'}</span>
						</div>
						<div style="display:flex;gap:6px;flex-wrap:wrap">
							<span style="padding:4px 8px;background:white;border:1px solid #e9d5ff;border-radius:6px;font-size:10px;font-weight:600;color:#7e22ce;display:inline-flex;align-items:center;gap:4px">${busTypes[secondBus.busType] || '🚌 일반버스'}</span>
							${secondBus.isFull ? '<span style="padding:4px 8px;background:#fee2e2;border:1px solid #fca5a5;border-radius:6px;font-size:10px;font-weight:700;color:#dc2626">🚫 만차</span>' : getCongestionBadge(secondBus.congestion)}
							${secondBus.isLast ? '<span style="padding:4px 8px;background:#fef3c7;border:1px solid #fcd34d;border-radius:6px;font-size:10px;font-weight:700;color:#d97706">🌙 막차</span>' : ''}
						</div>
					</div>
				` : ''}
			</div>
		`;
	}

	// 버스 정류장 InfoWindow 생성 및 표시 (실시간 API 호출)
	async function showBusStationInfo(marker, station, isFrom) {
		const infoWindowId = `bus-${isFrom ? 'from' : 'to'}-${station.id}`;

		// 다른 InfoWindow 모두 닫기
		busStationInfoWindowsRef.current.forEach(iw => {
			try { iw.close(); } catch {}
		});

		// 로딩 상태 InfoWindow 생성
		const badge = isFrom ?
			'<span style="display:inline-block;padding:6px 12px;background:linear-gradient(135deg,#3b82f6,#2563eb);color:white;border-radius:20px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;box-shadow:0 2px 8px rgba(59,130,246,0.3);">🚩 출발지 근처</span>' :
			'<span style="display:inline-block;padding:6px 12px;background:linear-gradient(135deg,#ef4444,#b91c1c);color:white;border-radius:20px;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;box-shadow:0 2px 8px rgba(239,68,68,0.3);">🏁 도착지 근처</span>';

		const loadingContent = `
			<div id="${infoWindowId}" style="position:relative;padding:20px;min-width:380px;max-width:500px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Roboto',sans-serif;background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,0.12),0 2px 8px rgba(0,0,0,0.08);border:1px solid rgba(255,255,255,0.8)">
				<button onclick="window.busStationInfoWindows['${infoWindowId}']?.close()" style="position:absolute;top:16px;right:16px;width:32px;height:32px;border:none;background:rgba(239,68,68,0.1);color:#ef4444;border-radius:50%;cursor:pointer;font-size:20px;line-height:1;transition:all 0.3s cubic-bezier(0.4,0,0.2,1);display:flex;align-items:center;justify-content:center;font-weight:600" onmouseover="this.style.background='#ef4444';this.style.color='white';this.style.transform='rotate(90deg)'" onmouseout="this.style.background='rgba(239,68,68,0.1)';this.style.color='#ef4444';this.style.transform='rotate(0deg)'">×</button>
				<div style="margin-bottom:16px">
					<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">${badge}</div>
					<h3 style="margin:0;font-size:22px;font-weight:800;color:#0f172a;letter-spacing:-0.5px">${station.name}</h3>
				</div>
				<div style="padding:14px;background:white;border-radius:12px;margin-bottom:16px;border:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,0.05)">
					<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
						<span style="font-size:12px;color:#64748b;font-weight:600">📍 정류장 번호</span>
						<span style="font-size:15px;font-weight:700;color:#1e293b;font-family:'Courier New',monospace">${station.arsId || station.id}</span>
					</div>
					<div style="display:flex;justify-content:space-between;align-items:center">
						<span style="font-size:12px;color:#64748b;font-weight:600">📏 거리</span>
						<span style="font-size:15px;font-weight:700;color:#3b82f6">${station.distance}m</span>
					</div>
				</div>
				<div style="background:white;border-radius:12px;border:1px solid #e2e8f0;overflow:hidden">
					<div style="padding:12px 14px;background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border-bottom:1px solid #bae6fd">
						<div style="display:flex;align-items:center;justify-content:space-between">
							<span style="font-size:13px;color:#075985;font-weight:700">🚌 버스 도착 정보</span>
							<span style="padding:4px 10px;background:white;border-radius:12px;font-size:11px;font-weight:700;color:#0369a1;box-shadow:0 1px 3px rgba(0,0,0,0.1)">${station.routeCount}개 노선</span>
						</div>
					</div>
					<div id="${infoWindowId}-bus-list" style="max-height:380px;overflow-y:auto;padding:16px;text-align:center">
						<div style="padding:20px">
							<div style="width:40px;height:40px;border:4px solid #e2e8f0;border-top-color:#3b82f6;border-radius:50%;margin:0 auto 12px;animation:spin 1s linear infinite"></div>
							<style>@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}</style>
							<p style="margin:0;color:#64748b;font-size:14px;font-weight:600">버스 도착 정보를 불러오는 중...</p>
						</div>
					</div>
				</div>
			</div>
		`;

		const infoWindow = new window.naver.maps.InfoWindow({
			content: loadingContent,
			borderWidth: 0,
			disableAnchor: true,
			backgroundColor: 'transparent',
			pixelOffset: new window.naver.maps.Point(0, -12)
		});

		window.busStationInfoWindows[infoWindowId] = infoWindow;
		infoWindow.open(mapRef.current, marker);

		// 실시간 버스 도착 정보 가져오기
		try {
			const routes = station.routes || [];
			const displayRoutes = routes.slice(0, 12);

			console.log('[BUS STATION INFO] Station:', station.name, 'ID:', station.id);
			console.log('[BUS STATION INFO] Routes count:', routes.length);
			console.log('[BUS STATION INFO] Routes:', routes);

			// 모든 route의 도착 정보를 병렬로 가져오기
			const arrivalPromises = displayRoutes.map(route => {
				console.log('[BUS STATION INFO] Fetching arrival for:', route.routeName, 'stId:', station.id, 'busRouteId:', route.routeId, 'ord:', route.sequence);
				return fetchBusArrival(station.id, route.routeId, route.sequence);
			});

			const arrivalResults = await Promise.all(arrivalPromises);

			// HTML 생성
			const busListHTML = displayRoutes.map((route, idx) => {
				return createBusArrivalHTML(route, arrivalResults[idx]);
			}).join('');

			const moreRoutesHTML = station.routeCount > 12 ? `
				<div style="text-align:center;padding:12px;color:#64748b;font-size:12px;font-weight:600;background:#f8fafc;border-radius:8px;margin-top:8px">
					+${station.routeCount - 12}개 노선 더 보기
				</div>
			` : '';

			// InfoWindow 내용 업데이트
			const finalContent = `
				<div id="${infoWindowId}" style="position:relative;padding:20px;min-width:380px;max-width:500px;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI','Roboto',sans-serif;background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);border-radius:16px;box-shadow:0 8px 32px rgba(0,0,0,0.12),0 2px 8px rgba(0,0,0,0.08);border:1px solid rgba(255,255,255,0.8)">
					<button onclick="window.busStationInfoWindows['${infoWindowId}']?.close()" style="position:absolute;top:16px;right:16px;width:32px;height:32px;border:none;background:rgba(239,68,68,0.1);color:#ef4444;border-radius:50%;cursor:pointer;font-size:20px;line-height:1;transition:all 0.3s cubic-bezier(0.4,0,0.2,1);display:flex;align-items:center;justify-content:center;font-weight:600" onmouseover="this.style.background='#ef4444';this.style.color='white';this.style.transform='rotate(90deg)'" onmouseout="this.style.background='rgba(239,68,68,0.1)';this.style.color='#ef4444';this.style.transform='rotate(0deg)'">×</button>
					<div style="margin-bottom:16px">
						<div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">${badge}</div>
						<h3 style="margin:0;font-size:22px;font-weight:800;color:#0f172a;letter-spacing:-0.5px">${station.name}</h3>
					</div>
					<div style="padding:14px;background:white;border-radius:12px;margin-bottom:16px;border:1px solid #e2e8f0;box-shadow:0 1px 3px rgba(0,0,0,0.05)">
						<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px">
							<span style="font-size:12px;color:#64748b;font-weight:600">📍 정류장 번호</span>
							<span style="font-size:15px;font-weight:700;color:#1e293b;font-family:'Courier New',monospace">${station.arsId || station.id}</span>
						</div>
						<div style="display:flex;justify-content:space-between;align-items:center">
							<span style="font-size:12px;color:#64748b;font-weight:600">📏 거리</span>
							<span style="font-size:15px;font-weight:700;color:#3b82f6">${station.distance}m</span>
						</div>
					</div>
					<div style="background:white;border-radius:12px;border:1px solid #e2e8f0;overflow:hidden">
						<div style="padding:12px 14px;background:linear-gradient(135deg,#f0f9ff,#e0f2fe);border-bottom:1px solid #bae6fd">
							<div style="display:flex;align-items:center;justify-content:space-between">
								<span style="font-size:13px;color:#075985;font-weight:700">🚌 버스 도착 정보</span>
								<span style="padding:4px 10px;background:white;border-radius:12px;font-size:11px;font-weight:700;color:#0369a1;box-shadow:0 1px 3px rgba(0,0,0,0.1)">${station.routeCount}개 노선</span>
							</div>
						</div>
						<div style="max-height:380px;overflow-y:auto;padding:8px">
							${busListHTML}
							${moreRoutesHTML}
						</div>
					</div>
				</div>
			`;

			infoWindow.setContent(finalContent);

		} catch (error) {
			console.error('[BUS STATION INFO] Error loading arrival data:', error);
			// 에러 발생 시 InfoWindow에 에러 메시지 표시
			const errorContent = loadingContent.replace(
				'버스 도착 정보를 불러오는 중...',
				'❌ 도착 정보를 불러올 수 없습니다'
			);
			infoWindow.setContent(errorContent);
		}
	}

	// 버스 정류장 표시 함수
	async function showBusStations() {
		setStatus("버스 정류장 정보를 불러오는 중입니다…");
		try {
			const fromCoord = routeMarkersRef.current.start?.getPosition();
			const toCoord = routeMarkersRef.current.end?.getPosition();

			if (!fromCoord || !toCoord) {
				throw new Error("출발지와 도착지를 모두 선택하세요.");
			}

			// 기존 버스 정류장 마커 제거
			clearBusStationMarkers();

			// 출발지와 도착지 주변 300m 내 버스 정류장 검색
			const fromLat = fromCoord.y;
			const fromLng = fromCoord.x;
			const toLat = toCoord.y;
			const toLng = toCoord.x;

			console.log(`[CLIENT] Searching bus stations:`, {
				from: { lat: fromLat, lng: fromLng },
				to: { lat: toLat, lng: toLng }
			});

			const [fromStationsRes, toStationsRes] = await Promise.all([
				fetch(`${API_BASE}/seoul/bus-stations?lat=${fromLat}&lng=${fromLng}&radius=300`),
				fetch(`${API_BASE}/seoul/bus-stations?lat=${toLat}&lng=${toLng}&radius=300`)
			]);

			console.log(`[CLIENT] Response status:`, {
				from: fromStationsRes.status,
				to: toStationsRes.status
			});

			if (!fromStationsRes.ok || !toStationsRes.ok) {
				const fromError = await fromStationsRes.text();
				const toError = await toStationsRes.text();
				console.error(`[CLIENT] API Error:`, { fromError, toError });
				throw new Error("버스 정류장 정보를 가져올 수 없습니다.");
			}

			const fromData = await fromStationsRes.json();
			const toData = await toStationsRes.json();

			console.log(`[CLIENT] API Response:`, {
				fromData,
				toData
			});

			const fromStations = fromData.stations || [];
			const toStations = toData.stations || [];

			console.log(`출발지 주변 버스 정류장: ${fromStations.length}개 (총 ${fromData.totalStationsChecked || 0}개 확인)`);
			console.log(`도착지 주변 버스 정류장: ${toStations.length}개 (총 ${toData.totalStationsChecked || 0}개 확인)`);
			console.log(`[CLIENT] From stations:`, fromStations);
			console.log(`[CLIENT] To stations:`, toStations);

			// 전역 InfoWindow 저장소 초기화
			if (!window.busStationInfoWindows) {
				window.busStationInfoWindows = {};
			}

			// 출발지 주변 버스 정류장 마커 표시 (파란색)
			console.log(`[CLIENT] Creating ${fromStations.length} markers for FROM stations`);
			fromStations.forEach((station, idx) => {
				console.log(`[CLIENT] Creating FROM marker ${idx}:`, station);
				const marker = new window.naver.maps.Marker({
					position: new window.naver.maps.LatLng(station.lat, station.lng),
					map: mapRef.current,
					icon: {
						content: `
							<div style="
								position: relative;
								width: 28px;
								height: 28px;
								border-radius: 50%;
								background: linear-gradient(135deg, #3b82f6, #1d4ed8);
								border: 3px solid #fff;
								box-shadow: 0 4px 12px rgba(59, 130, 246, 0.6);
								cursor: pointer;
								display: flex;
								align-items: center;
								justify-content: center;
								color: white;
								font-size: 14px;
								font-weight: bold;
							">🚏</div>
						`,
						anchor: new window.naver.maps.Point(14, 14)
					},
					zIndex: 200
				});

				// 마커 클릭 시 실시간 버스 도착 정보 표시
				window.naver.maps.Event.addListener(marker, 'click', () => {
					showBusStationInfo(marker, station, true);
				});

				busStationMarkersRef.current.push(marker);
			});

			// 도착지 주변 버스 정류장 마커 표시 (빨간색)
			console.log(`[CLIENT] Creating ${toStations.length} markers for TO stations`);
			toStations.forEach((station, idx) => {
				console.log(`[CLIENT] Creating TO marker ${idx}:`, station);
				const marker = new window.naver.maps.Marker({
					position: new window.naver.maps.LatLng(station.lat, station.lng),
					map: mapRef.current,
					icon: {
						content: `
							<div style="
								position: relative;
								width: 28px;
								height: 28px;
								border-radius: 50%;
								background: linear-gradient(135deg, #ef4444, #b91c1c);
								border: 3px solid #fff;
								box-shadow: 0 4px 12px rgba(239, 68, 68, 0.6);
								cursor: pointer;
								display: flex;
								align-items: center;
								justify-content: center;
								color: white;
								font-size: 14px;
								font-weight: bold;
							">🚏</div>
						`,
						anchor: new window.naver.maps.Point(14, 14)
					},
					zIndex: 200
				});

				// 샘플 버스 도착 데이터 생성
				const generateSampleBusData = (route) => {
					const busTypes = ['일반버스', '저상버스', '굴절버스'];
					const busTypeIcons = ['🚌', '♿', '🚍'];
					return {
						firstBus: {
							arrivalTime: Math.floor(Math.random() * 8) + 1,
							isFull: Math.random() > 0.7,
							isLastBus: Math.random() > 0.9,
							busType: Math.floor(Math.random() * 3),
							getBusTypeName() { return busTypes[this.busType]; },
							getBusTypeIcon() { return busTypeIcons[this.busType]; }
						},
						secondBus: {
							arrivalTime: Math.floor(Math.random() * 15) + 9,
							isFull: Math.random() > 0.7,
							isLastBus: Math.random() > 0.95,
							busType: Math.floor(Math.random() * 3),
							getBusTypeName() { return busTypes[this.busType]; },
							getBusTypeIcon() { return busTypeIcons[this.busType]; }
						}
					};
				};

				const infoWindowId = `bus-to-${idx}`;
				const infoWindow = new window.naver.maps.InfoWindow({
					content: `
						<div id="${infoWindowId}" style="
							position: relative;
							padding: 20px;
							min-width: 380px;
							max-width: 500px;
							font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
							background: linear-gradient(135deg, #ffffff 0%, #fef2f2 100%);
							border-radius: 16px;
							box-shadow: 0 8px 32px rgba(0,0,0,0.12), 0 2px 8px rgba(0,0,0,0.08);
							border: 1px solid rgba(255,255,255,0.8);
						">
							<button
								onclick="window.busStationInfoWindows['${infoWindowId}']?.close()"
								style="
									position: absolute;
									top: 16px;
									right: 16px;
									width: 32px;
									height: 32px;
									border: none;
									background: rgba(239, 68, 68, 0.1);
									color: #ef4444;
									border-radius: 50%;
									cursor: pointer;
									font-size: 20px;
									line-height: 1;
									transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
									display: flex;
									align-items: center;
									justify-content: center;
									font-weight: 600;
								"
								onmouseover="this.style.background='#ef4444'; this.style.color='white'; this.style.transform='rotate(90deg)'"
								onmouseout="this.style.background='rgba(239, 68, 68, 0.1)'; this.style.color='#ef4444'; this.style.transform='rotate(0deg)'"
							>×</button>

							<!-- 헤더 -->
							<div style="margin-bottom: 16px;">
								<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
									<span style="
										display: inline-block;
										padding: 6px 12px;
										background: linear-gradient(135deg, #ef4444, #dc2626);
										color: white;
										border-radius: 20px;
										font-size: 11px;
										font-weight: 700;
										text-transform: uppercase;
										letter-spacing: 0.5px;
										box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
									">🎯 도착지 근처</span>
								</div>
								<h3 style="
									margin: 0;
									font-size: 22px;
									font-weight: 800;
									color: #0f172a;
									letter-spacing: -0.5px;
								">${station.name}</h3>
							</div>

							<!-- 정류장 정보 -->
							<div style="
								padding: 14px;
								background: white;
								border-radius: 12px;
								margin-bottom: 16px;
								border: 1px solid #fecaca;
								box-shadow: 0 1px 3px rgba(239,68,68,0.05);
							">
								<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
									<span style="font-size: 12px; color: #64748b; font-weight: 600;">📍 정류장 번호</span>
									<span style="font-size: 15px; font-weight: 700; color: #1e293b; font-family: 'Courier New', monospace;">${station.arsId || station.id}</span>
								</div>
								<div style="display: flex; justify-content: space-between; align-items: center;">
									<span style="font-size: 12px; color: #64748b; font-weight: 600;">📏 도착지까지</span>
									<span style="font-size: 15px; font-weight: 700; color: #ef4444;">${station.distance}m</span>
								</div>
							</div>

							<!-- 버스 목록 -->
							${station.routes && station.routes.length > 0 ? `
								<div style="
									background: white;
									border-radius: 12px;
									border: 1px solid #fecaca;
									overflow: hidden;
								">
									<div style="
										padding: 12px 14px;
										background: linear-gradient(135deg, #fef2f2, #fee2e2);
										border-bottom: 1px solid #fca5a5;
									">
										<div style="display: flex; align-items: center; justify-content: space-between;">
											<span style="font-size: 13px; color: #991b1b; font-weight: 700;">🚌 버스 도착 정보</span>
											<span style="
												padding: 4px 10px;
												background: white;
												border-radius: 12px;
												font-size: 11px;
												font-weight: 700;
												color: #dc2626;
												box-shadow: 0 1px 3px rgba(0,0,0,0.1);
											">${station.routeCount}개 노선</span>
										</div>
									</div>
									<div style="
										max-height: 380px;
										overflow-y: auto;
										padding: 8px;
									">
										${station.routes.slice(0, 12).map(route => {
						const busData = generateSampleBusData(route);
						return `
												<div style="
													background: linear-gradient(135deg, #ffffff, #fafafa);
													border: 1px solid #fecaca;
													border-radius: 10px;
													padding: 12px;
													margin-bottom: 8px;
													transition: all 0.2s;
													cursor: pointer;
												"
												onmouseover="this.style.transform='translateX(4px)'; this.style.boxShadow='0 4px 12px rgba(239,68,68,0.15)'; this.style.borderColor='#ef4444'"
												onmouseout="this.style.transform='translateX(0)'; this.style.boxShadow='none'; this.style.borderColor='#fecaca'">
													<!-- 버스 번호 헤더 -->
													<div style="
														display: flex;
														align-items: center;
														justify-content: space-between;
														margin-bottom: 12px;
														padding-bottom: 10px;
														border-bottom: 2px dashed #fecaca;
													">
														<div style="display: flex; align-items: center; gap: 8px;">
															<span style="
																padding: 6px 12px;
																background: linear-gradient(135deg, #ef4444, #dc2626);
																color: white;
																border-radius: 8px;
																font-size: 16px;
																font-weight: 800;
																box-shadow: 0 2px 6px rgba(239,68,68,0.3);
															">${route.routeName}</span>
															<span style="
																padding: 3px 8px;
																background: #fef2f2;
																color: #991b1b;
																border-radius: 6px;
																font-size: 10px;
																font-weight: 600;
															">#${route.sequence}</span>
														</div>
													</div>

													<!-- 첫 번째 버스 -->
													<div style="
														background: linear-gradient(135deg, #ecfeff, #cffafe);
														border: 1px solid #a5f3fc;
														border-radius: 8px;
														padding: 10px;
														margin-bottom: 8px;
													">
														<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
															<span style="
																padding: 4px 8px;
																background: linear-gradient(135deg, #06b6d4, #0891b2);
																color: white;
																border-radius: 6px;
																font-size: 10px;
																font-weight: 700;
																text-transform: uppercase;
																letter-spacing: 0.5px;
															">1st BUS</span>
															<span style="
																font-size: 20px;
																font-weight: 800;
																color: #0e7490;
																text-shadow: 0 1px 2px rgba(0,0,0,0.1);
															">${busData.firstBus.arrivalTime}분</span>
														</div>
														<div style="display: flex; gap: 6px; flex-wrap: wrap;">
															<span style="
																padding: 4px 8px;
																background: white;
																border: 1px solid #67e8f9;
																border-radius: 6px;
																font-size: 10px;
																font-weight: 600;
																color: #0e7490;
																display: inline-flex;
																align-items: center;
																gap: 4px;
															">
																${busData.firstBus.getBusTypeIcon()} ${busData.firstBus.getBusTypeName()}
															</span>
															${busData.firstBus.isFull ? `
																<span style="
																	padding: 4px 8px;
																	background: #fee2e2;
																	border: 1px solid #fca5a5;
																	border-radius: 6px;
																	font-size: 10px;
																	font-weight: 700;
																	color: #dc2626;
																">🚫 만차</span>
															` : `
																<span style="
																	padding: 4px 8px;
																	background: #dcfce7;
																	border: 1px solid #86efac;
																	border-radius: 6px;
																	font-size: 10px;
																	font-weight: 700;
																	color: #16a34a;
																">✓ 여유</span>
															`}
															${busData.firstBus.isLastBus ? `
																<span style="
																	padding: 4px 8px;
																	background: #fef3c7;
																	border: 1px solid #fcd34d;
																	border-radius: 6px;
																	font-size: 10px;
																	font-weight: 700;
																	color: #d97706;
																">🌙 막차</span>
															` : ''}
														</div>
													</div>

													<!-- 두 번째 버스 -->
													<div style="
														background: linear-gradient(135deg, #f5f3ff, #ede9fe);
														border: 1px solid #d8b4fe;
														border-radius: 8px;
														padding: 10px;
													">
														<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
															<span style="
																padding: 4px 8px;
																background: linear-gradient(135deg, #a855f7, #9333ea);
																color: white;
																border-radius: 6px;
																font-size: 10px;
																font-weight: 700;
																text-transform: uppercase;
																letter-spacing: 0.5px;
															">2nd BUS</span>
															<span style="
																font-size: 18px;
																font-weight: 800;
																color: #7e22ce;
																text-shadow: 0 1px 2px rgba(0,0,0,0.1);
															">${busData.secondBus.arrivalTime}분</span>
														</div>
														<div style="display: flex; gap: 6px; flex-wrap: wrap;">
															<span style="
																padding: 4px 8px;
																background: white;
																border: 1px solid #e9d5ff;
																border-radius: 6px;
																font-size: 10px;
																font-weight: 600;
																color: #7e22ce;
																display: inline-flex;
																align-items: center;
																gap: 4px;
															">
																${busData.secondBus.getBusTypeIcon()} ${busData.secondBus.getBusTypeName()}
															</span>
															${busData.secondBus.isFull ? `
																<span style="
																	padding: 4px 8px;
																	background: #fee2e2;
																	border: 1px solid #fca5a5;
																	border-radius: 6px;
																	font-size: 10px;
																	font-weight: 700;
																	color: #dc2626;
																">🚫 만차</span>
															` : `
																<span style="
																	padding: 4px 8px;
																	background: #dcfce7;
																	border: 1px solid #86efac;
																	border-radius: 6px;
																	font-size: 10px;
																	font-weight: 700;
																	color: #16a34a;
																">✓ 여유</span>
															`}
															${busData.secondBus.isLastBus ? `
																<span style="
																	padding: 4px 8px;
																	background: #fef3c7;
																	border: 1px solid #fcd34d;
																	border-radius: 6px;
																	font-size: 10px;
																	font-weight: 700;
																	color: #d97706;
																">🌙 막차</span>
															` : ''}
														</div>
													</div>
												</div>
											`;
					}).join('')}
										${station.routeCount > 12 ? `
											<div style="
												text-align: center;
												padding: 12px;
												color: #64748b;
												font-size: 12px;
												font-weight: 600;
												background: #fef2f2;
												border-radius: 8px;
												margin-top: 8px;
											">
												+${station.routeCount - 12}개 노선 더 보기
											</div>
										` : ''}
									</div>
								</div>
							` : ''}
						</div>
					`,
					borderWidth: 0,
					disableAnchor: true,
					backgroundColor: 'transparent',
					pixelOffset: new window.naver.maps.Point(0, -12)
				});

				// 마커 클릭 시 실시간 버스 도착 정보 표시
				window.naver.maps.Event.addListener(marker, 'click', () => {
					showBusStationInfo(marker, station, false);
				});

				busStationMarkersRef.current.push(marker);
			});

			console.log(`[CLIENT] Total markers created: ${busStationMarkersRef.current.length}`);
			console.log(`[CLIENT] Total info windows created: ${busStationInfoWindowsRef.current.length}`);

			// 지도 범위 조정 (출발지, 도착지, 버스 정류장 모두 포함)
			const allPoints = [fromCoord, toCoord];
			fromStations.forEach(station => {
				allPoints.push(new window.naver.maps.LatLng(station.lat, station.lng));
			});
			toStations.forEach(station => {
				allPoints.push(new window.naver.maps.LatLng(station.lat, station.lng));
			});

			console.log(`[CLIENT] Total points for bounds: ${allPoints.length}`);

			if (allPoints.length > 2) {
				const bounds = allPoints.reduce(
					(b, point) => (b.extend(point), b),
					new window.naver.maps.LatLngBounds()
				);
				mapRef.current.fitBounds(bounds, { top: 120, bottom: 140, left: 80, right: 80 });
			}

			// 버스 경로 탐색 및 표시
			try {
				setStatus("버스 경로를 찾는 중입니다…");
				const routeRes = await fetch(
					`${API_BASE}/seoul/bus-route?fromLat=${fromLat}&fromLng=${fromLng}&toLat=${toLat}&toLng=${toLng}&maxTransfers=2`
				);

				if (routeRes.ok) {
					const routeData = await routeRes.json();
					console.log(`[CLIENT] Route found:`, routeData);

					if (routeData.success && routeData.path) {
						// 경로 그리기
						drawBusRoute(routeData.path);

						const routeInfo = routeData.path.segments.map(s => s.routeName).join(' → ');
						setStatus(`🚌 ${routeInfo} (환승 ${routeData.path.transfers}회, ${routeData.path.totalStations}개 정류장)`);
					} else {
						const totalStations = fromStations.length + toStations.length;
						setStatus(`버스 경로를 찾을 수 없습니다. 출발지 근처 ${fromStations.length}개, 도착지 근처 ${toStations.length}개 정류장 표시.`);
					}
				} else {
					const totalStations = fromStations.length + toStations.length;
					setStatus(`버스 정류장: 출발지 근처 ${fromStations.length}개, 도착지 근처 ${toStations.length}개 찾았습니다. (총 ${totalStations}개 마커 표시)`);
				}
			} catch (routeError) {
				console.error('[CLIENT] Route error:', routeError);
				const totalStations = fromStations.length + toStations.length;
				setStatus(`버스 정류장: 출발지 근처 ${fromStations.length}개, 도착지 근처 ${toStations.length}개 찾았습니다. (총 ${totalStations}개 마커 표시)`);
			}
		} catch (err) {
			console.error(err);
			setStatus(err.message || "버스 정류장 정보를 불러오는데 실패했습니다.");
		}
	}

	// 버스 경로 그리기 함수
	function drawBusRoute(path) {
		if (!path || !path.segments) return;

		console.log(`[CLIENT] Drawing bus route with ${path.segments.length} segments`);

		// 각 노선 구간별로 polyline 그리기
		path.segments.forEach((segment, segmentIndex) => {
			const routeColor = segmentIndex === 0 ? '#3b82f6' : '#ef4444'; // 첫 번째 노선: 파란색, 환승 노선: 빨간색

			// 구간의 모든 정류장 좌표
			const pathCoords = segment.stations.map(station =>
				new window.naver.maps.LatLng(station.lat, station.lng)
			);

			// Polyline 생성
			const polyline = new window.naver.maps.Polyline({
				path: pathCoords,
				strokeColor: routeColor,
				strokeWeight: 6,
				strokeOpacity: 0.8,
				strokeStyle: 'solid',
				map: mapRef.current,
				zIndex: 100
			});

			busRoutePolylinesRef.current.push(polyline);

			// 경로상의 정류장 마커 표시
			segment.stations.forEach((station, stationIndex) => {
				// 출발 정류장과 도착 정류장은 이미 표시되어 있으므로 중간 정류장만 표시
				const isFirst = segmentIndex === 0 && stationIndex === 0;
				const isLast = segmentIndex === path.segments.length - 1 && stationIndex === segment.stations.length - 1;
				const isTransfer = stationIndex === 0 && segmentIndex > 0;

				let markerIcon;
				if (isTransfer) {
					// 환승 정류장
					markerIcon = {
						content: `
							<div style="
								width: 24px;
								height: 24px;
								border-radius: 50%;
								background: linear-gradient(135deg, #fbbf24, #f59e0b);
								border: 3px solid #fff;
								box-shadow: 0 2px 8px rgba(245, 158, 11, 0.6);
								display: flex;
								align-items: center;
								justify-content: center;
								font-size: 12px;
								font-weight: bold;
								color: white;
							">🔄</div>
						`,
						anchor: new window.naver.maps.Point(12, 12)
					};
				} else if (!isFirst && !isLast) {
					// 중간 정류장
					markerIcon = {
						content: `
							<div style="
								width: 12px;
								height: 12px;
								border-radius: 50%;
								background: ${routeColor};
								border: 2px solid #fff;
								box-shadow: 0 1px 4px rgba(0,0,0,0.3);
							"></div>
						`,
						anchor: new window.naver.maps.Point(6, 6)
					};
				} else {
					// 출발/도착 정류장은 건너뛰기 (이미 표시됨)
					return;
				}

				const marker = new window.naver.maps.Marker({
					position: new window.naver.maps.LatLng(station.lat, station.lng),
					map: mapRef.current,
					icon: markerIcon,
					zIndex: isTransfer ? 250 : 150
				});

				// InfoWindow 추가
				const infoWindowId = `route-station-${segmentIndex}-${stationIndex}`;

				// 샘플 버스 도착 데이터 생성
				const generateRouteBusData = () => {
					const busTypes = ['일반버스', '저상버스', '굴절버스'];
					const busTypeIcons = ['🚌', '♿', '🚍'];
					return {
						firstBus: {
							arrivalTime: Math.floor(Math.random() * 8) + 1,
							isFull: Math.random() > 0.7,
							isLastBus: Math.random() > 0.9,
							busType: Math.floor(Math.random() * 3),
							getBusTypeName() { return busTypes[this.busType]; },
							getBusTypeIcon() { return busTypeIcons[this.busType]; }
						},
						secondBus: {
							arrivalTime: Math.floor(Math.random() * 15) + 9,
							isFull: Math.random() > 0.7,
							isLastBus: Math.random() > 0.95,
							busType: Math.floor(Math.random() * 3),
							getBusTypeName() { return busTypes[this.busType]; },
							getBusTypeIcon() { return busTypeIcons[this.busType]; }
						}
					};
				};

				const routeBusData = generateRouteBusData();

				const infoWindow = new window.naver.maps.InfoWindow({
					content: `
						<div id="${infoWindowId}" style="
							position: relative;
							padding: 18px;
							min-width: 360px;
							max-width: 450px;
							font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
							background: linear-gradient(135deg, #ffffff 0%, ${isTransfer ? '#fef3c7' : '#f9fafb'} 100%);
							border-radius: 14px;
							box-shadow: 0 6px 24px rgba(0,0,0,0.12), 0 2px 6px rgba(0,0,0,0.08);
							border: 2px solid ${isTransfer ? '#fbbf24' : '#e5e7eb'};
						">
							<button
								onclick="window.busStationInfoWindows['${infoWindowId}']?.close()"
								style="
									position: absolute;
									top: 12px;
									right: 12px;
									width: 28px;
									height: 28px;
									border: none;
									background: rgba(239, 68, 68, 0.1);
									color: #ef4444;
									border-radius: 50%;
									cursor: pointer;
									font-size: 18px;
									line-height: 1;
									transition: all 0.3s;
									display: flex;
									align-items: center;
									justify-content: center;
									font-weight: 600;
								"
								onmouseover="this.style.background='#ef4444'; this.style.color='white'; this.style.transform='rotate(90deg)'"
								onmouseout="this.style.background='rgba(239, 68, 68, 0.1)'; this.style.color='#ef4444'; this.style.transform='rotate(0deg)'"
							>×</button>

							<!-- 헤더 -->
							<div style="margin-bottom: 14px;">
								${isTransfer ? `
									<div style="
										display: inline-block;
										padding: 6px 12px;
										background: linear-gradient(135deg, #fbbf24, #f59e0b);
										color: white;
										border-radius: 16px;
										font-size: 11px;
										font-weight: 700;
										margin-bottom: 8px;
										box-shadow: 0 2px 6px rgba(251, 191, 36, 0.4);
									">🔄 환승 정류장</div>
								` : `
									<div style="
										display: inline-block;
										padding: 6px 12px;
										background: linear-gradient(135deg, ${routeColor}, ${routeColor}dd);
										color: white;
										border-radius: 16px;
										font-size: 11px;
										font-weight: 700;
										margin-bottom: 8px;
										box-shadow: 0 2px 6px rgba(0,0,0,0.2);
									">📍 경로 상 정류장</div>
								`}
								<h3 style="
									margin: 0;
									font-size: 20px;
									font-weight: 800;
									color: #0f172a;
									letter-spacing: -0.5px;
								">${station.name}</h3>
								<div style="
									margin-top: 8px;
									font-size: 13px;
									color: #64748b;
									font-weight: 600;
								">
									${isTransfer ?
						`<span style="color: #3b82f6; font-weight: 700;">${path.segments[segmentIndex-1].routeName}</span>
										<span style="color: #fbbf24;"> → </span>
										<span style="color: ${routeColor}; font-weight: 700;">${segment.routeName}</span>`
						:
						`<span style="color: ${routeColor}; font-weight: 700;">${segment.routeName}</span> 번 버스`
					}
								</div>
							</div>

							<!-- 버스 도착 정보 카드 -->
							<div style="
								background: white;
								border-radius: 10px;
								border: 1px solid #e5e7eb;
								overflow: hidden;
								box-shadow: 0 1px 3px rgba(0,0,0,0.05);
							">
								<div style="
									padding: 10px 12px;
									background: linear-gradient(135deg, ${routeColor}15, ${routeColor}25);
									border-bottom: 1px solid ${routeColor}40;
								">
									<div style="display: flex; align-items: center; gap: 8px;">
										<span style="
											padding: 5px 10px;
											background: ${routeColor};
											color: white;
											border-radius: 8px;
											font-size: 15px;
											font-weight: 800;
											box-shadow: 0 2px 4px ${routeColor}40;
										">${segment.routeName}</span>
										<span style="font-size: 12px; color: #64748b; font-weight: 600;">실시간 도착 정보</span>
									</div>
								</div>

								<div style="padding: 10px;">
									<!-- 첫 번째 버스 -->
									<div style="
										background: linear-gradient(135deg, #ecfeff, #cffafe);
										border: 1px solid #a5f3fc;
										border-radius: 8px;
										padding: 10px;
										margin-bottom: 8px;
									">
										<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
											<span style="
												padding: 4px 8px;
												background: linear-gradient(135deg, #06b6d4, #0891b2);
												color: white;
												border-radius: 6px;
												font-size: 10px;
												font-weight: 700;
												text-transform: uppercase;
												letter-spacing: 0.5px;
											">1st BUS</span>
											<span style="
												font-size: 20px;
												font-weight: 800;
												color: #0e7490;
												text-shadow: 0 1px 2px rgba(0,0,0,0.1);
											">${routeBusData.firstBus.arrivalTime}분</span>
										</div>
										<div style="display: flex; gap: 6px; flex-wrap: wrap;">
											<span style="
												padding: 4px 8px;
												background: white;
												border: 1px solid #67e8f9;
												border-radius: 6px;
												font-size: 10px;
												font-weight: 600;
												color: #0e7490;
												display: inline-flex;
												align-items: center;
												gap: 4px;
											">
												${routeBusData.firstBus.getBusTypeIcon()} ${routeBusData.firstBus.getBusTypeName()}
											</span>
											${routeBusData.firstBus.isFull ? `
												<span style="
													padding: 4px 8px;
													background: #fee2e2;
													border: 1px solid #fca5a5;
													border-radius: 6px;
													font-size: 10px;
													font-weight: 700;
													color: #dc2626;
												">🚫 만차</span>
											` : `
												<span style="
													padding: 4px 8px;
													background: #dcfce7;
													border: 1px solid #86efac;
													border-radius: 6px;
													font-size: 10px;
													font-weight: 700;
													color: #16a34a;
												">✓ 여유</span>
											`}
											${routeBusData.firstBus.isLastBus ? `
												<span style="
													padding: 4px 8px;
													background: #fef3c7;
													border: 1px solid #fcd34d;
													border-radius: 6px;
													font-size: 10px;
													font-weight: 700;
													color: #d97706;
												">🌙 막차</span>
											` : ''}
										</div>
									</div>

									<!-- 두 번째 버스 -->
									<div style="
										background: linear-gradient(135deg, #f5f3ff, #ede9fe);
										border: 1px solid #d8b4fe;
										border-radius: 8px;
										padding: 10px;
									">
										<div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px;">
											<span style="
												padding: 4px 8px;
												background: linear-gradient(135deg, #a855f7, #9333ea);
												color: white;
												border-radius: 6px;
												font-size: 10px;
												font-weight: 700;
												text-transform: uppercase;
												letter-spacing: 0.5px;
											">2nd BUS</span>
											<span style="
												font-size: 18px;
												font-weight: 800;
												color: #7e22ce;
												text-shadow: 0 1px 2px rgba(0,0,0,0.1);
											">${routeBusData.secondBus.arrivalTime}분</span>
										</div>
										<div style="display: flex; gap: 6px; flex-wrap: wrap;">
											<span style="
												padding: 4px 8px;
												background: white;
												border: 1px solid #e9d5ff;
												border-radius: 6px;
												font-size: 10px;
												font-weight: 600;
												color: #7e22ce;
												display: inline-flex;
												align-items: center;
												gap: 4px;
											">
												${routeBusData.secondBus.getBusTypeIcon()} ${routeBusData.secondBus.getBusTypeName()}
											</span>
											${routeBusData.secondBus.isFull ? `
												<span style="
													padding: 4px 8px;
													background: #fee2e2;
													border: 1px solid #fca5a5;
													border-radius: 6px;
													font-size: 10px;
													font-weight: 700;
													color: #dc2626;
												">🚫 만차</span>
											` : `
												<span style="
													padding: 4px 8px;
													background: #dcfce7;
													border: 1px solid #86efac;
													border-radius: 6px;
													font-size: 10px;
													font-weight: 700;
													color: #16a34a;
												">✓ 여유</span>
											`}
											${routeBusData.secondBus.isLastBus ? `
												<span style="
													padding: 4px 8px;
													background: #fef3c7;
													border: 1px solid #fcd34d;
													border-radius: 6px;
													font-size: 10px;
													font-weight: 700;
													color: #d97706;
												">🌙 막차</span>
											` : ''}
										</div>
									</div>
								</div>
							</div>
						</div>
					`,
					borderWidth: 0,
					disableAnchor: true,
					backgroundColor: 'transparent',
					pixelOffset: new window.naver.maps.Point(0, -10)
				});

				// 마커 클릭 시 실시간 버스 도착 정보 표시
				window.naver.maps.Event.addListener(marker, 'click', () => {
					// 경로 상의 정류장도 실시간 API 사용
					showBusStationInfo(marker, {
						id: station.id,
						name: station.name,
						arsId: station.arsId,
						lat: station.lat,
						lng: station.lng,
						distance: 0,
						routes: station.routes || [],
						routeCount: station.routes?.length || 0
					}, segmentIndex === 0); // 첫 번째 구간이면 출발지처럼 표시
				});

				busStationMarkersRef.current.push(marker);
			});
		});

		console.log(`[CLIENT] Drew ${busRoutePolylinesRef.current.length} polylines`);
	}

	// 점과 선분 사이의 최단 거리 계산 (Haversine 기반, 미터 단위)
	function distanceToLineSegment(point, lineStart, lineEnd) {
		const R = 6371000; // 지구 반지름 (미터)

		// 각도를 라디안으로 변환
		const toRad = (deg) => deg * Math.PI / 180;

		const lat1 = toRad(lineStart.lat);
		const lng1 = toRad(lineStart.lng);
		const lat2 = toRad(lineEnd.lat);
		const lng2 = toRad(lineEnd.lng);
		const latP = toRad(point.lat);
		const lngP = toRad(point.lng);

		// 선분의 시작점과 끝점 사이의 거리
		const dLat12 = lat2 - lat1;
		const dLng12 = lng2 - lng1;
		const d12 = Math.sqrt(dLat12 * dLat12 + dLng12 * dLng12);

		if (d12 === 0) {
			// 시작점과 끝점이 같은 경우
			const dLat = latP - lat1;
			const dLng = lngP - lng1;
			const a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(lat1) * Math.cos(latP) * Math.sin(dLng/2) * Math.sin(dLng/2);
			const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
			return R * c;
		}

		// 투영 계수 계산 (간소화된 버전)
		const dLatP1 = latP - lat1;
		const dLngP1 = lngP - lng1;
		const t = Math.max(0, Math.min(1, (dLatP1 * dLat12 + dLngP1 * dLng12) / (d12 * d12)));

		// 투영점 계산
		const projLat = lat1 + t * dLat12;
		const projLng = lng1 + t * dLng12;

		// 점과 투영점 사이의 거리 (Haversine)
		const dLat = latP - projLat;
		const dLng = lngP - projLng;
		const a = Math.sin(dLat/2) * Math.sin(dLat/2) + Math.cos(projLat) * Math.cos(latP) * Math.sin(dLng/2) * Math.sin(dLng/2);
		const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
		return R * c;
	}

	// 점이 경로 주변 distance 미터 내에 있는지 확인
	function isNearRoute(point, routePath, distance) {
		if (!routePath || routePath.length < 2) return false;

		for (let i = 0; i < routePath.length - 1; i++) {
			const current = routePath[i];
			const next = routePath[i + 1];

			// 좌표가 유효한지 확인
			if (!current || !next) continue;
			if (current.y === undefined || current.x === undefined) continue;
			if (next.y === undefined || next.x === undefined) continue;

			const lineStart = { lat: current.y, lng: current.x };
			const lineEnd = { lat: next.y, lng: next.x };
			const dist = distanceToLineSegment(point, lineStart, lineEnd);
			if (dist <= distance) {
				return true;
			}
		}
		return false;
	}

	// 여행 코스 추천 함수
	const drawTravelCourse = useCallback(async () => {
		setStatus("여행 코스를 추천하고 있습니다…");
		try {
			const fromCoord = routeMarkersRef.current.start?.getPosition();
			const toCoord = routeMarkersRef.current.end?.getPosition();
			if (!fromCoord || !toCoord) {
				throw new Error("출발지와 도착지를 모두 선택하세요.");
			}
			const from = `${fromCoord.y},${fromCoord.x}`;
			const to = `${toCoord.y},${toCoord.x}`;

			// 1. OSRM 경로 계산
			const res = await fetch(`${API_BASE}/route?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}&alternatives=false`);
			if (!res.ok) throw new Error("경로 API 오류");
			const json = await res.json();

			if (!json?.routes || !Array.isArray(json.routes) || json.routes.length === 0) {
				throw new Error("추천 경로를 찾을 수 없습니다");
			}

			// 기존 경로 및 문화행사 마커 제거
			routePathsRef.current.forEach(polyline => {
				try { polyline.setMap(null); } catch {}
			});
			routePathsRef.current = [];
			clearCulturalEventMarkers();

			// 경로 그리기
			const route = json.routes[0];
			const coords = route?.geometry?.coordinates;
			if (!Array.isArray(coords) || !coords.length) {
				throw new Error("경로 데이터가 올바르지 않습니다");
			}

			const path = coords.map(([lng, lat]) => new window.naver.maps.LatLng(lat, lng));

			const polyline = new window.naver.maps.Polyline({
				map: mapRef.current,
				path,
				strokeColor: "#38bdf8",
				strokeWeight: 6,
				strokeOpacity: 0.9,
				zIndex: 100,
			});

			routePathsRef.current.push(polyline);

			// 2. 문화행사 데이터 가져오기
			setStatus("문화행사 정보를 불러오는 중입니다…");
			const culturalRes = await fetch(`${API_BASE}/seoul/cultural-events?start=1&end=1000`);
			if (!culturalRes.ok) throw new Error("문화행사 API 오류");
			const culturalData = await culturalRes.json();

			const events = culturalData.events || [];

			// 3. 경로 주변 1.5km 내 문화행사 필터링
			const nearbyEvents = events.filter(event => {
				const lat = parseFloat(event.LAT);
				const lng = parseFloat(event.LOT);

				if (!lat || !lng || isNaN(lat) || isNaN(lng)) return false;

				const point = { lat, lng };
				return isNearRoute(point, path, 1500); // 1.5km = 1500m
			});

			console.log(`전체 문화행사: ${events.length}개, 경로 주변: ${nearbyEvents.length}개`);

			// 출발지에서 가까운 순서대로 정렬 (거리 정보 추가)
			const fromLat = fromCoord.y;
			const fromLng = fromCoord.x;
			const eventsWithDistance = nearbyEvents.map(event => {
				const lat = parseFloat(event.LAT);
				const lng = parseFloat(event.LOT);
				const distance = haversine(
					{ lat: fromLat, lng: fromLng },
					{ lat, lng }
				) / 1000; // 미터를 킬로미터로 변환
				return { ...event, distanceFromStart: distance };
			}).sort((a, b) => a.distanceFromStart - b.distanceFromStart);

			// 4. 문화행사 마커 표시
			// 전역 InfoWindow 저장소 초기화
			if (!window.culturalInfoWindows) {
				window.culturalInfoWindows = {};
			}

			eventsWithDistance.forEach((event, eventIndex) => {
				const lat = parseFloat(event.LAT);
				const lng = parseFloat(event.LOT);

				// 문화행사 전용 마커 HTML
				const markerContent = `
					<div style="
						position: relative;
						width: 24px;
						height: 24px;
						border-radius: 50%;
						background: linear-gradient(135deg, #ec4899, #8b5cf6);
						border: 3px solid #fff;
						box-shadow: 0 4px 12px rgba(236, 72, 153, 0.5);
						cursor: pointer;
					">
						<div style="
							position: absolute;
							top: 50%;
							left: 50%;
							transform: translate(-50%, -50%);
							color: white;
							font-size: 14px;
							font-weight: bold;
						">🎭</div>
					</div>
				`;

				const marker = new window.naver.maps.Marker({
					position: new window.naver.maps.LatLng(lat, lng),
					map: mapRef.current,
					icon: {
						content: markerContent,
						anchor: new window.naver.maps.Point(12, 12)
					},
					zIndex: 200
				});

				// 5. 마커 클릭 시 정보창 표시
				const imgUrl = event.MAIN_IMG || event.THUMBNAIL || event.IMGURL || '';
				const infoWindowId = `cultural-info-${eventIndex}`;

				const infoWindow = new window.naver.maps.InfoWindow({
					content: `
						<div id="${infoWindowId}" style="
							position: relative;
							padding: 16px;
							min-width: 280px;
							max-width: 400px;
							font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
							background: white;
							border-radius: 12px;
							box-shadow: 0 4px 20px rgba(0,0,0,0.15);
						">
							<button
								onclick="window.culturalInfoWindows['${infoWindowId}']?.close()"
								style="
									position: absolute;
									top: 12px;
									right: 12px;
									width: 28px;
									height: 28px;
									border: none;
									background: #f3f4f6;
									color: #6b7280;
									border-radius: 50%;
									cursor: pointer;
									font-size: 18px;
									font-weight: bold;
									display: flex;
									align-items: center;
									justify-content: center;
									transition: all 0.2s;
									padding: 0;
									line-height: 1;
									z-index: 10;
								"
								onmouseover="this.style.background='#e5e7eb'; this.style.color='#374151';"
								onmouseout="this.style.background='#f3f4f6'; this.style.color='#6b7280';"
							>×</button>
							${imgUrl ? `
								<img src="${imgUrl}"
									style="width: 100%; height: 160px; object-fit: cover; border-radius: 8px; margin-bottom: 12px;"
									onerror="this.style.display='none'"
								/>
							` : ''}
							<h3 style="
								margin: 0 0 8px 0;
								padding-right: 28px;
								font-size: 16px;
								font-weight: 700;
								color: #1f2937;
								line-height: 1.4;
							">${event.TITLE || event.EVENT_NM || '제목 없음'}</h3>

							${event.CODENAME ? `
								<div style="
									display: inline-block;
									padding: 4px 10px;
									background: linear-gradient(135deg, #ec4899, #8b5cf6);
									color: white;
									border-radius: 12px;
									font-size: 11px;
									font-weight: 600;
									margin-bottom: 12px;
								">${event.CODENAME}</div>
							` : ''}

							${event.DATE || event.EVENT_PERIOD ? `
								<p style="margin: 8px 0; font-size: 13px; color: #6b7280;">
									<strong style="color: #374151;">📅 기간:</strong> ${event.DATE || event.EVENT_PERIOD}
								</p>
							` : ''}

							${event.PLACE || event.EVENT_PLACE ? `
								<p style="margin: 8px 0; font-size: 13px; color: #6b7280;">
									<strong style="color: #374151;">📍 장소:</strong> ${event.PLACE || event.EVENT_PLACE}
								</p>
							` : ''}

							${event.ORG_NAME ? `
								<p style="margin: 8px 0; font-size: 13px; color: #6b7280;">
									<strong style="color: #374151;">🏢 주최:</strong> ${event.ORG_NAME}
								</p>
							` : ''}

							${event.USE_TRGT ? `
								<p style="margin: 8px 0; font-size: 13px; color: #6b7280;">
									<strong style="color: #374151;">👥 대상:</strong> ${event.USE_TRGT}
								</p>
							` : ''}

							${event.USE_FEE ? `
								<p style="margin: 8px 0; font-size: 13px; color: #6b7280;">
									<strong style="color: #374151;">💰 이용료:</strong> ${event.USE_FEE}
								</p>
							` : ''}

							${event.ETC_DESC || event.EVENT_ETC_DETAIL ? `
								<p style="margin: 12px 0 8px 0; font-size: 12px; color: #6b7280; line-height: 1.5;">
									${(event.ETC_DESC || event.EVENT_ETC_DETAIL).substring(0, 150)}${(event.ETC_DESC || event.EVENT_ETC_DETAIL).length > 150 ? '...' : ''}
								</p>
							` : ''}

							${event.HMPG_ADDR || event.URL ? `
								<a href="${event.HMPG_ADDR || event.URL}"
									target="_blank"
									style="
										display: inline-block;
										margin-top: 12px;
										padding: 8px 16px;
										background: #3b82f6;
										color: white;
										text-decoration: none;
										border-radius: 6px;
										font-size: 13px;
										font-weight: 600;
										transition: background 0.2s;
									"
									onmouseover="this.style.background='#2563eb'"
									onmouseout="this.style.background='#3b82f6'"
								>자세히 보기 →</a>
							` : ''}
						</div>
					`,
					borderWidth: 0,
					disableAnchor: true,
					backgroundColor: 'transparent',
					pixelOffset: new window.naver.maps.Point(0, -12)
				});

				// InfoWindow를 전역 객체에 저장
				window.culturalInfoWindows[infoWindowId] = infoWindow;

				window.naver.maps.Event.addListener(marker, 'click', () => {
					// 다른 InfoWindow 닫기
					culturalEventInfoWindowsRef.current.forEach(iw => {
						try { iw.close(); } catch {}
					});
					infoWindow.open(mapRef.current, marker);
				});

				culturalEventMarkersRef.current.push(marker);
				culturalEventInfoWindowsRef.current.push(infoWindow);
			});

			// 지도 범위 조정
			const allPoints = [...path];
			nearbyEvents.forEach(event => {
				const lat = parseFloat(event.LAT);
				const lng = parseFloat(event.LOT);
				if (lat && lng && !isNaN(lat) && !isNaN(lng)) {
					allPoints.push(new window.naver.maps.LatLng(lat, lng));
				}
			});

			if (allPoints.length > 0) {
				const bounds = allPoints.reduce(
					(b, point) => (b.extend(point), b),
					new window.naver.maps.LatLngBounds(allPoints[0], allPoints[0])
				);
				mapRef.current.fitBounds(bounds, { top: 120, bottom: 140, left: 80, right: 80 });
			}

			setCulturalEvents(eventsWithDistance);
			setRoutes([{
				index: 0,
				distance: route.distance,
				duration: route.duration,
				path,
				polyline,
				color: "#38bdf8"
			}]);
			setRouteSource('travel-course');
			setSelectedRouteIndex(0);
			setStatus(`여행 코스: 경로 주변 ${eventsWithDistance.length}개의 문화행사를 찾았습니다.`);
		} catch (err) {
			console.error(err);
			setStatus(err.message || "여행 코스 추천 실패");
			setCulturalEvents([]);
			setRoutes([]);
			setSelectedRouteIndex(0);
		}
	}, []);


	// 대중교통(ODsay) 경로 + 실시간 보정/채점
	const drawTransit = useCallback(async () => {
		setStatus("대중교통 경로를 계산 중입니다…");
		try {
			const fromCoord = routeMarkersRef.current.start?.getPosition();
			const toCoord   = routeMarkersRef.current.end?.getPosition();
			if (!fromCoord || !toCoord) {
				throw new Error("출발지와 도착지를 모두 선택하세요.");
			}

			// ODsay 파라미터 준비(ODsay는 lng/lat 순서 사용)
			const qs = new URLSearchParams({
				SX: String(fromCoord.x), SY: String(fromCoord.y),
				EX: String(toCoord.x),   EY: String(toCoord.y),
				SearchType: "0", // 기본값
			});

			// 1) Worker 프록시가 있을 경우 우선 시도:  /odsay/searchPubTransPathT
			//    (없으면 catch에서 2)로 폴백)
			let odsayData = null;
			let ok = false;
			try {
				const r = await fetch(`${API_BASE}/odsay/searchPubTransPathT?${qs.toString()}`);
				if (r.ok) { odsayData = await r.json(); ok = true; }
			} catch {}

			// 2) 직접 호출(키 필요). .env에 VITE_ODSAY_KEY 넣으면 작동
			if (!ok && import.meta.env.VITE_ODSAY_KEY) {
				qs.set("apiKey", String(import.meta.env.VITE_ODSAY_KEY));
				const r2 = await fetch(`https://api.odsay.com/v1/api/searchPubTransPathT?${qs.toString()}`);
				if (!r2.ok) throw new Error("ODsay API 오류");
				odsayData = await r2.json();
				ok = true;
			}

			if (!ok) {
				throw new Error("대중교통 API 엔드포인트가 없거나 ODsay 키가 없습니다.");
			}

			// ★ 바로 여기에서 '보정·채점'을 적용한다.
			const { scoredPaths /*, usedData*/ } = await enrichTransitPaths(
				odsayData?.result?.path || [],
				{
					origin: { lat: fromCoord.y, lng: fromCoord.x },
					dest:   { lat: toCoord.y,   lng: toCoord.x },
					// 가중치는 필요시 UI와 연결해서 바꿔도 됨
					prefs:  { alpha: 1, beta: 1.5, gamma: 4, delta: 1, epsilon: 1 },
					useTopis: false, // 링크ID 매핑 전이면 false 유지
				}
			);

			setTransitList(scoredPaths);               // 보정된 경로들(최적 경로는 scoredPaths[0])
			setStatus("대중교통 추천 경로를 불러왔습니다.");
		} catch (err) {
			console.error(err);
			setTransitList([]);
			setStatus(err.message || "대중교통 경로 계산 실패");
		}
	}, []);

	// 지도 줌 컨트롤 (부드러운 애니메이션)
	const zoomIn = useCallback(() => {
		if (mapRef.current) {
			const currentZoom = mapRef.current.getZoom();
			const currentCenter = mapRef.current.getCenter();
			mapRef.current.morph(currentCenter, currentZoom + 1, {
				duration: 300,
				easing: 'easeOutCubic'
			});
		}
	}, []);

	const zoomOut = useCallback(() => {
		if (mapRef.current) {
			const currentZoom = mapRef.current.getZoom();
			const currentCenter = mapRef.current.getCenter();
			mapRef.current.morph(currentCenter, currentZoom - 1, {
				duration: 300,
				easing: 'easeOutCubic'
			});
		}
	}, []);

	// 지하철 경로 그리기
	const drawSubwayRoute = useCallback(async () => {
		setStatus("지하철 경로를 계산 중입니다…");
		try {
			const fromCoord = routeMarkersRef.current.start?.getPosition();
			const toCoord = routeMarkersRef.current.end?.getPosition();
			if (!fromCoord || !toCoord) {
				throw new Error("출발지와 도착지를 모두 선택하세요.");
			}

			setStatus("지하철역 데이터를 불러오는 중…");
			const stations = await loadSubwayStations();

			if (stations.length === 0) {
				throw new Error("지하철역 데이터를 불러올 수 없습니다.");
			}

			setStatus("출발지 근처 지하철역을 찾는 중…");
			const nearStartStations = findNearestStations(fromCoord.y, fromCoord.x, stations, 3);
			const nearEndStations = findNearestStations(toCoord.y, toCoord.x, stations, 3);

			if (nearStartStations.length === 0 || nearEndStations.length === 0) {
				throw new Error("근처에 지하철역이 없습니다.");
			}

			console.log("근처 출발역:", nearStartStations.map(s => `${s.name}(${s.routeName})`));
			console.log("근처 도착역:", nearEndStations.map(s => `${s.name}(${s.routeName})`));

			setStatus("최적 지하철 경로를 계산 중…");

			// 모든 조합에서 최단 경로 찾기
			let bestPath = null;
			let bestDistance = Infinity;

			for (const startStation of nearStartStations) {
				for (const endStation of nearEndStations) {
					const path = findSubwayPath(startStation, endStation, stations);
					if (path && path.length < bestDistance) {
						bestPath = path;
						bestDistance = path.length;
					}
				}
			}

			if (!bestPath) {
				throw new Error("지하철 경로를 찾을 수 없습니다.");
			}

			console.log("찾은 경로:", bestPath.map(p => `${p.station.name}(${p.station.routeName})`));

			// 기존 경로 제거
			routePathsRef.current.forEach(polyline => {
				try { polyline.setMap(null); } catch {}
			});
			routePathsRef.current = [];

			// 경로를 구간별로 나누기 (같은 노선끼리 묶기)
			const segments = getPathSegments(bestPath);

			console.log("경로 구간:", segments.map(s => `${s.routeName} (${s.stations.length}개 역)`));

			// 각 구간을 지도에 그리기
			segments.forEach((segment, idx) => {
				const path = segment.stations.map(station =>
					new window.naver.maps.LatLng(station.lat, station.lng)
				);

				const polyline = new window.naver.maps.Polyline({
					map: mapRef.current,
					path: path,
					strokeColor: segment.color,
					strokeWeight: 6,
					strokeOpacity: 0.8,
					zIndex: 100 + idx,
				});

				routePathsRef.current.push(polyline);

				// 각 역에 마커 추가
				segment.stations.forEach((station, stationIdx) => {
					const marker = new window.naver.maps.Marker({
						map: mapRef.current,
						position: new window.naver.maps.LatLng(station.lat, station.lng),
						icon: {
							content: `<div style="
								background: ${segment.color};
								color: white;
								padding: 4px 8px;
								border-radius: 12px;
								font-size: 11px;
								font-weight: bold;
								border: 2px solid white;
								box-shadow: 0 2px 4px rgba(0,0,0,0.3);
								white-space: nowrap;
							">${station.name}</div>`,
							anchor: new window.naver.maps.Point(0, 0)
						},
						zIndex: 200
					});

					// 마커는 버스 정류장 마커 ref에 저장 (polyline이 아니므로 routePathsRef에 추가하지 않음)
					busStationMarkersRef.current.push(marker);
				});
			});

			// 지도 범위 조정
			const bounds = new window.naver.maps.LatLngBounds(
				new window.naver.maps.LatLng(fromCoord.y, fromCoord.x),
				new window.naver.maps.LatLng(toCoord.y, toCoord.x)
			);

			bestPath.forEach(item => {
				bounds.extend(new window.naver.maps.LatLng(item.station.lat, item.station.lng));
			});

			mapRef.current.fitBounds(bounds, { top: 50, right: 50, bottom: 50, left: 50 });

			// 경로 정보 표시
			const totalStations = bestPath.length;
			const transfers = segments.length - 1;
			const routeInfo = segments.map(s => s.routeName).join(' → ');

			setStatus(`지하철 경로: ${routeInfo} (총 ${totalStations}개 역, 환승 ${transfers}회)`);

		} catch (err) {
			console.error(err);
			setStatus(err.message || "지하철 경로 계산 실패");
		}
	}, []);


	const refreshData = useCallback(async (forceRefresh = false) => {
		if (!placesRef.current.length) return;
		setLoading(true);
		setStatus("120개 지역 데이터를 불러오는 중... (최대 30초 소요)");

		try {
			// 🔧 수정: forceRefresh 파라미터로 제어
			const batch = await fetchCitydataBatch(
				placesRef.current,
				geoCacheRef.current,
				forceRefresh  // 호출하는 쪽에서 제어
			);

			// 성공/실패 통계
			const successCount = batch.filter(item =>
				item.stats?.level && item.stats.level !== "데이터없음"
			).length;
			const failCount = batch.length - successCount;

			placesRef.current = placesRef.current.map((p) => ({
				...p,
				coord: geoCacheRef.current[p.name] || p.coord
			}));

			setPlaces([...placesRef.current]);
			setDataset(batch);
			datasetRef.current = batch;

			// 상태 메시지
			if (failCount > batch.length * 0.5) {
				setStatus(`⚠️ 일부 데이터만 로드됨 (${successCount}/${batch.length}개)`);
				console.error(`High failure rate: ${failCount}/${batch.length}`);
			} else if (failCount > 0) {
				setStatus(`✓ ${successCount}개 지역 로드 (${failCount}개 데이터 없음)`);
			} else {
				setStatus(`✓ ${successCount}개 지역 데이터 로드 완료`);
			}

		} catch (err) {
			console.error("refreshData error:", err);
			setStatus(`❌ 데이터 로딩 실패: ${err.message}`);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		datasetRef.current = dataset;
	}, [dataset]);

	useEffect(() => {
		(async () => {
			try {
				if (!NAVER_KEY_ID && !NAVER_CUSTOM_URL) {
					throw new Error("네이버 지도 API 키가 설정되어 있지 않습니다. .env에 VITE_NAVER_KEY_ID 또는 VITE_NAVER_MAPS_URL을 설정하세요.");
				}
				setStatus("네이버 지도를 불러오는 중…");

// utils/naverReady.js 로더만 사용 (geocoder + visualization 강제 로드)
				await loadNaverMaps(NAVER_KEY_ID, NAVER_CUSTOM_URL);

				await waitForGeocoder();
				setMapReady(true);


				const naver = window.naver;
				await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

				const mapEl = document.getElementById("map");
				if (!mapEl) throw new Error("map element not found");
				await waitForNonZeroSize(mapEl);

				const map = new naver.maps.Map(mapEl, {
					center: new naver.maps.LatLng(37.5665, 126.9780),
					zoom: 12,
					logoControl: false,
					mapDataControl: true,
					zoomControl: false, // 기본 줌 컨트롤 숨김
				});
				mapRef.current = map;

				await Promise.race([
					new Promise(res => naver.maps.Event.once(map, 'idle', res)),
					new Promise(res => setTimeout(res, 3000))
				]);

				mapRef.current = map;
				heatRef.current = new naver.maps.visualization.HeatMap({
					map,
					data: [],
					radius,
					opacity,
					gradient: HEAT_GRADIENT,
				});

				naver.maps.Event.addListener(map, "click", (e) => {
					if (pickModeRef.current) {
						const mode = pickModeRef.current;
						pickModeRef.current = null;
						const coord = { lat: e.coord.y, lng: e.coord.x };
						const label = `${coord.lat.toFixed(5)}, ${coord.lng.toFixed(5)}`;
						adoptRoutePoint(mode, { name: label, coord }).catch(() => {});
						return;
					}
					if (!datasetRef.current.length) return;
					const point = { lat: e.coord.y, lng: e.coord.x };
					let best = null;
					let bestDistance = Infinity;
					for (const item of datasetRef.current) {
						if (!item.coord) continue;
						const dx = item.coord.lat - point.lat;
						const dy = item.coord.lng - point.lng;
						const distance = Math.sqrt(dx * dx + dy * dy) * 111_139;
						if (distance < bestDistance) {
							bestDistance = distance;
							best = item;
						}
					}
					if (best && bestDistance < 500) {
						highlightPlace(best, { panTo: false });
					}
				});

				setMapReady(true);
				setStatus("장소 데이터를 불러오는 중…");
				const placesJson = await fetchJSON("/places.json");
				const normalized = Array.isArray(placesJson)
					? placesJson.map((p, idx) => ({ ...p, no: p.no ?? idx + 1 }))
					: [];
				placesRef.current = normalized;
				setPlaces(normalized);

				const categoryList = Array.from(new Set(normalized.map((p) => p.category))).sort();
				initialCategoriesRef.current = new Set(categoryList);
				userOverrideRef.current = false;
				setCategories(categoryList);
				setActiveCats(new Set(categoryList));

				// 초기 로딩: 캐시 사용 (forceRefresh = false)
				await refreshData(false);

				setLoading(false);
				setStatus("지도 준비 완료");
			} catch (err) {
				console.error(err);
				setStatus(err.message || "초기화 실패");
				setLoading(false);
			}
		})();
	}, []);

	useEffect(() => {
		if (!dataset.length) return;
		updateHeatmap();
	}, [
		dataset,         // 데이터 바뀌면 히트맵 다시 그림
		activeCats,      // 카테고리 필터 반영
		radius,          // 히트맵 반지름
		opacity,         // 히트맵 불투명도
		showHeatmap,     // 표시/숨김 토글
		updateHeatmap
	]);

	useEffect(() => {
		if (!dataset.length) return;
		updatePlaceMarkers();
	}, [
		dataset,      // 좌표/목록 변경 반영
		activeCats,   // 카테고리 필터 반영
		updatePlaceMarkers
	]);
	useEffect(() => {
		if (!mapReady || !dataset.length) return;
		updateHeatmap();
		updatePlaceMarkers();
	}, [mapReady, dataset, updateHeatmap, updatePlaceMarkers]);
	useEffect(() => {
		if (!mapRef.current) return;
		const map = mapRef.current;
		const listener = window.naver.maps.Event.addListener(map, 'zoom_changed', () => {
			const z = map.getZoom();
			// 예: 줌이 낮으면 마커 숨김
			if (z < 11) {
				placeMarkersRef.current.forEach(m => m.setMap(null));
			} else {
				placeMarkersRef.current.forEach(m => m.setMap(map));
			}
		});
		return () => window.naver.maps.Event.removeListener(listener);
	}, []);

	// 맵 준비 시 문화행사 데이터 로드 및 마커 생성
	useEffect(() => {
		if (!mapReady) return;

		const initCulturalEvents = async () => {
			const events = await loadCulturalEvents();
			if (events && events.length > 0) {
				createCulturalMarkers(events);
			}
		};

		initCulturalEvents();
	}, [mapReady, loadCulturalEvents, createCulturalMarkers]);

	// 문화행사 데이터 또는 필터 변경 시 마커 재생성
	useEffect(() => {
		if (!mapReady || culturalEvents.length === 0) return;
		createCulturalMarkers(filteredCulturalEvents);
	}, [culturalEvents, filteredCulturalEvents, mapReady, createCulturalMarkers]);

	// 문화행사 마커 표시/숨김 토글
	useEffect(() => {
		if (!mapRef.current) return;

		culturalEventMarkersRef.current.forEach(marker => {
			try {
				if (showCulturalMarkers) {
					marker.setMap(mapRef.current);
				} else {
					marker.setMap(null);
				}
			} catch (e) {
				console.error('Error toggling cultural marker:', e);
			}
		});
	}, [showCulturalMarkers]);

	// 여행지 목록 패널 열릴 때 limit와 검색어 리셋
	useEffect(() => {
		if (showDestinationList) {
			setDestinationListLimit(50);
			setDestinationSearchQuery('');
		}
	}, [showDestinationList]);

	// 검색어 변경 시 limit 리셋
	useEffect(() => {
		setDestinationListLimit(50);
	}, [destinationSearchQuery]);

	// 20분마다 자동으로 도시 데이터 갱신
	useEffect(() => {
		// 초기 로딩 중이거나 데이터가 없으면 실행하지 않음
		if (loading || !dataset.length) return;

		const REFRESH_INTERVAL = 20 * 60 * 1000; // 20분 (밀리초)

		const intervalId = setInterval(() => {
			console.log('[AUTO-REFRESH] 20분 경과 - 도시 데이터 자동 갱신 시작');
			refreshData(false); // Edge Cache에서 최신 데이터 가져오기
		}, REFRESH_INTERVAL);

		console.log('[AUTO-REFRESH] 자동 갱신 타이머 시작 (20분 간격)');

		return () => {
			clearInterval(intervalId);
			console.log('[AUTO-REFRESH] 자동 갱신 타이머 정리');
		};
	}, [loading, dataset.length, refreshData]);

	// dataset이 업데이트되면 focusItem도 자동으로 갱신
	useEffect(() => {
		if (!focusItem || !dataset.length) return;

		// dataset에서 현재 focusItem과 같은 장소 찾기
		const updatedItem = dataset.find(item =>
			item.place?.name === focusItem.place?.name ||
			item.place?.code === focusItem.place?.code
		);

		if (updatedItem) {
			console.log('[AUTO-REFRESH] focusItem 자동 갱신:', updatedItem.place?.name);
			setFocusItem(updatedItem);
		}
	}, [dataset]);

	// Update polyline styles when route selection changes
	useEffect(() => {
		if (!routes.length) return;
		routes.forEach((route, idx) => {
			if (route.polyline) {
				const isSelected = idx === selectedRouteIndex;
				route.polyline.setOptions({
					strokeWeight: isSelected ? 6 : 4,
					strokeOpacity: isSelected ? 0.9 : 0.6,
					zIndex: isSelected ? 100 : 50 - idx
				});
			}
		});
	}, [selectedRouteIndex, routes]);




	const handleToggleCategory = (cat) => {
		const wasOverride = userOverrideRef.current;
		userOverrideRef.current = true;
		setActiveCats((prev) => {
			const base = prev.size === 0 && !wasOverride && initialCategoriesRef.current.size
				? new Set(initialCategoriesRef.current)
				: new Set(prev);
			if (base.has(cat)) base.delete(cat); else base.add(cat);
			return base;
		});
	};

	const handleSelectAllCategories = () => {
		if (!categories.length) return;
		const wasOverride = userOverrideRef.current;
		userOverrideRef.current = true;
		setActiveCats((prev) => {
			const prevSize = prev.size === 0 && !wasOverride
				? initialCategoriesRef.current.size
				: prev.size;
			if (prevSize === categories.length) {
				return new Set();
			}
			return new Set(categories);
		});
	};

	// 고급 필터 옵션 목록
	const businessTypeOptions = useMemo(() => [
		'유흥',
		'스포츠/문화/레저',
		'기타요식',
		'한식',
		'일식/중식/양식',
		'편의점',
		'할인점/슈퍼마켓',
		'제과/커피/패스트푸드',
		'기타'
	], []);

	const genderRatioOptions = useMemo(() => [
		'0:10~4:6',
		'4:6~5:5',
		'5:5~6:4',
		'6:4~10:0'
	], []);

	const ageGroupOptions = useMemo(() => [
		'10세 미만(4% 이상)',
		'10대(9% 이상)',
		'20대(20% 이상)',
		'30대(20% 이상)',
		'40대(20% 이상)',
		'50대(18% 이상)',
		'60대(14% 이상)',
		'70세 이상(10% 이상)',
		'기타'
	], []);

	const residencyRatioOptions = useMemo(() => [
		'0:10~4:6',
		'4:6~5:5',
		'5:5~6:4',
		'6:4~10:0'
	], []);

	// 고급 필터 토글 핸들러
	const handleToggleBusinessType = (type) => {
		setActiveBusinessTypes(prev => {
			const next = new Set(prev);
			if (next.has(type)) {
				next.delete(type);
			} else {
				next.add(type);
			}
			return next;
		});
	};

	const handleToggleGenderRatio = (ratio) => {
		setActiveGenderRatios(prev => {
			const next = new Set(prev);
			if (next.has(ratio)) {
				next.delete(ratio);
			} else {
				next.add(ratio);
			}
			return next;
		});
	};

	const handleToggleAgeGroup = (group) => {
		setActiveAgeGroups(prev => {
			const next = new Set(prev);
			if (next.has(group)) {
				next.delete(group);
			} else {
				next.add(group);
			}
			return next;
		});
	};

	const handleToggleResidencyRatio = (ratio) => {
		setActiveResidencyRatios(prev => {
			const next = new Set(prev);
			if (next.has(ratio)) {
				next.delete(ratio);
			} else {
				next.add(ratio);
			}
			return next;
		});
	};

	// 전체 필터 초기화
	const handleResetAllFilters = () => {
		setActiveBusinessTypes(new Set());
		setActiveGenderRatios(new Set());
		setActiveAgeGroups(new Set());
		setActiveResidencyRatios(new Set());
	};

	const handleSearchSelection = useCallback(async (selection) => {
		try {
			const coord = await ensureSelectionCoordinate(selection);
			const baseItem = datasetRef.current.find((item) =>
				item.place.name === selection?.name || item.place.code === selection?.code
			);

			let item = baseItem;

			// 데이터가 없는 경우, 주변 장소 기반으로 추정
			if (!baseItem && coord) {
				// 1. 주변 장소 찾기 (거리 계산 및 정렬)
				const nearbyPlaces = datasetRef.current
					.filter(place =>
						place.coord &&
						place.stats?.population != null &&
						place.stats?.level != null
					)
					.map(place => ({
						...place,
						distance: haversine(
							{ lat: coord.lat, lng: coord.lng },
							{ lat: place.coord.lat, lng: place.coord.lng }
						)
					}))
					.filter(place => place.distance <= 5000) // 5km 이내
					.sort((a, b) => a.distance - b.distance)
					.slice(0, 4); // 가장 가까운 최대 4개

				let estimatedStats = {
					level: null,
					population: null,
					message: "실시간 데이터가 제공되지 않는 지점입니다.",
					index: null,
					dataTime: null,
					timestamp: null,
					roadSpeed: null,
					fcstLevel: null,
					fcstTime: null,
				};

				// 2. 주변 장소가 2개 이상이면 추정 계산
				if (nearbyPlaces.length >= 2) {
					let totalWeight = 0;
					let weightedPop = 0;
					let weightedIndex = 0;
					let levelCounts = {};

					nearbyPlaces.forEach(place => {
						// 역거리 가중치 (거리가 가까울수록 가중치 높음)
						const weight = 1 / (place.distance + 100);
						totalWeight += weight;

						weightedPop += (place.stats.population || 0) * weight;
						weightedIndex += (place.stats.index || 0) * weight;

						// 혼잡도 레벨 카운트
						const level = place.stats.level || "정보없음";
						levelCounts[level] = (levelCounts[level] || 0) + weight;
					});

					// 가중 평균 계산
					const estimatedPop = Math.round(weightedPop / totalWeight);
					const estimatedIndex = weightedIndex / totalWeight;

					// 가장 많이 나온 혼잡도 레벨 선택 (가중치 기준)
					const estimatedLevel = Object.entries(levelCounts)
						.sort((a, b) => b[1] - a[1])[0][0];

					const nearestPlace = nearbyPlaces[0];
					const distanceText = nearestPlace.distance < 1000
						? `${Math.round(nearestPlace.distance)}m`
						: `${(nearestPlace.distance / 1000).toFixed(1)}km`;

					estimatedStats = {
						level: estimatedLevel,
						population: estimatedPop,
						message: `주변 ${nearbyPlaces.length}개 장소 데이터 기반 추정 (가장 가까운 곳: ${nearestPlace.place.name}, ${distanceText})`,
						index: estimatedIndex,
						dataTime: nearestPlace.stats.dataTime,
						timestamp: nearestPlace.stats.timestamp,
						roadSpeed: null,
						fcstLevel: null,
						fcstTime: null,
					};

					console.log(`[추정] ${selection?.name}: 인구 ${estimatedPop}명, 혼잡도 ${estimatedLevel}, 주변 ${nearbyPlaces.length}개 장소 기반`);
				}

				item = {
					place: {
						name: selection?.name || selection,
						code: selection?.code || null,
						category: selection?.category || "검색 위치",
					},
					coord,
					stats: estimatedStats,
					updatedAt: null,
				};
			}

			highlightPlace(item, { panTo: true });
		} catch (err) {
			console.error(err);
			setStatus(err.message || "검색에 실패했습니다");
		}
	}, [ensureSelectionCoordinate, highlightPlace]);

	const handleRouteSelection = useCallback(async (type, selection) => {
		const name = typeof selection === "string" ? selection : selection?.name;
		if (!name) return;
		try {
			await adoptRoutePoint(type, selection);
			if (focusItem && focusItem.place?.name === name) return;
			const matched = datasetRef.current.find((item) => item.place.name === name || item.place.code === selection?.code);
			if (matched) {
				highlightPlace(matched, { panTo: false });
			}
		} catch (err) {
			console.error(err);
		}
	}, [adoptRoutePoint, focusItem, highlightPlace]);

	const displayActiveSet = activeCats.size
		? activeCats
		: (!userOverrideRef.current && initialCategoriesRef.current.size
			? initialCategoriesRef.current
			: activeCats);

	const summaryChips = useMemo(() => ([
		{ label: "대상지", value: `${summary.count}/${places.length}` },
		{ label: "총 유동인구", value: summary.totalPopulation ? `${Math.round(summary.totalPopulation).toLocaleString()}명` : "-" },
		{ label: "최고 혼잡", value: summary.busiest ? `${summary.busiest.place.name}` : "-" },
		{ label: "갱신", value: summary.updatedAt ? formatRelative(summary.updatedAt) : "-" },
	]), [places.length, summary]);

	return (
		<div className="app-shell">
			<div className="topbar" style={{ zIndex: 1001 }}>
				<div className="brand">
					<div className="brand-mark" />
					<div>
						<div className="brand-caption">Seoul Realtime Information</div>
						<div className="brand-title">KoreaAiMap</div>
					</div>
				</div>
				<SearchBar
					suggestions={suggestionItems}
					onPick={handleSearchSelection}
					onSearch={(value) => handleSearchSelection(typeof value === "string" ? { name: value } : value)}
					disabled={!mapReady}
				/>
				<div className="chip-row">
					{summaryChips.map((chip) => (
						<div className="chip" key={chip.label}>
							<span>{chip.label}</span>
							<strong>{chip.value}</strong>
						</div>
					))}
					<button
						className="btn primary"
						onClick={drawRoute}
						disabled={!mapReady || !routeLabels.from || !routeLabels.to}
						style={{ marginLeft: '8px' }}
					>
						차량 경로
					</button>
					<button
						className="btn primary"
						onClick={() => setShowTravelPanel(!showTravelPanel)}
						disabled={!mapReady}
						style={{ marginLeft: '8px', background: 'linear-gradient(135deg, #ec4899, #8b5cf6)' }}
					>
						여행지 조회
					</button>
					<button
						className="btn primary"
						onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
						disabled={!mapReady}
						style={{
							marginLeft: '8px',
							background: showAdvancedFilters
								? 'linear-gradient(135deg, #8b5cf6, #6366f1)'
								: 'linear-gradient(135deg, #6366f1, #4f46e5)',
							boxShadow: showAdvancedFilters ? '0 0 20px rgba(139,92,246,0.5)' : 'none'
						}}
					>
						장소 필터 {(activeBusinessTypes.size + activeGenderRatios.size + activeAgeGroups.size + activeResidencyRatios.size) > 0 && `(${activeBusinessTypes.size + activeGenderRatios.size + activeAgeGroups.size + activeResidencyRatios.size})`}
					</button>
				</div>
			</div>

			{/* 중앙 상태 표시 (로딩 중일 때만) */}
			{loading && (
				<div style={{
					position: 'fixed',
					top: '50%',
					left: '50%',
					transform: 'translate(-50%, -50%)',
					zIndex: 1002,
					display: 'flex',
					alignItems: 'center',
					gap: '12px',
					color: '#1f2937',
					fontSize: '20px',
					fontWeight: '700',
					textShadow: '0 2px 4px rgba(255,255,255,0.95), 0 0 15px rgba(255,255,255,0.6)',
					pointerEvents: 'none'
				}}>
					<span className="spinner" aria-hidden />
					<span>{status}</span>
				</div>
			)}

			<div
				id="map"
				className="map-canvas"
			/>

			{/* 지도 줌 컨트롤 */}
			<div
				className="map-zoom-controls"
				style={{
					right: rightPanelOpen ? '256px' : '52px',
					transition: 'right 0.3s ease'
				}}
			>
				<button className="zoom-btn zoom-in" onClick={zoomIn} title="확대">
					+
				</button>
				<button className="zoom-btn zoom-out" onClick={zoomOut} title="축소">
					−
				</button>
			</div>

			<div className={`glass-panel panel-left ${leftPanelOpen ? 'panel-open' : 'panel-closed'}`}>
				<header className="panel-header">
					<button
						className="btn ghost sm panel-toggle-btn"
						onClick={() => setLeftPanelOpen(!leftPanelOpen)}
						title={leftPanelOpen ? "패널 닫기" : "패널 열기"}
					>
						{leftPanelOpen ? '◀' : '▶'}
					</button>
					{leftPanelOpen && (
						<>
							<div>
								<p className="panel-eyebrow">Realtime Population Heatmap</p>
								<h2 className="panel-title">서울시 주요 120 장소</h2>
							</div>
							<button className="btn primary" onClick={() => refreshData(true)} disabled={loading}>새로고침</button>
						</>
					)}
				</header>
				{leftPanelOpen && (
					<>
						<div className="panel-body">
							{/* 🆕 새로운 섹션: 실시간 도시 정보 */}
							<section className="section">
								<div className="section-header">
									<h3>실시간 도시 정보</h3>
									<span className="section-caption">선택한 지역의 종합 정보</span>
								</div>
								{focusItem?.cityData ? (
									<div className="city-info-grid">
										{/* 문화행사 */}
										<div className="info-card">
											<div className="info-card-header">
												<span className="info-icon">🎭</span>
												<h4>문화행사</h4>
											</div>
											<div className="info-card-body">
												{focusItem?.cityData?.events?.length > 0 ? (
													focusItem.cityData.events.slice(0, 7).map((event, idx) => (
														<div key={idx} className="info-item">
															<div className="info-item-title">{event.name}</div>
															<div className="info-item-meta">{event.period}</div>
														</div>
													))
												) : (
													<div className="info-empty">진행중인 행사 없음</div>
												)}
											</div>
										</div>

										{/* 사고통제 */}
										<div className="info-card">
											<div className="info-card-header">
												<span className="info-icon">🚨</span>
												<h4>사고통제</h4>
											</div>
											<div className="info-card-body">
												{focusItem?.cityData?.accidents?.length > 0 ? (
													focusItem.cityData.accidents.map((acc, idx) => (
														<div key={idx} className="info-item alert">
															<div className="info-item-title">{acc.type}</div>
															<div className="info-item-meta">{acc.location}</div>
														</div>
													))
												) : (
													<div className="info-empty success">통제 중인 사고 없음</div>
												)}
											</div>
										</div>

										{/* 상권 현황 */}
										<div className="info-card">
											<div className="info-card-header">
												<span className="info-icon">💳</span>
												<h4>상권 현황</h4>
											</div>
											<div className="info-card-body">
												{focusItem?.cityData?.commerce?.level ? (
													<>
														<div className="info-metric">
															<span>활성도</span>
															<strong>{focusItem.cityData.commerce.level}</strong>
														</div>
														<div className="info-metric">
															<span>결제건수</span>
															<strong>{focusItem.cityData.commerce.paymentCount}건/시간</strong>
														</div>
														<div className="info-metric">
															<span>평균금액</span>
															<strong>{focusItem.cityData.commerce.paymentMin?.toLocaleString()}~{focusItem.cityData.commerce.paymentMax?.toLocaleString()}원</strong>
														</div>
													</>
												) : (
													<div className="info-empty">상권 정보 없음</div>
												)}
											</div>
										</div>

										{/* 날씨 */}
										<div className="info-card">
											<div className="info-card-header">
												<span className="info-icon">🌤️</span>
												<h4>날씨</h4>
											</div>
											<div className="info-card-body">
												{focusItem?.cityData?.weather?.temp ? (
													<>
														<div className="info-metric">
															<span>기온</span>
															<strong>{focusItem.cityData.weather.temp}°C</strong>
														</div>
														<div className="info-metric">
															<span>정보</span>
															<strong>{focusItem.cityData.weather.sky}</strong>
														</div>
													</>
												) : (
													<div className="info-empty">날씨 정보 없음</div>
												)}
											</div>
										</div>

										{/* 미세먼지 - 독립 카드 (전체 너비) */}
										<div className="info-card info-card-full">
											<div className="info-card-header">
												<span className="info-icon">🌫️</span>
												<h4>대기질</h4>
											</div>
											<div className="info-card-body">
												{focusItem?.cityData?.weather ? (
													<AirQuality weather={focusItem.cityData.weather} />
												) : (
													<div className="info-empty">대기질 정보 없음</div>
												)}
											</div>
										</div>
									</div>
								) : (
									<div className="city-info-empty-state">
										<div className="empty-state-icon">
											<div className="empty-state-pin">📍</div>
											<div className="empty-state-pulse"></div>
										</div>
										<h4 className="empty-state-title">장소를 선택해주세요</h4>
										<p className="empty-state-description">
											지도 위의 마커를 클릭하거나<br />
											상단 검색창에서 장소를 검색하세요
										</p>
										<div className="empty-state-hint">
											<span className="hint-item">🎭 문화행사</span>
											<span className="hint-item">🚨 사고통제</span>
											<span className="hint-item">💳 상권정보</span>
											<span className="hint-item">🌤️ 날씨</span>
											<span className="hint-item">🌫️ 대기질</span>
										</div>
									</div>
								)}
							</section>

							{/* 기존 상위 혼잡 지역 섹션 */}
							<section className="section">
								<div className="section-header">
									<h3>상위 혼잡 지역</h3>
									<span className="section-caption">현재 가중치 상위 15개 장소</span>
								</div>
								<div className="toplist-scroll">
									<TopList
										items={topItems}
										onZoomTo={(item) => highlightPlace(item, { panTo: true })}
										onAdoptRoute={(type, item) => handleRouteSelection(type, { name: item.place?.name, code: item.place?.code })}
									/>
								</div>
							</section>

							<section className="section">
								<div className="section-header">
									<h3>경로 안내 (차량)</h3>
									<span className="section-caption">OSRM 기반 추천 경로</span>
								</div>
								<div className="route-grid">
									<PlacePicker
										label={routeLabels.from}
										suggestions={suggestionItems}
										placeholder="출발지를 검색하세요"
										disabled={!mapReady}
										onPick={(item) => handleRouteSelection("from", item)}
										onEnter={(value) => handleRouteSelection("from", value)}
									/>
									<PlacePicker
										label={routeLabels.to}
										suggestions={suggestionItems}
										placeholder="도착지를 검색하세요"
										disabled={!mapReady}
										onPick={(item) => handleRouteSelection("to", item)}
										onEnter={(value) => handleRouteSelection("to", value)}
									/>
									<div className="route-actions">
										<button className="btn primary" onClick={drawRoute} disabled={!mapReady}>ai 추천 경로</button>
										<button className="btn primary" onClick={drawNaverDirection5} disabled={!mapReady}>네이버 추천 경로</button>
										<button className="btn ghost" onClick={() => { pickModeRef.current = "from"; setStatus("지도에서 출발지를 선택하세요"); }}>지도에서 출발 지정</button>
										<button className="btn ghost" onClick={() => { pickModeRef.current = "to"; setStatus("지도에서 도착지를 선택하세요"); }}>지도에서 도착 지정</button>
									</div>
									{routes.length > 0 && (
										<div className="routes-list">
											{routes.map((route, idx) => {
												// AI reason이 있으면 사용, 없으면 자동 생성
												const aiReason = aiScores.find(s => s.id === idx)?.reason || "";
												const autoAnalysis = analyzeRouteCharacteristics(route, routes, idx);
												const displayReason = aiReason || autoAnalysis.recommendation;
												const icon = autoAnalysis.icon;
												const userType = autoAnalysis.userType;

												return (
													<div
														key={idx}
														className={`route-card ${idx === selectedRouteIndex ? 'selected' : ''}`}
														onClick={() => setSelectedRouteIndex(idx)}
														style={{
															borderLeft: `4px solid ${route.color}`,
															cursor: 'pointer',
															opacity: idx === selectedRouteIndex ? 1 : 0.7
														}}
													>
														<div className="route-card-header">
															<span className="route-label">{route.optionName ? route.optionName : `경로 ${idx + 1}`}</span>
															{idx === selectedRouteIndex && <span className="route-badge">선택됨</span>}
															{aiRecIndex === idx && <span className="route-ml-badge">⭐ AI 추천</span>}
														</div>
														<div className="route-metrics">
															<div className="route-metric">
																<span>거리</span>
																<strong>{(route.distance / 1000).toFixed(2)} km</strong>
															</div>
															<div className="route-metric">
																<span>예상 시간</span>
																<strong>{Math.round(route.duration / 60)} 분</strong>
															</div>
															<div className="route-metric">
																<span>평균 속도</span>
																<strong>{autoAnalysis.stats.speed} km/h</strong>
															</div>
														</div>
														<div className="route-reason">
															<div style={{
																display: 'flex',
																alignItems: 'center',
																gap: '6px',
																marginBottom: '6px',
																fontWeight: '600',
																fontSize: '11.2px'
															}}>
																<span>{icon}</span>
																<span>{userType}</span>
															</div>
															<div style={{ fontSize: '10.4px', lineHeight: '1.6' }}>
																{displayReason}
															</div>
														</div>
													</div>
												)})}
										</div>
									)}
								</div>
							</section>

							{/* 🆕 새로운 섹션: 대중교통 경로 안내 */}
							<section className="section">
								<div className="section-header">
									<h3>대중교통 경로 안내</h3>
									<span className="section-caption">지하철 및 버스 추천</span>
								</div>
								<div className="transit-buttons">
									<button className="btn primary" onClick={drawSubwayRoute}>
										🚇 지하철 추천 경로 보기
									</button>
									<button className="btn primary" onClick={showBusStations}>
										🚌 버스 추천 경로 보기
									</button>
								</div>
							</section>

						</div>
						<footer className="panel-footer">
							<div className="footer-status">
								<span className="footer-status-text">{status}</span>
								{loading && <span className="spinner" aria-hidden />}
							</div>
							<div className="footer-attribution">
								<img
									src={gonggongnuriLogo}
									alt="공공누리"
									className="footer-logo"
								/>
								<span className="footer-source">서울특별시-서울 열린데이터 광장, 교통정보센터</span>
							</div>
						</footer>
					</>
				)}
			</div>

			<div className={`glass-panel panel-right ${rightPanelOpen ? 'panel-open' : 'panel-closed'}`}>
				<header className="panel-header panel-header--compact">
					<button
						className="btn ghost sm panel-toggle-btn"
						onClick={() => setRightPanelOpen(!rightPanelOpen)}
						title={rightPanelOpen ? "패널 닫기" : "패널 열기"}
					>
						{rightPanelOpen ? '▶' : '◀'}
					</button>
					{rightPanelOpen && (
						<>
							<div>
								<p className="panel-eyebrow">Realtime Filters</p>
								<h2 className="panel-title">실시간 인구 정보</h2>
							</div>
							<button className="btn ghost sm" onClick={handleSelectAllCategories}>전체 토글</button>
						</>
					)}
				</header>
				{rightPanelOpen && (
					<>
						<div className="panel-body panel-body--filters">
							{focusItem?.cityData?.forecast?.length > 0 && (
								<section className="section forecast-section" style={{ marginTop: '16px' }}>
									<div className="section-header">
										<h3>실시간 인구 및 혼잡도 추이 전망</h3>
										<span className="section-caption">{focusItem.place?.name}</span>
									</div>
									<div className="forecast-chart">
										{focusItem.cityData.forecast.map((f, idx) => {
											const timeStr = f.time || '';
											const hour = timeStr.length >= 13 ? parseInt(timeStr.substring(11, 13)) : 0;
											const time = `${hour}시`;

											const avg = (f.min + f.max) / 2;
											const maxVal = Math.max(...focusItem.cityData.forecast.map(x => (x.min + x.max) / 2));
											const height = (avg / maxVal) * 100;

											// 고급 색상 및 그라디언트
											const levelConfig = f.level?.includes('붐빔') ? {
												gradient: 'linear-gradient(180deg, #dc2626 0%, #ef4444 50%, #b91c1c 100%)',
												shadow: 'rgba(220, 38, 38, 0.6)',
												glow: 'rgba(239, 68, 68, 0.4)'
											} : f.level?.includes('약간') ? {
												gradient: 'linear-gradient(180deg, #ea580c 0%, #f97316 50%, #c2410c 100%)',
												shadow: 'rgba(234, 88, 12, 0.6)',
												glow: 'rgba(249, 115, 22, 0.4)'
											} : f.level?.includes('보통') ? {
												gradient: 'linear-gradient(180deg, #eab308 0%, #fbbf24 50%, #ca8a04 100%)',
												shadow: 'rgba(234, 179, 8, 0.6)',
												glow: 'rgba(251, 191, 36, 0.4)'
											} : {
												gradient: 'linear-gradient(180deg, #059669 0%, #10b981 50%, #047857 100%)',
												shadow: 'rgba(5, 150, 105, 0.6)',
												glow: 'rgba(16, 185, 129, 0.4)'
											};

											return (
												<div key={idx} className="forecast-bar-container">
													<div
														className="forecast-bar"
														style={{
															height: `${height}%`,
															background: levelConfig.gradient,
															boxShadow: `
														0 4px 12px ${levelConfig.shadow},
														inset 0 2px 4px rgba(255, 255, 255, 0.3),
														inset 0 -2px 4px rgba(0, 0, 0, 0.2)
													`
														}}
														title={`${time}: ${f.level} (${f.min.toLocaleString()}-${f.max.toLocaleString()}명)`}
														data-level={f.level}
													>
														<div className="forecast-bar-shine"></div>
														<span className="forecast-value">{Math.round(avg / 1000)}K</span>
														<span className="forecast-time">{time}</span>
													</div>
													{idx % 3 === 0 && (
														<span className="forecast-label">{f.level?.substring(0, 2)}</span>
													)}
												</div>
											);
										})}
									</div>
								</section>
							)}

							{/* 🆕 실시간 인구 구성 비율 */}
							{focusItem?.cityData?.demographics && (
								<section className="section" style={{ marginTop: '16px' }}>
									<div className="section-header">
										<h3>실시간 인구 구성 비율</h3>
										<span className="section-caption">{focusItem.place?.name}</span>
									</div>

									{/* 성별 비율 */}
									<div className="demo-card">
										<h4 className="demo-title">성별 비율</h4>
										<div className="donut-container">
											<div className="donut-chart-wrapper">
												<svg className="donut-chart" viewBox="0 0 100 100">
													{/* 배경 원 - 3D 효과 */}
													<defs>
														<linearGradient id="gradient-male" x1="0%" y1="0%" x2="100%" y2="100%">
															<stop offset="0%" stopColor="#5b5fc7"/>
															<stop offset="50%" stopColor="#6366f1"/>
															<stop offset="100%" stopColor="#4f46e5"/>
														</linearGradient>
														<linearGradient id="gradient-female" x1="0%" y1="0%" x2="100%" y2="100%">
															<stop offset="0%" stopColor="#db2777"/>
															<stop offset="50%" stopColor="#ec4899"/>
															<stop offset="100%" stopColor="#be185d"/>
														</linearGradient>
														<radialGradient id="bg-gradient">
															<stop offset="60%" stopColor="rgba(255,255,255,0.15)"/>
															<stop offset="100%" stopColor="rgba(255,255,255,0.05)"/>
														</radialGradient>
													</defs>
													<circle cx="50" cy="50" r="35" fill="url(#bg-gradient)" stroke="rgba(255,255,255,0.2)" strokeWidth="12"/>
													{/* 남성 세그먼트 */}
													<circle
														cx="50"
														cy="50"
														r="35"
														fill="none"
														stroke="url(#gradient-male)"
														strokeWidth="12"
														strokeDasharray={`${focusItem.cityData.demographics.male * 2.199} ${220 - focusItem.cityData.demographics.male * 2.199}`}
														strokeDashoffset="55"
														className="donut-segment"
														style={{filter: 'drop-shadow(0 2px 4px rgba(99, 102, 241, 0.5))'}}
													/>
													{/* 여성 세그먼트 */}
													<circle
														cx="50"
														cy="50"
														r="35"
														fill="none"
														stroke="url(#gradient-female)"
														strokeWidth="12"
														strokeDasharray={`${focusItem.cityData.demographics.female * 2.199} ${220 - focusItem.cityData.demographics.female * 2.199}`}
														strokeDashoffset={`${55 - focusItem.cityData.demographics.male * 2.199}`}
														className="donut-segment"
														style={{filter: 'drop-shadow(0 2px 4px rgba(236, 72, 153, 0.5))'}}
													/>
													{/* 중앙 텍스트 */}
													<text x="50" y="46" textAnchor="middle" className="donut-center-label">성별</text>
													<text x="50" y="56" textAnchor="middle" className="donut-center-value">비율</text>
												</svg>
											</div>
											<div className="donut-legend">
												<div className="donut-legend-item">
													<span className="donut-legend-color" style={{
														background: 'linear-gradient(135deg, #5b5fc7, #4f46e5)'
													}}></span>
													<span className="donut-legend-label">남성</span>
													<span className="donut-legend-value">{focusItem.cityData.demographics.male}%</span>
												</div>
												<div className="donut-legend-item">
													<span className="donut-legend-color" style={{
														background: 'linear-gradient(135deg, #db2777, #be185d)'
													}}></span>
													<span className="donut-legend-label">여성</span>
													<span className="donut-legend-value">{focusItem.cityData.demographics.female}%</span>
												</div>
											</div>
										</div>
									</div>

									{/* 연령대별 비율 */}
									<div className="demo-card">
										<h4 className="demo-title">연령대별 비율</h4>
										<div className="donut-container">
											<div className="donut-chart-wrapper">
												<svg className="donut-chart" viewBox="0 0 100 100">
													{/* 배경 원 - 3D 효과 */}
													<defs>
														<radialGradient id="bg-gradient-age">
															<stop offset="60%" stopColor="rgba(255,255,255,0.15)"/>
															<stop offset="100%" stopColor="rgba(255,255,255,0.05)"/>
														</radialGradient>
													</defs>
													<circle cx="50" cy="50" r="35" fill="url(#bg-gradient-age)" stroke="rgba(255,255,255,0.2)" strokeWidth="12"/>
													{/* 연령대별 세그먼트 */}
													{(() => {
														const ageGroups = [
															{ key: 'age0', label: '10대↓', color: '#f59e0b', shadow: 'rgba(245, 158, 11, 0.5)' },
															{ key: 'age10', label: '10대', color: '#f97316', shadow: 'rgba(249, 115, 22, 0.5)' },
															{ key: 'age20', label: '20대', color: '#ef4444', shadow: 'rgba(239, 68, 68, 0.5)' },
															{ key: 'age30', label: '30대', color: '#a855f7', shadow: 'rgba(168, 85, 247, 0.5)' },
															{ key: 'age40', label: '40대', color: '#6366f1', shadow: 'rgba(99, 102, 241, 0.5)' },
															{ key: 'age50', label: '50대', color: '#3b82f6', shadow: 'rgba(59, 130, 246, 0.5)' },
															{ key: 'age60', label: '60대', color: '#8b5cf6', shadow: 'rgba(139, 92, 246, 0.5)' },
															{ key: 'age70', label: '70대↑', color: '#10b981', shadow: 'rgba(16, 185, 129, 0.5)' },
														];
														let offset = 55;
														return ageGroups.map((age, idx) => {
															const value = focusItem.cityData.demographics.age[age.key];
															if (value > 0) {
																const dashArray = `${value * 2.199} ${220 - value * 2.199}`;
																const currentOffset = offset;
																offset -= value * 2.199;
																return (
																	<circle
																		key={idx}
																		cx="50"
																		cy="50"
																		r="35"
																		fill="none"
																		stroke={age.color}
																		strokeWidth="12"
																		strokeDasharray={dashArray}
																		strokeDashoffset={currentOffset}
																		className="donut-segment"
																		style={{filter: `drop-shadow(0 2px 4px ${age.shadow})`}}
																	/>
																);
															}
															return null;
														});
													})()}
													{/* 중앙 텍스트 */}
													<text x="50" y="46" textAnchor="middle" className="donut-center-label">연령대</text>
													<text x="50" y="56" textAnchor="middle" className="donut-center-value">분포</text>
												</svg>
											</div>
											<div className="donut-legend">
												{[
													{ key: 'age0', label: '10대↓', color: '#f59e0b' },
													{ key: 'age10', label: '10대', color: '#f97316' },
													{ key: 'age20', label: '20대', color: '#ef4444' },
													{ key: 'age30', label: '30대', color: '#a855f7' },
													{ key: 'age40', label: '40대', color: '#6366f1' },
													{ key: 'age50', label: '50대', color: '#3b82f6' },
													{ key: 'age60', label: '60대', color: '#8b5cf6' },
													{ key: 'age70', label: '70대↑', color: '#10b981' },
												].map((age, idx) => {
													const value = focusItem.cityData.demographics.age[age.key];
													return value > 0 ? (
														<div key={idx} className="donut-legend-item">
															<span className="donut-legend-color" style={{
																backgroundColor: age.color
															}}></span>
															<span className="donut-legend-label">{age.label}</span>
															<span className="donut-legend-value">{value}%</span>
														</div>
													) : null;
												})}
											</div>
										</div>
									</div>

									{/* 상주/비상주 비율 */}
									<div className="demo-card">
										<h4 className="demo-title">상주·비상주 비율</h4>
										<div className="donut-container">
											<div className="donut-chart-wrapper">
												<svg className="donut-chart" viewBox="0 0 100 100">
													{/* 배경 원 - 3D 효과 */}
													<defs>
														<linearGradient id="gradient-resident" x1="0%" y1="0%" x2="100%" y2="100%">
															<stop offset="0%" stopColor="#7e22ce"/>
															<stop offset="50%" stopColor="#9333ea"/>
															<stop offset="100%" stopColor="#6b21a8"/>
														</linearGradient>
														<linearGradient id="gradient-nonresident" x1="0%" y1="0%" x2="100%" y2="100%">
															<stop offset="0%" stopColor="#0891b2"/>
															<stop offset="50%" stopColor="#06b6d4"/>
															<stop offset="100%" stopColor="#0e7490"/>
														</linearGradient>
														<radialGradient id="bg-gradient-resident">
															<stop offset="60%" stopColor="rgba(255,255,255,0.15)"/>
															<stop offset="100%" stopColor="rgba(255,255,255,0.05)"/>
														</radialGradient>
													</defs>
													<circle cx="50" cy="50" r="35" fill="url(#bg-gradient-resident)" stroke="rgba(255,255,255,0.2)" strokeWidth="12"/>
													{/* 상주 세그먼트 */}
													<circle
														cx="50"
														cy="50"
														r="35"
														fill="none"
														stroke="url(#gradient-resident)"
														strokeWidth="12"
														strokeDasharray={`${focusItem.cityData.demographics.resident * 2.199} ${220 - focusItem.cityData.demographics.resident * 2.199}`}
														strokeDashoffset="55"
														className="donut-segment"
														style={{filter: 'drop-shadow(0 2px 4px rgba(147, 51, 234, 0.5))'}}
													/>
													{/* 비상주 세그먼트 */}
													<circle
														cx="50"
														cy="50"
														r="35"
														fill="none"
														stroke="url(#gradient-nonresident)"
														strokeWidth="12"
														strokeDasharray={`${focusItem.cityData.demographics.nonResident * 2.199} ${220 - focusItem.cityData.demographics.nonResident * 2.199}`}
														strokeDashoffset={`${55 - focusItem.cityData.demographics.resident * 2.199}`}
														className="donut-segment"
														style={{filter: 'drop-shadow(0 2px 4px rgba(6, 182, 212, 0.5))'}}
													/>
													{/* 중앙 텍스트 */}
													<text x="50" y="46" textAnchor="middle" className="donut-center-label">거주</text>
													<text x="50" y="56" textAnchor="middle" className="donut-center-value">구분</text>
												</svg>
											</div>
											<div className="donut-legend">
												<div className="donut-legend-item">
													<span className="donut-legend-color" style={{
														background: 'linear-gradient(135deg, #7e22ce, #6b21a8)'
													}}></span>
													<span className="donut-legend-label">상주</span>
													<span className="donut-legend-value">{focusItem.cityData.demographics.resident}%</span>
												</div>
												<div className="donut-legend-item">
													<span className="donut-legend-color" style={{
														background: 'linear-gradient(135deg, #0891b2, #0e7490)'
													}}></span>
													<span className="donut-legend-label">비상주</span>
													<span className="donut-legend-value">{focusItem.cityData.demographics.nonResident}%</span>
												</div>
											</div>
										</div>
									</div>
								</section>
							)}

							<section className="section section--filters" style={{ marginTop: '16px' }}>
								<Controls
									categories={categories}
									active={displayActiveSet}
									onToggle={handleToggleCategory}
									onToggleAll={handleSelectAllCategories}
									showHeatmap={showHeatmap}
									onToggleHeatmap={setShowHeatmap}
									radius={radius}
									onRadius={setRadius}
									opacity={opacity}
									onOpacity={setOpacity}
									pointCount={summary.count}
								/>
							</section>
						</div>
					</>
				)}
			</div>

			{/* Old focus-card removed - now using InfoWindow above marker */}

			{/* 행사 및 상권 정보 패널 */}
			{showEventsCommercePanel && focusItem && (
				<div style={{
					position: 'fixed',
					top: '50%',
					left: '50%',
					transform: 'translate(-50%, -50%)',
					width: '90%',
					maxWidth: '1000px',
					maxHeight: '85vh',
					backgroundColor: 'white',
					borderRadius: '20px',
					boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
					zIndex: 2000,
					display: 'flex',
					flexDirection: 'column',
					overflow: 'hidden'
				}}>
					{/* 헤더 */}
					<div style={{
						padding: '24px',
						borderBottom: '2px solid #e5e7eb',
						background: 'linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%)',
						display: 'flex',
						justifyContent: 'space-between',
						alignItems: 'center'
					}}>
						<div>
							<h2 style={{
								margin: '0 0 8px 0',
								fontSize: '24px',
								fontWeight: '700',
								color: 'white'
							}}>
								{focusItem.place?.name}
							</h2>
							<p style={{
								margin: 0,
								fontSize: '14px',
								color: 'rgba(255,255,255,0.9)'
							}}>
								문화행사 및 상권 정보
							</p>
						</div>
						<button
							onClick={() => setShowEventsCommercePanel(false)}
							style={{
								width: '36px',
								height: '36px',
								border: 'none',
								background: 'rgba(255,255,255,0.2)',
								color: 'white',
								borderRadius: '50%',
								cursor: 'pointer',
								fontSize: '20px',
								fontWeight: 'bold',
								display: 'flex',
								alignItems: 'center',
								justifyContent: 'center',
								transition: 'all 0.2s'
							}}
							onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.3)'}
							onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.2)'}
						>
							×
						</button>
					</div>

					{/* 내용 */}
					<div style={{
						flex: 1,
						overflowY: 'auto',
						padding: '24px'
					}}>
						{/* 문화행사 섹션 */}
						<div style={{ marginBottom: '32px' }}>
							<h3 style={{
								margin: '0 0 20px 0',
								fontSize: '20px',
								fontWeight: '700',
								color: '#111827',
								display: 'flex',
								alignItems: 'center',
								gap: '8px'
							}}>
								<span>🎭</span>
								<span>문화행사</span>
							</h3>
							<div style={{
								display: 'grid',
								gap: '16px'
							}}>
								{focusItem.cityData?.events && focusItem.cityData.events.length > 0 ? (
									focusItem.cityData.events.map((event, idx) => {
										const imgUrl = event.THUMBNAIL || event.MAIN_IMG || event.IMGURL || '';
										return (
											<div key={idx} style={{
												display: 'flex',
												gap: '16px',
												padding: '16px',
												background: '#f9fafb',
												borderRadius: '12px',
												border: '1px solid #e5e7eb'
											}}>
												{/* 왼쪽: 문화행사 내용 */}
												<div style={{ flex: 1 }}>
													<div style={{
														fontSize: '16px',
														fontWeight: '600',
														color: '#111827',
														marginBottom: '8px'
													}}>
														{event.name || event.TITLE || '제목 없음'}
													</div>

													{(event.period || event.DATE || event.EVENT_PERIOD) && (
														<div style={{
															fontSize: '14px',
															color: '#6b7280',
															marginBottom: '4px'
														}}>
															📅 <strong>기간:</strong> {event.period || event.DATE || event.EVENT_PERIOD}
														</div>
													)}

													{(event.PLACE || event.EVENT_PLACE) && (
														<div style={{
															fontSize: '14px',
															color: '#6b7280',
															marginBottom: '4px'
														}}>
															📍 <strong>장소:</strong> {event.PLACE || event.EVENT_PLACE}
														</div>
													)}

													{event.USE_FEE && (
														<div style={{
															fontSize: '14px',
															color: '#6b7280',
															marginBottom: '4px'
														}}>
															💰 <strong>이용료:</strong> {event.USE_FEE}
														</div>
													)}

													{(event.HMPG_ADDR || event.URL) && (
														<div style={{
															marginTop: '12px'
														}}>
															<a
																href={event.HMPG_ADDR || event.URL}
																target="_blank"
																rel="noopener noreferrer"
																style={{
																	display: 'inline-block',
																	padding: '8px 16px',
																	background: '#3b82f6',
																	color: 'white',
																	textDecoration: 'none',
																	borderRadius: '6px',
																	fontSize: '13px',
																	fontWeight: '600',
																	transition: 'background 0.2s'
																}}
																onMouseOver={(e) => e.currentTarget.style.background = '#2563eb'}
																onMouseOut={(e) => e.currentTarget.style.background = '#3b82f6'}
															>
																🔗 상세정보 보기
															</a>
														</div>
													)}
												</div>

												{/* 우측: 이미지 */}
												{imgUrl && (
													<div style={{
														width: '200px',
														flexShrink: 0
													}}>
														<img
															src={imgUrl}
															alt={event.TITLE || event.name}
															style={{
																width: '100%',
																height: '200px',
																objectFit: 'cover',
																borderRadius: '8px'
															}}
															onError={(e) => e.currentTarget.style.display = 'none'}
														/>
													</div>
												)}
											</div>
										);
									})
								) : (
									<div style={{
										padding: '32px',
										textAlign: 'center',
										color: '#9ca3af',
										fontSize: '14px'
									}}>
										등록된 문화행사 정보가 없습니다.
									</div>
								)}
							</div>
						</div>

						{/* 상권 정보 섹션 */}
						<div>
							<h3 style={{
								margin: '0 0 20px 0',
								fontSize: '20px',
								fontWeight: '700',
								color: '#111827',
								display: 'flex',
								alignItems: 'center',
								gap: '8px'
							}}>
								<span>💳</span>
								<span>상권 현황</span>
							</h3>

							{focusItem.cityData?.commerce && focusItem.cityData.commerce.level ? (
								<div>
									{/* 전체 상권 현황 */}
									<div style={{
										padding: '20px',
										background: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)',
										borderRadius: '12px',
										marginBottom: '20px',
										border: '2px solid #fbbf24'
									}}>
										<div style={{
											display: 'grid',
											gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
											gap: '16px'
										}}>
											<div>
												<div style={{ fontSize: '12px', color: '#92400e', marginBottom: '4px' }}>상권 현황</div>
												<div style={{ fontSize: '24px', fontWeight: '700', color: '#78350f' }}>
													{focusItem.cityData.commerce.level}
												</div>
											</div>
											<div>
												<div style={{ fontSize: '12px', color: '#92400e', marginBottom: '4px' }}>결제 건수</div>
												<div style={{ fontSize: '20px', fontWeight: '600', color: '#78350f' }}>
													{focusItem.cityData.commerce.paymentCount || 0} 건
												</div>
											</div>
											<div>
												<div style={{ fontSize: '12px', color: '#92400e', marginBottom: '4px' }}>결제 금액</div>
												<div style={{ fontSize: '16px', fontWeight: '600', color: '#78350f' }}>
													{focusItem.cityData.commerce.paymentMin && focusItem.cityData.commerce.paymentMax ?
														`${(focusItem.cityData.commerce.paymentMin / 10000).toFixed(0)}만 ~ ${(focusItem.cityData.commerce.paymentMax / 10000).toFixed(0)}만원` :
														'정보 없음'}
												</div>
											</div>
										</div>
									</div>

									{/* 업종별 상세 정보 */}
									{focusItem.cityData.commerce.businesses && focusItem.cityData.commerce.businesses.length > 0 && (
										<div style={{ marginBottom: '20px' }}>
											<h4 style={{
												margin: '0 0 12px 0',
												fontSize: '16px',
												fontWeight: '600',
												color: '#374151'
											}}>업종별 상세</h4>
											<div style={{
												display: 'grid',
												gap: '12px'
											}}>
												{focusItem.cityData.commerce.businesses.map((business, idx) => (
													<div key={idx} style={{
														padding: '16px',
														background: '#ffffff',
														borderRadius: '12px',
														border: '1px solid #e5e7eb'
													}}>
														<div style={{
															display: 'flex',
															justifyContent: 'space-between',
															alignItems: 'start',
															marginBottom: '12px'
														}}>
															<div>
																<div style={{ fontSize: '16px', fontWeight: '600', color: '#111827' }}>
																	{business.RSB_LRG_CTGR} / {business.RSB_MID_CTGR}
																</div>
																<div style={{ fontSize: '14px', color: '#8b5cf6', fontWeight: '600', marginTop: '4px' }}>
																	{business.RSB_PAYMENT_LVL}
																</div>
															</div>
															<div style={{
																padding: '4px 12px',
																background: '#f3f4f6',
																borderRadius: '6px',
																fontSize: '12px',
																color: '#6b7280'
															}}>
																가맹점 {business.RSB_MCT_CNT}개
															</div>
														</div>
														<div style={{
															display: 'grid',
															gridTemplateColumns: 'repeat(2, 1fr)',
															gap: '8px',
															fontSize: '13px',
															color: '#6b7280'
														}}>
															<div>결제 건수: <strong>{business.RSB_SH_PAYMENT_CNT}건</strong></div>
															<div>
																결제 금액: <strong>
																{business.RSB_SH_PAYMENT_AMT_MIN && business.RSB_SH_PAYMENT_AMT_MAX ?
																	`${(business.RSB_SH_PAYMENT_AMT_MIN / 10000).toFixed(0)}만~${(business.RSB_SH_PAYMENT_AMT_MAX / 10000).toFixed(0)}만원` :
																	'정보없음'}
															</strong>
															</div>
														</div>
													</div>
												))}
											</div>
										</div>
									)}

									{/* 소비 통계 */}
									<div style={{
										display: 'grid',
										gridTemplateColumns: 'repeat(2, 1fr)',
										gap: '16px'
									}}>
										{/* 성별 비율 */}
										{(focusItem.cityData.commerce.maleRate !== undefined || focusItem.cityData.commerce.femaleRate !== undefined) && (
											<div style={{
												padding: '16px',
												background: '#f0f9ff',
												borderRadius: '12px',
												border: '1px solid #bfdbfe'
											}}>
												<div style={{
													fontSize: '14px',
													fontWeight: '600',
													color: '#1e40af',
													marginBottom: '12px'
												}}>성별 소비 비율</div>
												<div style={{ fontSize: '13px', color: '#1e3a8a' }}>
													남성: <strong>{focusItem.cityData.commerce.maleRate?.toFixed(1) || 0}%</strong>
													<br />
													여성: <strong>{focusItem.cityData.commerce.femaleRate?.toFixed(1) || 0}%</strong>
												</div>
											</div>
										)}

										{/* 연령대 비율 */}
										{(focusItem.cityData.commerce.age20Rate !== undefined || focusItem.cityData.commerce.age30Rate !== undefined) && (
											<div style={{
												padding: '16px',
												background: '#fef3c7',
												borderRadius: '12px',
												border: '1px solid #fde68a'
											}}>
												<div style={{
													fontSize: '14px',
													fontWeight: '600',
													color: '#92400e',
													marginBottom: '12px'
												}}>연령대별 소비 비율</div>
												<div style={{
													fontSize: '13px',
													color: '#78350f',
													display: 'grid',
													gridTemplateColumns: 'repeat(2, 1fr)',
													gap: '4px'
												}}>
													{focusItem.cityData.commerce.age10Rate > 0 && <div>10대: <strong>{focusItem.cityData.commerce.age10Rate}%</strong></div>}
													{focusItem.cityData.commerce.age20Rate > 0 && <div>20대: <strong>{focusItem.cityData.commerce.age20Rate}%</strong></div>}
													{focusItem.cityData.commerce.age30Rate > 0 && <div>30대: <strong>{focusItem.cityData.commerce.age30Rate}%</strong></div>}
													{focusItem.cityData.commerce.age40Rate > 0 && <div>40대: <strong>{focusItem.cityData.commerce.age40Rate}%</strong></div>}
													{focusItem.cityData.commerce.age50Rate > 0 && <div>50대: <strong>{focusItem.cityData.commerce.age50Rate}%</strong></div>}
													{focusItem.cityData.commerce.age60Rate > 0 && <div>60대+: <strong>{focusItem.cityData.commerce.age60Rate}%</strong></div>}
												</div>
											</div>
										)}
									</div>

									{/* 업데이트 시간 */}
									{focusItem.cityData.commerce.time && (
										<div style={{
											marginTop: '16px',
											fontSize: '12px',
											color: '#9ca3af',
											textAlign: 'right'
										}}>
											마지막 업데이트: {focusItem.cityData.commerce.time}
										</div>
									)}
								</div>
							) : (
								<div style={{
									padding: '32px',
									textAlign: 'center',
									color: '#9ca3af',
									fontSize: '14px',
									background: '#f9fafb',
									borderRadius: '12px'
								}}>
									상권 정보가 없습니다.
								</div>
							)}
						</div>
					</div>
				</div>
			)}

			{/* 여행지 조회 패널 */}
			{showTravelPanel && (
				<div
					style={{
						position: 'fixed',
						top: '80px',
						left: '50%',
						transform: 'translateX(-50%)',
						width: '323px',
						maxHeight: 'calc(100vh - 102px)',
						backgroundColor: 'white',
						borderRadius: '14px',
						boxShadow: '0 7px 27px rgba(0, 0, 0, 0.12)',
						zIndex: 1500,
						display: 'flex',
						flexDirection: 'column',
						overflow: 'hidden'
					}}
				>
					{/* 헤더 */}
					<div style={{
						padding: '17px',
						borderBottom: '2px solid #e5e7eb',
						background: 'linear-gradient(135deg, #ec4899, #8b5cf6)',
					}}>
						<div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
							<h2 style={{
								margin: 0,
								fontSize: '15px',
								fontWeight: '700',
								color: 'white'
							}}>
								🎭 여행지 조회
							</h2>
							<button
								onClick={() => setShowTravelPanel(false)}
								style={{
									width: '24px',
									height: '24px',
									border: 'none',
									background: 'rgba(255,255,255,0.2)',
									color: 'white',
									borderRadius: '50%',
									cursor: 'pointer',
									fontSize: '15px',
									fontWeight: 'bold',
									display: 'flex',
									alignItems: 'center',
									justifyContent: 'center',
									transition: 'all 0.2s'
								}}
								onMouseOver={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.3)'}
								onMouseOut={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.2)'}
							>
								×
							</button>
						</div>
					</div>

					{/* 내용 */}
					<div style={{
						flex: 1,
						overflowY: 'auto',
						padding: '17px'
					}}>
						{/* 여행지 토글 */}
						<div style={{
							marginBottom: '17px',
							padding: '14px',
							background: 'linear-gradient(135deg, #fef3c7, #fde68a)',
							borderRadius: '10px',
							border: '2px solid #fbbf24'
						}}>
							<div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
								<span style={{ fontWeight: '600', color: '#92400e', fontSize: '13px' }}>🎭 여행지 마커 표시</span>
								<label style={{
									position: 'relative',
									display: 'inline-block',
									width: '42px',
									height: '22px',
									cursor: 'pointer'
								}}>
									<input
										type="checkbox"
										checked={showCulturalMarkers}
										onChange={(e) => setShowCulturalMarkers(e.target.checked)}
										style={{ opacity: 0, width: 0, height: 0 }}
									/>
									<span style={{
										position: 'absolute',
										top: 0,
										left: 0,
										right: 0,
										bottom: 0,
										backgroundColor: showCulturalMarkers ? '#10b981' : '#d1d5db',
										borderRadius: '22px',
										transition: 'background-color 0.3s'
									}}>
										<span style={{
											position: 'absolute',
											content: '',
											height: '17px',
											width: '17px',
											left: showCulturalMarkers ? '23px' : '3px',
											bottom: '2px',
											backgroundColor: 'white',
											borderRadius: '50%',
											transition: 'left 0.3s'
										}} />
									</span>
								</label>
							</div>
						</div>

						{/* 필터 섹션 */}
						<div style={{
							marginBottom: '17px',
							padding: '14px',
							background: '#f9fafb',
							borderRadius: '10px',
							border: '1px solid #e5e7eb'
						}}>
							<h3 style={{ margin: '0 0 10px 0', fontSize: '12px', fontWeight: '600', color: '#374151' }}>
								🔍 필터
							</h3>

							{/* 경로 주변 여부 */}
							<div style={{ marginBottom: '14px', display: 'none' }}>
								<label style={{
									display: 'flex',
									alignItems: 'center',
									gap: '7px',
									cursor: 'pointer',
									fontSize: '11px',
									color: '#4b5563'
								}}>
									<input
										type="checkbox"
										checked={filterNearRoute}
										onChange={(e) => setFilterNearRoute(e.target.checked)}
										disabled={routePathsRef.current.length === 0}
										style={{ cursor: 'pointer' }}
									/>
									<span>📍 경로 주변만 보기 (1.5km)</span>
								</label>
								{routePathsRef.current.length === 0 && (
									<p style={{ margin: '3px 0 0 20px', fontSize: '9px', color: '#9ca3af' }}>
										* 차량/버스/지하철 경로가 있을 때 활성화
									</p>
								)}
							</div>

							{/* 이용료 필터 */}
							<div>
								<p style={{ margin: '0 0 7px 0', fontSize: '11px', fontWeight: '600', color: '#374151' }}>
									💰 이용료
								</p>
								<div style={{
									display: 'grid',
									gridTemplateColumns: 'repeat(2, 1fr)',
									gap: '7px'
								}}>
									{[
										{ value: 'all', label: '전체' },
										{ value: 'free', label: '무료' },
										{ value: '1-5000', label: '1~5천원' },
										{ value: '5001-10000', label: '5천~1만원' },
										{ value: '10001-20000', label: '1만~2만원' },
										{ value: '20001-30000', label: '2만~3만원' },
										{ value: '30000+', label: '3만원 이상' },
										{ value: 'other', label: '기타' }
									].map(option => (
										<label
											key={option.value}
											style={{
												display: 'flex',
												alignItems: 'center',
												gap: '5px',
												padding: '7px',
												borderRadius: '7px',
												border: filterFeeCategories.has(option.value) ? '2px solid #8b5cf6' : '1px solid #e5e7eb',
												background: filterFeeCategories.has(option.value) ? '#f3e8ff' : 'white',
												cursor: 'pointer',
												fontSize: '10px',
												transition: 'all 0.2s',
												color: filterFeeCategories.has(option.value) ? '#7c3aed' : '#374151'
											}}
											onClick={() => {
												const newSet = new Set(filterFeeCategories);
												if (option.value === 'all') {
													newSet.clear();
													newSet.add('all');
												} else {
													newSet.delete('all');
													if (newSet.has(option.value)) {
														newSet.delete(option.value);
													} else {
														newSet.add(option.value);
													}
													if (newSet.size === 0) {
														newSet.add('all');
													}
												}
												setFilterFeeCategories(newSet);
											}}
										>
											<input
												type="checkbox"
												checked={filterFeeCategories.has(option.value)}
												readOnly
												style={{ cursor: 'pointer' }}
											/>
											<span style={{ fontWeight: '600' }}>{option.label}</span>
										</label>
									))}
								</div>
							</div>
						</div>

						{/* 여행지 목록 버튼 */}
						<button
							onClick={() => setShowDestinationList(true)}
							disabled={filteredCulturalEvents.length === 0}
							style={{
								width: '100%',
								padding: '12px',
								border: 'none',
								borderRadius: '10px',
								background: filteredCulturalEvents.length > 0 ? 'linear-gradient(135deg, #8b5cf6, #6366f1)' : '#d1d5db',
								color: 'white',
								fontSize: '12px',
								fontWeight: '600',
								cursor: filteredCulturalEvents.length > 0 ? 'pointer' : 'not-allowed',
								transition: 'all 0.2s',
								boxShadow: filteredCulturalEvents.length > 0 ? '0 3px 10px rgba(139, 92, 246, 0.3)' : 'none'
							}}
							onMouseOver={(e) => {
								if (filteredCulturalEvents.length > 0) {
									e.currentTarget.style.transform = 'translateY(-2px)';
									e.currentTarget.style.boxShadow = '0 5px 14px rgba(139, 92, 246, 0.4)';
								}
							}}
							onMouseOut={(e) => {
								e.currentTarget.style.transform = 'translateY(0)';
								e.currentTarget.style.boxShadow = filteredCulturalEvents.length > 0 ? '0 3px 10px rgba(139, 92, 246, 0.3)' : 'none';
							}}
						>
							📋 여행지 목록 보기 ({filteredCulturalEvents.length}개)
						</button>

						{filteredCulturalEvents.length === 0 && culturalEvents.length === 0 && (
							<p style={{
								marginTop: '10px',
								textAlign: 'center',
								fontSize: '10px',
								color: '#9ca3af'
							}}>
								출발지와 도착지를 선택한 후<br />
								차량/버스/지하철 경로를 표시하면<br />
								경로 주변 문화행사가 자동으로 로드됩니다.
							</p>
						)}
					</div>
				</div>
			)}

			{/* 여행지 목록 패널 */}
			{showDestinationList && (
				<div
					className="destination-list-overlay"
					onClick={() => setShowDestinationList(false)}
					style={{
						position: 'fixed',
						top: 0,
						left: 0,
						right: 0,
						bottom: 0,
						backgroundColor: 'rgba(0, 0, 0, 0.5)',
						zIndex: 2000,
						display: 'flex',
						justifyContent: 'flex-end',
						alignItems: 'stretch'
					}}
				>
					<div
						className="destination-list-panel"
						onClick={(e) => e.stopPropagation()}
						style={{
							width: '425px',
							maxWidth: '90vw',
							backgroundColor: 'white',
							boxShadow: '-3px 0 17px rgba(0, 0, 0, 0.15)',
							overflowY: 'auto',
							animation: 'slideInRight 0.3s ease-out'
						}}
					>
						{/* 헤더 */}
						<div style={{
							position: 'sticky',
							top: 0,
							backgroundColor: 'white',
							borderBottom: '2px solid #e5e7eb',
							padding: '17px',
							zIndex: 10
						}}>
							<div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
								<h2 style={{
									margin: 0,
									fontSize: '17px',
									fontWeight: '700',
									background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
									WebkitBackgroundClip: 'text',
									WebkitTextFillColor: 'transparent'
								}}>
									📋 여행지 목록
								</h2>
								<button
									onClick={() => setShowDestinationList(false)}
									style={{
										width: '27px',
										height: '27px',
										border: 'none',
										background: '#f3f4f6',
										color: '#6b7280',
										borderRadius: '50%',
										cursor: 'pointer',
										fontSize: '17px',
										fontWeight: 'bold',
										display: 'flex',
										alignItems: 'center',
										justifyContent: 'center',
										transition: 'all 0.2s'
									}}
									onMouseOver={(e) => {
										e.currentTarget.style.background = '#e5e7eb';
										e.currentTarget.style.color = '#374151';
									}}
									onMouseOut={(e) => {
										e.currentTarget.style.background = '#f3f4f6';
										e.currentTarget.style.color = '#6b7280';
									}}
								>
									×
								</button>
							</div>
							<p style={{
								margin: '7px 0 0 0',
								fontSize: '12px',
								color: '#6b7280'
							}}>
								{searchedCulturalEvents.length > 0 && searchedCulturalEvents[0].distanceFromStart !== undefined
									? `출발지에서 가까운 순서대로 정렬되었습니다 (${searchedCulturalEvents.length}개)`
									: `총 ${searchedCulturalEvents.length}개의 문화행사가 있습니다`}
							</p>
						</div>

						{/* 검색 입력 */}
						<div style={{ padding: '0 17px 14px 17px', borderBottom: '1px solid #e5e7eb' }}>
							<input
								type="text"
								placeholder="제목, 장소, 카테고리로 검색..."
								value={destinationSearchQuery}
								onChange={(e) => setDestinationSearchQuery(e.target.value)}
								style={{
									width: '100%',
									padding: '10px 14px',
									border: '2px solid #e5e7eb',
									borderRadius: '7px',
									fontSize: '12px',
									outline: 'none',
									transition: 'border-color 0.2s'
								}}
								onFocus={(e) => e.currentTarget.style.borderColor = '#8b5cf6'}
								onBlur={(e) => e.currentTarget.style.borderColor = '#e5e7eb'}
							/>
						</div>

						{/* 문화 행사 목록 */}
						<div style={{ padding: '14px' }}>
							{searchedCulturalEvents.slice(0, destinationListLimit).map((event, index) => {
								const imgUrl = event.MAIN_IMG || event.THUMBNAIL || event.IMGURL || '';
								const distance = event.distanceFromStart;
								const distanceText = distance !== undefined
									? (distance < 1 ? `${Math.round(distance * 1000)}m` : `${distance.toFixed(1)}km`)
									: '';

								return (
									<div
										key={index}
										style={{
											backgroundColor: 'white',
											borderRadius: '10px',
											boxShadow: '0 2px 7px rgba(0, 0, 0, 0.1)',
											marginBottom: '14px',
											overflow: 'hidden',
											border: '1px solid #e5e7eb',
											transition: 'all 0.2s'
										}}
										onMouseOver={(e) => {
											e.currentTarget.style.boxShadow = '0 3px 14px rgba(0, 0, 0, 0.15)';
											e.currentTarget.style.transform = 'translateY(-2px)';
										}}
										onMouseOut={(e) => {
											e.currentTarget.style.boxShadow = '0 2px 7px rgba(0, 0, 0, 0.1)';
											e.currentTarget.style.transform = 'translateY(0)';
										}}
									>
										{/* 순번 배지 */}
										<div style={{
											position: 'relative',
											backgroundColor: '#f9fafb',
											padding: '7px 14px',
											borderBottom: '1px solid #e5e7eb',
											display: 'flex',
											justifyContent: 'space-between',
											alignItems: 'center'
										}}>
											<span style={{
												fontSize: '10px',
												fontWeight: '700',
												color: '#6366f1',
												background: 'linear-gradient(135deg, #8b5cf6, #6366f1)',
												padding: '3px 8px',
												borderRadius: '10px',
											}}>
												#{index + 1}
											</span>
											{distanceText && (
												<span style={{
													fontSize: '10px',
													fontWeight: '600',
													color: '#6b7280'
												}}>
													📍 출발지에서 {distanceText}
												</span>
											)}
										</div>

										{/* 이미지 */}
										{imgUrl && (
											<img
												src={imgUrl}
												alt={event.TITLE || event.EVENT_NM}
												style={{
													width: '100%',
													height: '170px',
													objectFit: 'cover'
												}}
												onError={(e) => e.target.style.display = 'none'}
											/>
										)}

										{/* 내용 */}
										<div style={{ padding: '14px' }}>
											<h3 style={{
												margin: '0 0 7px 0',
												fontSize: '14px',
												fontWeight: '700',
												color: '#1f2937',
												lineHeight: '1.4'
											}}>
												{event.TITLE || event.EVENT_NM || '제목 없음'}
											</h3>

											{event.CODENAME && (
												<div style={{
													display: 'inline-block',
													padding: '3px 8px',
													background: 'linear-gradient(135deg, #ec4899, #8b5cf6)',
													color: 'white',
													borderRadius: '10px',
													fontSize: '9px',
													fontWeight: '600',
													marginBottom: '10px'
												}}>
													{event.CODENAME}
												</div>
											)}

											{(event.DATE || event.EVENT_PERIOD) && (
												<p style={{
													margin: '7px 0',
													fontSize: '11px',
													color: '#6b7280'
												}}>
													<strong style={{ color: '#374151' }}>📅 기간:</strong> {event.DATE || event.EVENT_PERIOD}
												</p>
											)}

											{(event.PLACE || event.EVENT_PLACE) && (
												<p style={{
													margin: '7px 0',
													fontSize: '11px',
													color: '#6b7280'
												}}>
													<strong style={{ color: '#374151' }}>📍 장소:</strong> {event.PLACE || event.EVENT_PLACE}
												</p>
											)}

											{event.ORG_NAME && (
												<p style={{
													margin: '7px 0',
													fontSize: '11px',
													color: '#6b7280'
												}}>
													<strong style={{ color: '#374151' }}>🏢 주최:</strong> {event.ORG_NAME}
												</p>
											)}

											{event.USE_TRGT && (
												<p style={{
													margin: '7px 0',
													fontSize: '11px',
													color: '#6b7280'
												}}>
													<strong style={{ color: '#374151' }}>👥 대상:</strong> {event.USE_TRGT}
												</p>
											)}

											{event.USE_FEE && (
												<p style={{
													margin: '7px 0',
													fontSize: '11px',
													color: '#6b7280'
												}}>
													<strong style={{ color: '#374151' }}>💰 이용료:</strong> {event.USE_FEE}
												</p>
											)}

											{(event.ETC_DESC || event.EVENT_ETC_DETAIL) && (
												<p style={{
													margin: '10px 0 7px 0',
													fontSize: '10px',
													color: '#6b7280',
													lineHeight: '1.5',
													borderTop: '1px solid #e5e7eb',
													paddingTop: '10px'
												}}>
													{(event.ETC_DESC || event.EVENT_ETC_DETAIL).substring(0, 170)}
													{(event.ETC_DESC || event.EVENT_ETC_DETAIL).length > 170 ? '...' : ''}
												</p>
											)}

											{(event.HMPG_ADDR || event.URL) && (
												<a
													href={event.HMPG_ADDR || event.URL}
													target="_blank"
													rel="noopener noreferrer"
													style={{
														display: 'inline-block',
														marginTop: '10px',
														padding: '7px 14px',
														background: '#3b82f6',
														color: 'white',
														textDecoration: 'none',
														borderRadius: '5px',
														fontSize: '11px',
														fontWeight: '600',
														transition: 'background 0.2s'
													}}
													onMouseOver={(e) => e.currentTarget.style.background = '#2563eb'}
													onMouseOut={(e) => e.currentTarget.style.background = '#3b82f6'}
												>
													자세히 보기 →
												</a>
											)}
										</div>
									</div>
								);
							})}

							{/* 더 보기 버튼 */}
							{destinationListLimit < searchedCulturalEvents.length && (
								<button
									onClick={() => setDestinationListLimit(prev => prev + 50)}
									style={{
										width: '100%',
										padding: '14px',
										backgroundColor: '#f3f4f6',
										border: '1px solid #e5e7eb',
										borderRadius: '8px',
										color: '#374151',
										fontSize: '14px',
										fontWeight: '600',
										cursor: 'pointer',
										transition: 'all 0.2s',
										marginTop: '8px'
									}}
									onMouseOver={(e) => {
										e.currentTarget.style.backgroundColor = '#e5e7eb';
										e.currentTarget.style.borderColor = '#d1d5db';
									}}
									onMouseOut={(e) => {
										e.currentTarget.style.backgroundColor = '#f3f4f6';
										e.currentTarget.style.borderColor = '#e5e7eb';
									}}
								>
									더 보기 ({searchedCulturalEvents.length - destinationListLimit}개 남음)
								</button>
							)}
						</div>
					</div>
				</div>
			)}
			{/* 고급 필터 모달 */}
			{showAdvancedFilters && (
				<AdvancedFilters
					categories={categories}
					activeCats={displayActiveSet}
					onToggleCategory={handleToggleCategory}
					businessTypes={businessTypeOptions}
					activeBusinessTypes={activeBusinessTypes}
					onToggleBusinessType={handleToggleBusinessType}
					genderRatios={genderRatioOptions}
					activeGenderRatios={activeGenderRatios}
					onToggleGenderRatio={handleToggleGenderRatio}
					ageGroups={ageGroupOptions}
					activeAgeGroups={activeAgeGroups}
					onToggleAgeGroup={handleToggleAgeGroup}
					residencyRatios={residencyRatioOptions}
					activeResidencyRatios={activeResidencyRatios}
					onToggleResidencyRatio={handleToggleResidencyRatio}
					onResetAll={handleResetAllFilters}
					onClose={() => setShowAdvancedFilters(false)}
				/>
			)}
		</div>
	);
}
