import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { prefetchPopularPlaces } from './utils/prefetch.js'

// Service Worker 등록 함수
function registerServiceWorker() {
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker
        .register('/sw.js')
        .then((registration) => {
          console.log('[App] Service Worker registered successfully:', registration.scope);

          // Service Worker가 활성화되면 프리페치 시작
          if (registration.active) {
            console.log('[App] Service Worker is active, starting prefetch...');
            prefetchPopularPlaces();
          }

          // 업데이트 확인
          registration.addEventListener('updatefound', () => {
            const newWorker = registration.installing;
            console.log('[App] New Service Worker found, installing...');

            newWorker.addEventListener('statechange', () => {
              if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                console.log('[App] New Service Worker installed, ready to activate');
                // 새 버전이 설치되면 자동으로 활성화
                newWorker.postMessage({ type: 'SKIP_WAITING' });
              }

              // 새 Service Worker가 활성화되면 프리페치
              if (newWorker.state === 'activated') {
                console.log('[App] Service Worker activated, starting prefetch...');
                prefetchPopularPlaces();
              }
            });
          });
        })
        .catch((error) => {
          console.error('[App] Service Worker registration failed:', error);
          // Service Worker 없이도 프리페치 시도 (브라우저 캐시 활용)
          prefetchPopularPlaces();
        });

      // Service Worker 활성화 시 알림 (자동 새로고침 비활성화)
      navigator.serviceWorker.addEventListener('controllerchange', () => {
        console.log('[App] New Service Worker activated. Refresh the page to use the latest version.');
      });
    });
  } else {
    console.warn('[App] Service Workers are not supported in this browser');
    // Service Worker 미지원 브라우저에서도 브라우저 캐시 활용
    window.addEventListener('load', () => {
      prefetchPopularPlaces();
    });
  }
}

// Service Worker 등록
registerServiceWorker();

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
