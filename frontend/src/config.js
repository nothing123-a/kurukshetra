// API Configuration
const isDevelopment = import.meta.env.DEV || window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
const LOCALHOST_API = `${window.location.protocol}//${window.location.hostname || 'localhost'}:8000`

// In production (Docker/EC2), VITE_API_BASE_URL should be empty
// so all API calls go through Nginx reverse proxy as relative URLs (/api/...)
// In development, calls go directly to localhost:8000
let apiUrl = import.meta.env.VITE_API_BASE_URL || (isDevelopment ? LOCALHOST_API : '')

// Fix any malformed URLs (only if apiUrl is non-empty)
if (apiUrl && !apiUrl.startsWith('http://') && !apiUrl.startsWith('https://')) {
  apiUrl = `http://${apiUrl}`
}

export const API_BASE_URL = apiUrl

console.log('🔗 API Base URL:', API_BASE_URL || '(relative - using Nginx proxy)')

// Environment-based configuration
export const config = {
  apiBaseUrl: API_BASE_URL,
  isDevelopment,
  // Add other configuration options here
};
