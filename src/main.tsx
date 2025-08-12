import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.tsx'
import './index.css'

// Add global error handler for WebLLM issues in production
if (import.meta.env.PROD) {
  window.addEventListener('error', (event) => {
    // Check if the error is related to WebLLM
    if (event.error && (
      event.error.message?.includes('WebLLM') ||
      event.error.message?.includes('Object.keys') ||
      event.error.message?.includes('Cannot convert undefined or null to object')
    )) {
      console.warn('WebLLM error caught by global handler:', event.error);
      // Prevent the error from crashing the app
      event.preventDefault();
    }
  });

  window.addEventListener('unhandledrejection', (event) => {
    // Check if the promise rejection is related to WebLLM
    if (event.reason && (
      event.reason.message?.includes('WebLLM') ||
      event.reason.message?.includes('Object.keys') ||
      event.reason.message?.includes('Cannot convert undefined or null to object')
    )) {
      console.warn('WebLLM promise rejection caught by global handler:', event.reason);
      // Prevent the rejection from crashing the app
      event.preventDefault();
    }
  });
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
