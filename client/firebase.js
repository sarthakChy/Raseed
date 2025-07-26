import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

// Log all window.env variables (in production build)
console.log("[firebase.js] window.env:", window.env);

// Build and log Firebase config from injected env.js
const firebaseConfig = {
  apiKey: window.env?.VITE_FIREBASE_API_KEY,
  authDomain: window.env?.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: window.env?.VITE_FIREBASE_PROJECT_ID,
  storageBucket: window.env?.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: window.env?.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: window.env?.VITE_FIREBASE_APP_ID,
  measurementId: window.env?.VITE_FIREBASE_MEASUREMENT_ID,
};

console.log("[firebase.js] Firebase config:", firebaseConfig);

// Initialize Firebase
const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
