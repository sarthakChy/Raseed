import { initializeApp } from "firebase/app";
import { getAuth } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSyBueTG4oGQNUGPE4SmPI_HIoIK8--y3XBQ",
  authDomain: "massive-incline-466204-t5.firebaseapp.com",
  projectId: "massive-incline-466204-t5",
  storageBucket: "massive-incline-466204-t5.firebasestorage.app",
  messagingSenderId: "499686015140",
  appId: "1:499686015140:web:4b321c1aba3bbf8664e631",
  measurementId: "G-PM6HE1FGCJ"
};


const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
