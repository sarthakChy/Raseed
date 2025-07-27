import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import PrivateRoute from './components/PrivateRoute';

import SignInPage from './pages/SignIn';
import SignUpPage from './pages/SignUp';

import Dashboard from './pages/Dashboard';
import History from './pages/History';
import ScanReceipts from './pages/ScanReceipts';
import ReceiptResult from './pages/ReceiptResult';
import Chatbot from './pages/Chatbot';

import Hero from './pages/Hero';
import About from './pages/About';
import Upgrade from './pages/Upgrade';
import FAQs from './pages/FAQs';
import Contact from './pages/Contact';

import './App.css';
import GetStarted from './pages/GetStarted';

export default function App() {
  return (
    <Router>
      <div className="bg-[#FEFBF6] min-h-screen text-gray-900 overflow-x-hidden">
        <div className="max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8 ">
          <main className="py-4 ">
            <Routes>
              {/* Public Routes */}
              <Route path="/" element={<Hero />} />
              <Route path="/about" element={<About />} />
              <Route path="/upgrade" element={<Upgrade />} />
              <Route path="/faq" element={<FAQs />} />
              <Route path="/contact" element={<Contact />} />
              <Route path="/signin" element={<SignInPage />} />
              <Route path="/signup" element={<SignUpPage />} />

              {/* Private Routes */}
              <Route path="/chatbot" element={
                <PrivateRoute>
                  <Chatbot />
                </PrivateRoute>
              } />
              <Route path="/dashboard" element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              } />
              <Route path="/history" element={
                <PrivateRoute>
                  <History />
                </PrivateRoute>
              } />
              <Route path="/scanreceipts" element={
                <PrivateRoute>
                  <ScanReceipts />
                </PrivateRoute>
              } />
              <Route path="/receipt-result" element={
                <PrivateRoute>
                  <ReceiptResult />
                </PrivateRoute>
              } />
              <Route path="/getstarted" element={
                <PrivateRoute>
                  <GetStarted />
                </PrivateRoute>
              } />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}
