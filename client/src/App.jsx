import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import PrivateRoute from './components/PrivateRoute';
import DashboardPage from './pages/Dashboard';
import CaptureReceiptPage from './pages/CaptureReceipt';
import Chat from './pages/Chat';
import SignInPage from './pages/SignIn';
import SignUpPage from './pages/SignUp';
import Home from './pages/Home';
import './App.css';
import ReceiptPassPreview from './components/AnalysisResultCard';

export default function App() {
  return (
    <div className="bg-slate-50 min-h-screen font-sans text-gray-800">
      <Router>
        <Routes>
          {/* Public Routes */}
          <Route path="/" element={<Home />} />
          <Route path="/signin" element={<SignInPage />} />
          <Route path="/signup" element={<SignUpPage />} />

          {/* Private Routes */}
          <Route
            path="/dashboard"
            element={
              <PrivateRoute>
                <DashboardPage />
              </PrivateRoute>
            }
          />
          <Route
            path="/capture"
            element={
              <PrivateRoute>
                <div className="max-w-md mx-auto">
                  <CaptureReceiptPage />
                </div>
              </PrivateRoute>
            }
          />
          <Route
            path="/ask"
            element={
              <PrivateRoute>
                <div>
                  <Chat />
                </div>
              </PrivateRoute>
            }
          />

          <Route path="*" element={<Navigate to="/" />} />
        </Routes>
      </Router>
    </div>
  );
}
