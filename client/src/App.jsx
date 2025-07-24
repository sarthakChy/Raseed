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

import Header from './components/Header'; // Assuming these exist
import Hero from './components/Hero';
import About from './pages/About';
import Upgrade from './pages/Upgrade';
import FAQs from './pages/FAQs';
import Contact from './pages/Contact';
<<<<<<< HEAD
=======
import Upgrade from './pages/Upgrade';
import GetStarted from './pages/GetStarted';
import Chatbot from './pages/Chatbot';
import SignIn from './pages/SignIn';
import SignUp from './pages/SignUp';
import './App.css';
import PrivateRoute from './components/PrivateRoute';
import Dashboard from './pages/Dashboard';
import History from "./pages/History";
import ScanReceipts from './pages/ScanReceipts';
import ReceiptResult from './pages/ReceiptResult';
>>>>>>> 72e950a (I have no idea what is this at this point)

import './App.css';
import GetStarted from './pages/GetStarted';

export default function App() {
  return (
    <Router>
      <div className="bg-[#FEFBF6] min-h-screen text-gray-900 overflow-x-hidden hide-scrollbar">
        <div className="container h-screen mx-auto px-6 lg:px-8">
          <Header />
          <main className="h-4/5">
            <Routes>
              {/* Public Routes */}
              <Route path="/" element={<Hero />} />
              <Route path="/about" element={<About />} />
              <Route path="/getstarted" element={<GetStarted />} />
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
<<<<<<< HEAD
=======

>>>>>>> 72e950a (I have no idea what is this at this point)
              <Route path="/receipt-result" element={
                <PrivateRoute>
                  <ReceiptResult />
                </PrivateRoute>
              } />
<<<<<<< HEAD
=======

              <Route path="/signin" element={<SignIn />} />
              <Route path="/signup" element={<SignUp />} />
>>>>>>> 72e950a (I have no idea what is this at this point)
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
}
