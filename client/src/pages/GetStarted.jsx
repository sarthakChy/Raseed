import { useNavigate } from 'react-router-dom';
import React from 'react';
// --- ICON COMPONENTS ---
const ScanIcon = () => (
  <svg viewBox="0 0 100 100" className="w-20 h-20" aria-hidden="true">
    <circle cx="50" cy="50" r="40" fill="none" stroke="#4285F4" strokeWidth="15" strokeDasharray="251.2" />
    <circle cx="50" cy="50" r="40" fill="none" stroke="#34A853" strokeWidth="15" strokeDasharray="251.2" strokeDashoffset="62.8" />
    <circle cx="50" cy="50" r="40" fill="none" stroke="#FBBC05" strokeWidth="15" strokeDasharray="251.2" strokeDashoffset="125.6" />
    <circle cx="50" cy="50" r="40" fill="none" stroke="#EA4335" strokeWidth="15" strokeDasharray="251.2" strokeDashoffset="188.4" />
  </svg>
);

const AskIcon = () => (
  <svg viewBox="0 0 100 100" className="w-20 h-20" aria-hidden="true">
    <path
      d="M85,10H15C9.477,10,5,14.477,5,20v40c0,5.523,4.477,10,10,10h15v15l15-15h40c5.523,0,10-4.477,10-10V20C95,14.477,90.523,10,85,10z"
      fill="none"
      stroke="#374151"
      strokeWidth="5"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <line x1="25" y1="30" x2="75" y2="30" stroke="#EA4335" strokeWidth="7" strokeLinecap="round" />
    <line x1="25" y1="45" x2="75" y2="45" stroke="#34A853" strokeWidth="7" strokeLinecap="round" />
    <line x1="25" y1="60" x2="55" y2="60" stroke="#FBBC05" strokeWidth="7" strokeLinecap="round" />
  </svg>
);

// --- MAIN COMPONENT ---
const GetStarted = () => {
  const navigate = useNavigate();

  const openChatbot = () => {
    window.open('/chatbot', '_blank');
  };

  return (
    <div className="bg-gray-50 h-full flex flex-col items-center justify-center p-10 font-sans">
      <h1 className="text-5xl font-bold text-gray-800 mb-12 text-center">
        Get Started
      </h1>

      <div className="flex flex-col md:flex-row space-y-8 md:space-y-0 md:space-x-10">
        {/* Card 1: Scan a receipt */}
        <button
          onClick={() => navigate('/capture')}
          className="bg-white rounded-3xl shadow-md hover:shadow-xl transition-shadow duration-300 p-8 w-72 h-72 flex flex-col justify-center items-center space-y-6"
          aria-label="Scan a receipt"
        >
          <ScanIcon />
          <span className="text-2xl font-semibold text-gray-700 text-center">
            Scan a receipt
          </span>
        </button>

        {/* Card 2: Ask questions */}
        <button
          onClick={openChatbot}
          className="bg-white rounded-3xl shadow-md hover:shadow-xl transition-shadow duration-300 p-8 w-72 h-72 flex flex-col justify-center items-center space-y-6"
          aria-label="Ask questions"
        >
          <AskIcon />
          <span className="text-2xl font-semibold text-gray-700 text-center">
            Ask questions
          </span>
        </button>
      </div>
    </div>
  );
};

export default GetStarted;
