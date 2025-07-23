import React from 'react';
import { useNavigate } from 'react-router-dom';

// --- ICON COMPONENTS ---
const ScanIcon: React.FC = () => (
  <svg viewBox="0 0 100 100" className="w-20 h-20" aria-hidden="true">
    <circle cx="50" cy="50" r="40" fill="none" stroke="#4285F4" strokeWidth="15" strokeDasharray="251.2" />
    <circle cx="50" cy="50" r="40" fill="none" stroke="#34A853" strokeWidth="15" strokeDasharray="251.2" strokeDashoffset="62.8" />
    <circle cx="50" cy="50" r="40" fill="none" stroke="#FBBC05" strokeWidth="15" strokeDasharray="251.2" strokeDashoffset="125.6" />
    <circle cx="50" cy="50" r="40" fill="none" stroke="#EA4335" strokeWidth="15" strokeDasharray="251.2" strokeDashoffset="188.4" />
  </svg>
);

const AskIcon: React.FC = () => (
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

const HistoryIcon: React.FC = () => (
    <svg viewBox="0 0 100 100" className="w-20 h-20" aria-hidden="true">
        <path d="M50,10A40,40,0,1,0,90,50,40,40,0,0,0,50,10ZM50,82A32,32,0,1,1,82,50,32,32,0,0,1,50,82Z" fill="#4A90E2"/>
        <path d="M55,25H45V52.5l17.68,10.2,5-8.66L55,50Z" fill="#4A90E2"/>
    </svg>
);

const DashboardIcon: React.FC = () => (
    <svg viewBox="0 0 100 100" className="w-20 h-20" aria-hidden="true">
        <rect x="15" y="15" width="30" height="40" rx="5" fill="#34A853"/>
        <rect x="55" y="15" width="30" height="25" rx="5" fill="#FBBC05"/>
        <rect x="15" y="65" width="70" height="20" rx="5" fill="#EA4335"/>
    </svg>
);


// --- MAIN COMPONENT ---
const GetStarted: React.FC = () => {
  const navigate = useNavigate();

  const openChatbot = () => {
    window.open('/chatbot', '_blank'); // ✅ Opens new tab
  };

  return (
    <div className="bg-gray-50 min-h-screen flex flex-col items-center justify-center p-5 font-sans">
      <h1 className="text-5xl font-bold text-gray-800 mb-10 text-center">
        Get Started
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Card 1: Scan a receipt */}
        <button
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

        {/* Card 3: Receipts History */}
        <button
          className="bg-white rounded-3xl shadow-md hover:shadow-xl transition-shadow duration-300 p-8 w-72 h-72 flex flex-col justify-center items-center space-y-6"
          aria-label="Receipts History"
        >
          <HistoryIcon />
          <span className="text-2xl font-semibold text-gray-700 text-center">
            Receipts History
          </span>
        </button>

        {/* Card 4: Dashboard */}
        <button
        onClick={() => navigate('/dashboard')}
          className="bg-white rounded-3xl shadow-md hover:shadow-xl transition-shadow duration-300 p-8 w-72 h-72 flex flex-col justify-center items-center space-y-6"
          aria-label="Dashboard"
        >
          <DashboardIcon />
          <span className="text-2xl font-semibold text-gray-700 text-center">
            Dashboard
          </span>
        </button>

      </div>
       
        <div className="bg-white shadow-sm rounded-full flex items-center justify-between p-4 mb-8 mt-7 max-w-xl w-full">
        <p className="text-gray-800 pl-4">Unlock more with Premium Subscription.</p>
        <button className="bg-yellow-400 text-black font-bold py-2 px-6 rounded-full hover:bg-yellow-500 transition-colors duration-300">
          Upgrade Now
        </button>
      </div>
    </div>

    
  );
};

export default GetStarted;