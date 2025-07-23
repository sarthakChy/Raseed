import React from 'react';

// --- Icon Components (Self-contained SVGs for easy use) ---

const RaseedLogoIcon = () => (
  <div className="w-7 h-7 grid grid-cols-2 grid-rows-2 gap-0.5">
    <div className="bg-blue-500 rounded-tl-md"></div>
    <div className="bg-red-500 rounded-tr-md"></div>
    <div className="bg-yellow-400 rounded-bl-md"></div>
    <div className="bg-green-500 rounded-br-md"></div>
  </div>
);

const PlusIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2.5" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
  </svg>
);

const BackArrowIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M17 8l4 4m0 0l-4 4m4-4H3" />
  </svg>
);

const ReceiptIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const ReportIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
  </svg>
);

const RobotIcon = () => (
  <div className="w-10 h-10 flex-shrink-0 bg-blue-500 rounded-full flex items-center justify-center">
    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  </div>
);

const MoreIcon = () => (
  <svg className="w-6 h-6 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
    <path d="M6 10a2 2 0 11-4 0 2 2 0 014 0zM12 10a2 2 0 11-4 0 2 2 0 014 0zM16 12a2 2 0 100-4 2 2 0 000 4z" />
  </svg>
);

const MicIcon = () => (
  <svg className="w-5 h-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
  </svg>
);

const SendIcon = () => (
  <svg className="w-5 h-5 text-white" fill="currentColor" viewBox="0 0 20 20">
    <path fillRule="evenodd" clipRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" />
  </svg>
);

const InputModeIcon = () => (
  <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);

// --- Main Chatbot Component ---

const Chatbot = () => {
  return (
    <div className="font-sans flex w-full max-w-6xl h-full mx-auto bg-white border border-gray-200 rounded-lg shadow-lg">
      {/* Sidebar */}
      <aside className="w-full max-w-xs bg-gray-50 p-6 border-r border-gray-200 flex flex-col space-y-8">
        <div className="flex items-center space-x-2">
          <RaseedLogoIcon />
          <span className="text-2xl font-bold text-gray-800">RASEED</span>
        </div>

        <button className="flex items-center space-x-3 text-lg font-semibold text-gray-700 w-full p-2 rounded-md hover:bg-gray-200 text-left">
          <PlusIcon />
          <span>New Chat</span>
        </button>

        <div className="flex-grow">
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider">My Conversations</h3>
          <ul className="mt-3 space-y-1 text-gray-700">
            <li className="bg-blue-100 text-blue-800 font-semibold p-2 rounded-md cursor-pointer">Chat on: Amazon Rec...</li>
            <li className="p-2 rounded-md hover:bg-gray-100 cursor-pointer">Chat on: Travel Expenses June</li>
          </ul>
        </div>

        <div>
          <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider">Quick Links</h3>
          <ul className="mt-3 space-y-2 text-gray-600 font-medium">
            <li className="flex items-center space-x-3 hover:text-black cursor-pointer"><BackArrowIcon /> <span>Back to Dashboard</span></li>
            <li className="flex items-center space-x-3 hover:text-black cursor-pointer"><ReceiptIcon /> <span>My Receipts</span></li>
            <li className="flex items-center space-x-3 hover:text-black cursor-pointer"><ReportIcon /> <span>Reports</span></li>
          </ul>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="w-full flex flex-col bg-white">
        <header className="p-4 border-b border-gray-200 flex justify-between items-center flex-shrink-0">
          <h2 className="text-lg font-semibold text-gray-800">Chat on: Amazon Receipts</h2>
          <button><MoreIcon /></button>
        </header>

        <div className="flex-1 p-6 space-y-6 overflow-y-auto">
          <div className="flex justify-end">
            <div className="bg-gray-100 p-3 rounded-lg max-w-lg">
              <p className="text-sm text-gray-800">How much did I spend on Amazon this month?</p>
            </div>
          </div>

          <div className="flex items-start space-x-3">
            <RobotIcon />
            <div className="bg-gray-100 p-3 rounded-lg max-w-lg">
              <p className="font-bold text-sm text-gray-900">Raseed Assistant</p>
              <p className="text-sm text-gray-800">You spent $1,200 on Amazon this month.</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 pt-4">
            <button className="bg-blue-500 text-white font-semibold py-3 px-4 rounded-lg hover:bg-blue-600 text-sm">Summarize My Receipts</button>
            <button className="bg-yellow-400 text-black font-semibold py-3 px-4 rounded-lg hover:bg-yellow-500 text-sm">Find Unusual Transactions</button>
            <button className="bg-red-500 text-white font-semibold py-3 px-4 rounded-lg hover:bg-red-600 text-sm">Top Spending Categories</button>
            <button className="bg-green-500 text-white font-semibold py-3 px-4 rounded-lg hover:bg-green-600 text-sm">Download Report</button>
          </div>
        </div>

        <div className="p-4 border-t border-gray-200 flex-shrink-0">
          <div className="flex items-center space-x-3 bg-white border border-gray-300 rounded-lg p-2 focus-within:ring-2 focus-within:ring-blue-500">
            <InputModeIcon />
            <input 
              type="text" 
              placeholder="What's my highest spend this week?" 
              className="w-full bg-transparent focus:outline-none text-sm"
            />
            <button><MicIcon /></button>
            <button className="bg-blue-500 p-2 rounded-md hover:bg-blue-600"><SendIcon /></button>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Chatbot;
