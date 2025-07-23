import React from 'react';

const Upgrade = () => {
  return (
    <div className="bg-white h-full font-sans">
      <main className="p-10">
        <h1 className="text-4xl font-bold text-gray-800 mb-8">Upgrade Plan</h1>
        <div className="max-w-4xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 border border-gray-200 rounded-lg overflow-hidden">
            <div className="p-8">
              <h2 className="text-3xl font-bold text-green-600">Free</h2>
              <p className="text-2xl font-semibold text-gray-700 mb-6">₹ 0 / mo</p>
              <ul className="space-y-6 text-gray-600">
                <li>Receipts per month</li>
                <li>Devices</li>
                <li>Basic analysis</li>
                <li>Export to Excel/PDF</li>
                <li>Telegram insights</li>
                <li>AI-based suggestions</li>
              </ul>
            </div>
            <div className="p-8 border-l border-gray-200 bg-blue-50 rounded-r-lg">
              <h2 className="text-3xl font-bold text-blue-600">Pro</h2>
              <p className="text-2xl font-semibold text-gray-700 mb-6">₹ 149 / mo</p>
              <ul className="space-y-6">
                <li className="flex justify-between items-center">
                  <span></span>
                  <span className="text-gray-700 font-medium">Unlimited</span>
                </li>
                <li className="flex justify-between items-center">
                  <span></span>
                  <span className="text-gray-700 font-medium">1</span>
                </li>
                <li className="flex justify-end items-center">
                  <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </li>
                <li className="flex justify-end items-center">
                  <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </li>
                <li className="flex justify-end items-center">
                  <svg className="w-6 h-6 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                  </svg>
                </li>
                <li className="flex justify-end items-center pt-4">
                  <button className="bg-blue-600 text-white px-8 py-3 rounded-lg font-semibold w-full hover:bg-blue-700">
                    Upgrade
                  </button>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default Upgrade;
