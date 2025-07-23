import React from 'react';
import { PieChart, Pie, Cell, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { FaAngleDown, FaArrowUp, FaArrowDown } from 'react-icons/fa';

// --- Reusable Card Component ---
const Card = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <div className={`bg-white p-6 rounded-xl border border-gray-200 shadow-sm ${className}`}>
    {children}
  </div>
);

// --- Data for Charts ---
const spendingByCategoryData = [
  { name: 'Food', value: 400 },
  { name: 'Shopping', value: 300 },
  { name: 'Travel', value: 200 },
  { name: 'Utilities', value: 150 },
];

const paymentMethodData = [
    { name: 'Debit Card', value: 50 },
    { name: 'Cash', value: 25 },
    { name: 'Other', value: 25 },
];

const mostFrequentMerchantsData = [
    { name: 'Reliance', value: 45 },
    { name: 'Amazon', value: 35 },
    { name: 'Swiggy', value: 20 },
];

const spendingOverTimeData = [
  { name: 'Jan', spending: 2400 },
  { name: 'Feb', spending: 2800 },
  { name: 'Mar', spending: 4200 },
  { name: 'Apr', spending: 3500 },
  { name: 'May', spending: 4800 },
];

// --- Color Palettes (FIXED) ---
// By separating palettes, we ensure that we only access string values for single-color styles, resolving the TypeScript error.
const CATEGORY_COLORS = {
  Food: '#0088FE',
  Shopping: '#FFBB28',
  Travel: '#00C49F',
  Utilities: '#A8D582',
};

const PAYMENT_METHOD_COLORS = {
  'Debit Card': '#0088FE',
  'Cash': '#CFD8DC', // A light grey for cash
  'Other': '#82B1FF',
};

const MERCHANT_COLORS = ['#0088FE', '#82B1FF', '#CFD8DC'];


// --- Main Dashboard Component ---
const Dashboard = () => {
  return (
    <div className="bg-gray-50 min-h-screen p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800">Dashboard</h1>
          <button className="bg-white border border-gray-300 text-gray-700 font-medium py-2 px-4 rounded-lg flex items-center space-x-2">
            <span>Apr 1, 2024 - Apr 20, 2024</span>
            <FaAngleDown />
          </button>
        </header>

        {/* Top Stats Cards */}
        <section className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
          <Card>
            <p className="text-gray-500 mb-1">Receipts Scanned</p>
            <p className="text-4xl font-semibold text-gray-800">28</p>
          </Card>
          <Card>
            <p className="text-gray-500 mb-1">Total Spent</p>
            <div className="flex justify-between items-center">
              <p className="text-4xl font-semibold text-gray-800">₹56,000</p>
              <div className="flex items-center text-green-600 font-semibold">
                <FaArrowUp className="mr-1" /> 4.2%
              </div>
            </div>
          </Card>
          <Card>
            <p className="text-gray-500 mb-1">Saved</p>
            <div className="flex justify-between items-center">
              <p className="text-4xl font-semibold text-gray-800">₹5,400</p>
              <div className="flex items-center text-green-600 font-semibold">
                <FaArrowUp className="mr-1" /> 5.5%
              </div>
            </div>
          </Card>
        </section>

        {/* Main Content Grid */}
        <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Left Column */}
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <h2 className="font-bold text-lg mb-3">Spending Pattern Analysis</h2>
              <ul className="space-y-3 text-gray-700">
                <li className="flex items-center">
                  <FaArrowDown className="text-red-500 mr-2" /> 4.2% increase over weekends
                </li>
                <li>• You spend more on shopping after your salary credit date</li>
              </ul>
            </Card>
            <Card>
              <h2 className="font-bold text-lg mb-3">Smart Insights & Recommendations</h2>
              <ul className="space-y-2 text-gray-700 list-disc list-inside">
                <li>You spent 25% more on food this month</li>
                <li>Amazon was your most used merchant (8 receipts)</li>
              </ul>
            </Card>
            <Card>
              <h2 className="font-bold text-lg mb-3">Pro Tips</h2>
              <ul className="space-y-2 text-gray-700 list-disc list-inside">
                <li>Consider limiting your entertainment budget by 15%.</li>
              </ul>
            </Card>
          </div>

          {/* Right Column */}
          <div className="lg:col-span-2 grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <h2 className="font-bold text-lg mb-4">Spending by Category</h2>
              <div className="flex items-center">
                <div className="w-1/2 h-40">
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie data={spendingByCategoryData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60} innerRadius={30}>
                        {spendingByCategoryData.map((entry) => <Cell key={`cell-${entry.name}`} fill={CATEGORY_COLORS[entry.name as keyof typeof CATEGORY_COLORS]} />)}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <ul className="w-1/2 space-y-2">
                  {spendingByCategoryData.map(item => (
                     <li key={item.name} className="flex items-center"><span className="w-3 h-3 rounded-full mr-2" style={{backgroundColor: CATEGORY_COLORS[item.name as keyof typeof CATEGORY_COLORS]}}></span>{item.name}</li>
                  ))}
                </ul>
              </div>
            </Card>
            <Card>
              <h2 className="font-bold text-lg mb-4">Spending Over Time</h2>
              <div className="h-48">
                 <ResponsiveContainer>
                    <LineChart data={spendingOverTimeData} margin={{ top: 5, right: 20, left: -10, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false}/>
                        <XAxis dataKey="name" tickLine={false} axisLine={false}/>
                        <YAxis tickLine={false} axisLine={false} tickFormatter={(value) => `₹${value/1000}k`} />
                        <Tooltip formatter={(value: number) => `₹${value.toLocaleString()}`} />
                        <Line type="monotone" dataKey="spending" stroke="#0088FE" strokeWidth={2} dot={{ r: 4 }} activeDot={{ r: 6 }} />
                    </LineChart>
                 </ResponsiveContainer>
              </div>
            </Card>
            <Card>
              <h2 className="font-bold text-lg mb-4">Payment Method Breakdown</h2>
               <div className="flex items-center">
                <div className="w-1/2 h-40">
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie data={paymentMethodData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={60}>
                         {paymentMethodData.map((entry) => <Cell key={`cell-${entry.name}`} fill={PAYMENT_METHOD_COLORS[entry.name as keyof typeof PAYMENT_METHOD_COLORS]} />)}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <ul className="w-1/2 space-y-2">
                    {paymentMethodData.map(item => (
                        <li key={item.name} className="flex items-center">
                            <span className="w-3 h-3 rounded-full mr-2" style={{backgroundColor: PAYMENT_METHOD_COLORS[item.name as keyof typeof PAYMENT_METHOD_COLORS]}}></span>
                            {item.name}: {item.value}%
                        </li>
                    ))}
                </ul>
              </div>
            </Card>
             <Card>
              <h2 className="font-bold text-lg mb-4">Most Frequent Merchants</h2>
              <div className="flex items-center justify-between">
                <ul className="space-y-2 text-gray-700">
                    {mostFrequentMerchantsData.map(item => <li key={item.name}>{item.name}</li>)}
                </ul>
                <div className="w-1/3 h-24">
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie data={mostFrequentMerchantsData} dataKey="value" nameKey="name" cx="50%" cy="50%" innerRadius={20}>
                         {mostFrequentMerchantsData.map((entry, index) => <Cell key={`cell-${entry.name}`} fill={MERCHANT_COLORS[index % MERCHANT_COLORS.length]} />)}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
               <button className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-lg mt-4 hover:bg-blue-700 transition-colors">Export Data</button>
            </Card>
          </div>

        </main>

      </div>
    </div>
  );
};

export default Dashboard;