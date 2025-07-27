import React from "react";
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { FaArrowUp, FaArrowDown } from "react-icons/fa";
import Header from "../components/Header";

// --- Reusable Card Component ---
const Card = ({ children, className }) => (
  <div
    className={`bg-white p-6 rounded-xl border border-gray-200 shadow-sm ${
      className || ""
    }`}
  >
    {children}
  </div>
);

// --- Chart Data ---
const spendingByCategoryData = [
  { name: "Food", value: 400 },
  { name: "Shopping", value: 300 },
  { name: "Travel", value: 200 },
  { name: "Utilities", value: 150 },
];

const paymentMethodData = [
  { name: "Debit Card", value: 50 },
  { name: "Cash", value: 25 },
  { name: "Other", value: 25 },
];

const mostFrequentMerchantsData = [
  { name: "Reliance", value: 45 },
  { name: "Amazon", value: 35 },
  { name: "Swiggy", value: 20 },
];

const spendingOverTimeData = [
  { name: "Jan", spending: 2400 },
  { name: "Feb", spending: 2800 },
  { name: "Mar", spending: 4200 },
  { name: "Apr", spending: 3500 },
  { name: "May", spending: 4800 },
];

// --- Colors ---
const CATEGORY_COLORS = {
  Food: "#0088FE",
  Shopping: "#FFBB28",
  Travel: "#00C49F",
  Utilities: "#A8D582",
};

const PAYMENT_METHOD_COLORS = {
  "Debit Card": "#0088FE",
  Cash: "#CFD8DC",
  Other: "#82B1FF",
};

const MERCHANT_COLORS = ["#0088FE", "#82B1FF", "#CFD8DC"];

// --- Dashboard Component ---
const Dashboard = () => {
  return (
    <>
      <Header />
      <div className="bg-gray-50 min-h-screen flex flex-col">
      <div className="max-w-7xl w-full mx-auto p-4 sm:p-8 flex-grow">
        {/* Top Stats */}
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

        {/* Main Grid */}
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
                      <Pie
                        data={spendingByCategoryData}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={60}
                        innerRadius={30}
                      >
                        {spendingByCategoryData.map((entry) => (
                          <Cell key={entry.name} fill={CATEGORY_COLORS[entry.name]} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <ul className="w-1/2 space-y-2">
                  {spendingByCategoryData.map((item) => (
                    <li key={item.name} className="flex items-center">
                      <span
                        className="w-3 h-3 rounded-full mr-2"
                        style={{ backgroundColor: CATEGORY_COLORS[item.name] }}
                      ></span>
                      {item.name}
                    </li>
                  ))}
                </ul>
              </div>
            </Card>

            <Card>
              <h2 className="font-bold text-lg mb-4">Spending Over Time</h2>
              <div className="h-48">
                <ResponsiveContainer>
                  <LineChart
                    data={spendingOverTimeData}
                    margin={{ top: 5, right: 20, left: -10, bottom: 5 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} />
                    <YAxis
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(value) => `₹${value / 1000}k`}
                    />
                    <Tooltip formatter={(value) => `₹${value.toLocaleString()}`} />
                    <Line
                      type="monotone"
                      dataKey="spending"
                      stroke="#0088FE"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      activeDot={{ r: 6 }}
                    />
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
                      <Pie
                        data={paymentMethodData}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={60}
                      >
                        {paymentMethodData.map((entry) => (
                          <Cell key={entry.name} fill={PAYMENT_METHOD_COLORS[entry.name]} />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <ul className="w-1/2 space-y-2">
                  {paymentMethodData.map((item) => (
                    <li key={item.name} className="flex items-center">
                      <span
                        className="w-3 h-3 rounded-full mr-2"
                        style={{ backgroundColor: PAYMENT_METHOD_COLORS[item.name] }}
                      ></span>
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
                  {mostFrequentMerchantsData.map((item) => (
                    <li key={item.name}>{item.name}</li>
                  ))}
                </ul>
                <div className="w-1/3 h-24">
                  <ResponsiveContainer>
                    <PieChart>
                      <Pie
                        data={mostFrequentMerchantsData}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        innerRadius={20}
                      >
                        {mostFrequentMerchantsData.map((entry, index) => (
                          <Cell
                            key={entry.name}
                            fill={MERCHANT_COLORS[index % MERCHANT_COLORS.length]}
                          />
                        ))}
                      </Pie>
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <button className="w-full bg-blue-600 text-white font-bold py-2 px-4 rounded-lg mt-4 hover:bg-blue-700 transition-colors">
                Export Data
              </button>
            </Card>
          </div>
        </main>
      </div>
    </div>
    </>
    
  );
};

export default Dashboard;
