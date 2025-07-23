import React from 'react';
import { 
  FaSearch, 
  FaAngleDown, 
  FaListUl, 
  FaGoogle, 
  FaRegListAlt, 
  FaTable, 
  FaShoppingCart, 
  FaUtensils, 
  FaCheck, 
  FaBan 
} from 'react-icons/fa';

// --- Type Definitions ---
type Receipt = {
  id: number;
  icon: React.ElementType;
  storeName: string;
  date: string;
  amount: string;
  subText: string;
  category: 'Groceries' | 'Dining' | 'Electronics' | 'Not Added';
};

// --- Mock Data ---
const receiptsData: Receipt[] = [
  {
    id: 1,
    icon: FaListUl,
    storeName: 'Supermart',
    date: 'Jul 20, 2025',
    amount: '₹1,234.56',
    subText: 'Total av',
    category: 'Groceries',
  },
  {
    id: 2,
    icon: FaGoogle,
    storeName: 'Coffee Cafe',
    date: 'Jul 12, 2025',
    amount: '₹180.00',
    subText: 'Total av',
    category: 'Dining',
  },
  {
    id: 3,
    icon: FaTable,
    storeName: 'Store Name',
    date: 'Jun 30, 2025',
    amount: '₹2,500.00',
    subText: 'Total av',
    category: 'Electronics',
  },
  {
    id: 4,
    icon: FaRegListAlt,
    storeName: 'Market Bazaar',
    date: 'June 21, 2025',
    amount: '₹325.00',
    subText: 'Groceries',
    category: 'Not Added',
  },
];

// --- Reusable Category Tag Component ---
const CategoryTag = ({ category }: { category: Receipt['category'] }) => {
  const styles = {
    Groceries: {
      bg: 'bg-green-100',
      text: 'text-green-700',
      icon: <FaShoppingCart />,
    },
    Dining: {
      bg: 'bg-yellow-100',
      text: 'text-yellow-700',
      icon: <FaUtensils />,
    },
    Electronics: {
      bg: 'bg-green-100',
      text: 'text-green-700',
      icon: <FaCheck />,
    },
    'Not Added': {
      bg: 'bg-gray-200',
      text: 'text-gray-600',
      icon: <FaBan />,
    },
  };

  const style = styles[category];

  return (
    <div className={`flex items-center space-x-2 py-1 px-3 rounded-full text-sm font-medium ${style.bg} ${style.text}`}>
      {style.icon}
      <span>{category}</span>
    </div>
  );
};

// --- Reusable Receipt Row Component ---
const ReceiptRow = ({ receipt }: { receipt: Receipt }) => (
  <div className="flex items-center space-x-4 p-4 border-b border-gray-200 last:border-b-0">
    <div className="bg-gray-100 p-3 rounded-lg text-gray-600">
      <receipt.icon size={20} />
    </div>
    <div className="flex-grow">
      <p className="font-bold text-lg text-gray-800">{receipt.storeName}</p>
      <p className="text-sm text-gray-500">{receipt.date}</p>
    </div>
    <div className="text-right">
      <p className="font-bold text-lg text-gray-800">{receipt.amount}</p>
      <p className="text-sm text-gray-500">{receipt.subText}</p>
    </div>
    <div className="w-40 flex justify-center">
      <CategoryTag category={receipt.category} />
    </div>
    <div className="flex items-center space-x-6 text-gray-600 font-medium">
      <button className="hover:text-blue-600">View</button>
      <button className="hover:text-red-600">Delete</button>
    </div>
  </div>
);


// --- Main History Component ---
const History = () => {
  return (
    <div className="bg-gray-50 min-h-screen p-4 sm:p-8">
      <div className="max-w-7xl mx-auto bg-white p-8 rounded-2xl border border-gray-200">
        
        {/* Header Section */}
        <header className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-4 sm:mb-0">
            Receipts History
          </h1>
          <div className="flex items-center space-x-3 w-full sm:w-auto">
            <div className="relative flex-grow">
              <FaSearch className="absolute top-1/2 left-3 transform -translate-y-1/2 text-gray-400" />
              <input 
                type="text" 
                placeholder="Search by store or something..."
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
            </div>
            <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg bg-white">
              <span>All</span>
              <FaAngleDown />
            </button>
            <button className="flex items-center space-x-2 px-4 py-2 border border-gray-300 rounded-lg bg-white">
              <span>All dates</span>
              <FaAngleDown />
            </button>
          </div>
        </header>

        {/* Receipts List */}
        <div className="bg-white rounded-lg border border-gray-200">
          {receiptsData.map(receipt => (
            <ReceiptRow key={receipt.id} receipt={receipt} />
          ))}
        </div>

        {/* Action Buttons */}
        <footer className="flex justify-start items-center space-x-4 mt-8">
          <button className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors">
            Edit Receipts
          </button>
          <button className="bg-white text-gray-800 font-bold py-3 px-6 rounded-lg border-2 border-gray-300 hover:bg-gray-100 transition-colors">
            View Dashboard
          </button>
          <button className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors">
            Export Data
          </button>
        </footer>

      </div>
    </div>
  );
};

export default History;