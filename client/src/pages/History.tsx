import React, { useState, useEffect } from 'react';
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
  FaBan,
  FaLaptop,
  FaLeaf,
  FaTshirt,
  FaHeartbeat,
  FaBus,
  FaBolt,
  FaQuestionCircle,
} from 'react-icons/fa';
import { getAuth } from 'firebase/auth';
import ReceiptDetailsDialog from '../components/ReceiptDetailsDialog';
import { useNavigate } from 'react-router-dom';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

// --- Category Tag Component ---
const CategoryTag = ({ category }) => {
  const styles = {
    Shopping: {
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
      bg: 'bg-blue-100',
      text: 'text-blue-700',
      icon: <FaLaptop />,
    },
    Grocery: {
      bg: 'bg-lime-100',
      text: 'text-lime-700',
      icon: <FaLeaf />,
    },
    Groceries: {
      bg: 'bg-lime-100',
      text: 'text-lime-700',
      icon: <FaLeaf />,
    },
    Clothing: {
      bg: 'bg-pink-100',
      text: 'text-pink-700',
      icon: <FaTshirt />,
    },
    Healthcare: {
      bg: 'bg-red-100',
      text: 'text-red-700',
      icon: <FaHeartbeat />,
    },
    Travel: {
      bg: 'bg-cyan-100',
      text: 'text-cyan-700',
      icon: <FaBus />,
    },
    Utilities: {
      bg: 'bg-indigo-100',
      text: 'text-indigo-700',
      icon: <FaBolt />,
    },
    'Not Added': {
      bg: 'bg-gray-200',
      text: 'text-gray-600',
      icon: <FaQuestionCircle />, // better icon for unknown
    },
  };

  const { bg, text, icon } = styles[category] || styles['Not Added'];

  return (
    <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${bg} ${text}`}>
      {icon} {category}
    </span>
  );
};

// --- Receipt Row Component ---
const ReceiptRow = ({ receipt, onView }) => (
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
      <button className="hover:text-blue-600" onClick={() => onView(receipt)}>View</button>
      <button className="hover:text-red-600">Delete</button>
    </div>
  </div>
);


// --- Main History Component ---
const History = () => {
  const navigate = useNavigate();
  const [receipts, setReceipts] = useState([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedReceipt, setSelectedReceipt] = useState(null);

  useEffect(() => {
    const fetchReceipts = async () => {
  try {
    const auth = getAuth();
    const user = auth.currentUser;

    if (!user) {
      console.warn("User not authenticated.");
      setLoading(false);
      return;
    }

    const token = await user.getIdToken();
    const userId = user.uid;

    const response = await fetch(`${BACKEND_URL}/receipts/user/${userId}`, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    const transformed = data.receipts.map((r) => {
      const extracted = r.ocrData?.extractedData || {};

      const iconMap = {
        Shopping: FaShoppingCart,
        Dining: FaUtensils,
        Electronics: FaLaptop,
        Grocery: FaLeaf,
        Clothing: FaTshirt,
        Healthcare: FaHeartbeat,
        Travel: FaBus,
        Utilities: FaBolt,
      };

      const titlecase = (str) => {
        if (typeof str !== 'string') return 'Not Added';
        return str
          .toLowerCase()
          .replace(/_/g, ' ')
          .split(' ')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');
      };

      const category = titlecase(extracted.category);
      const payment = titlecase(extracted.paymentMethod);

      const formattedDate = new Date(r.processedAt).toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
      });

      return {
        id: r.receiptId,
        icon: iconMap[category] || FaListUl,
        storeName: extracted.merchantName || "Unknown Store",
        date: formattedDate,
        amount: `₹${(extracted.totalAmount || 0).toFixed(2)}`,
        subText: payment || 'Unknown',
        category,
        fullData: extracted,
      };
    });

    setReceipts(transformed);
  } catch (error) {
    console.error("Error fetching receipts:", error);
  } finally {
    setLoading(false);
  }
};


    fetchReceipts();
  }, []);
  
  const [loading, setLoading] = useState(true);


  return (
    <div className="bg-gray-50 h-full p-4 sm:p-8">
      <div className="max-w-7xl mx-auto bg-white p-8 rounded-2xl border border-gray-200">
        {/* Header */}
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

        {/* Receipt List */}
        <div className="bg-white rounded-lg border border-gray-200">
          {loading ? (
  <div className="p-6 text-center text-gray-500">Loading receipts...</div>
) : receipts.length === 0 ? (
  <div className="p-6 text-center text-gray-500">No receipts available.</div>
) : (
  receipts.map((receipt) => (
    <ReceiptRow
      key={receipt.id}
      receipt={receipt}
      onView={(r) => {
        setSelectedReceipt(r);
        setDialogOpen(true);
      }}
    />
  ))
)}
        </div>

        {/* Actions */}
        <footer className="flex justify-start items-center space-x-4 mt-8">
          {/* <button className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors">
            Edit Receipts
          </button> */}
          <button className="bg-white text-gray-800 font-bold py-3 px-6 rounded-lg border-2 border-gray-300 hover:bg-gray-100 transition-colors"
            onClick={() => navigate('/dashboard')}
          >
            View Dashboard
          </button>
          <button className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors">
            Export Data
          </button>
        </footer>
      </div>
      <ReceiptDetailsDialog
  isOpen={dialogOpen}
  onClose={() => setDialogOpen(false)}
  receipt={selectedReceipt}
/>
    </div>
  );
};

export default History;