import React, { useState, useEffect } from 'react';
import {
  FaSearch,
  FaListUl,
  FaShoppingCart,
  FaUtensils,
  FaLaptop,
  FaLeaf,
  FaTshirt,
  FaHeartbeat,
  FaBus,
  FaBolt,
  FaQuestionCircle,
} from 'react-icons/fa';
import { getAuth } from 'firebase/auth';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';
import ReceiptDetailsDialog from '../components/ReceiptDetailsDialog';
import DeleteConfirmationDialog from "../components/DeleteConfirmationDialog";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

// --- Category Tag Component ---
const CategoryTag = ({ category }) => {
  const styles = {
    Shopping: { bg: 'bg-green-100', text: 'text-green-700', icon: <FaShoppingCart /> },
    Dining: { bg: 'bg-yellow-100', text: 'text-yellow-700', icon: <FaUtensils /> },
    Food: { bg: 'bg-yellow-100', text: 'text-yellow-700', icon: <FaUtensils /> },
    Electronics: { bg: 'bg-blue-100', text: 'text-blue-700', icon: <FaLaptop /> },
    Grocery: { bg: 'bg-lime-100', text: 'text-lime-700', icon: <FaLeaf /> },
    Groceries: { bg: 'bg-lime-100', text: 'text-lime-700', icon: <FaLeaf /> },
    Clothing: { bg: 'bg-pink-100', text: 'text-pink-700', icon: <FaTshirt /> },
    Healthcare: { bg: 'bg-red-100', text: 'text-red-700', icon: <FaHeartbeat /> },
    Travel: { bg: 'bg-cyan-100', text: 'text-cyan-700', icon: <FaBus /> },
    Utilities: { bg: 'bg-indigo-100', text: 'text-indigo-700', icon: <FaBolt /> },
    'Not Added': { bg: 'bg-gray-200', text: 'text-gray-600', icon: <FaQuestionCircle /> },
  };

  const { bg, text, icon } = styles[category] || styles['Not Added'];

  return (
    <span className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-medium ${bg} ${text}`}>
      {icon} {category}
    </span>
  );
};

// --- Receipt Row Component ---
const ReceiptRow = ({ receipt, onView, onDelete }) => (
  <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 p-4 border-b border-gray-200 last:border-b-0">
    <div className="flex-1">
      <div className="text-lg font-semibold text-gray-800">{receipt.storeName}</div>
      <div className="text-sm text-gray-500">{receipt.subText} • {receipt.date}</div>
    </div>
    <div className="flex flex-col sm:flex-row items-start sm:items-center sm:gap-3">
      <div className="text-md font-bold text-gray-800">{receipt.amount}</div>
      <CategoryTag category={receipt.category} />
    </div>
    <div className="flex items-center gap-4 text-gray-600 font-medium">
      <button className="hover:text-blue-600" onClick={() => onView(receipt)}>View</button>
      <button className="hover:text-red-600" onClick={onDelete}>Delete</button>
    </div>
  </div>
);

// --- Main History Component ---
const History = () => {
  const navigate = useNavigate();
  const [receipts, setReceipts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [selectedReceipt, setSelectedReceipt] = useState(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [receiptToDelete, setReceiptToDelete] = useState(null);

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
          headers: { Authorization: `Bearer ${token}` },
        });

        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();

        const transformed = data.receipts.map((r) => {
          const extracted = r.ocrData?.extractedData || {};

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

  const confirmDelete = (receiptId) => {
    setReceiptToDelete(receiptId);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirmed = async () => {
    try {
      const auth = getAuth();
      const user = auth.currentUser;
      if (!user) return;

      const token = await user.getIdToken();

      const res = await fetch(`${BACKEND_URL}/receipts/${receiptToDelete}`, {
        method: 'DELETE',
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!res.ok) {
        const error = await res.json();
        throw new Error(error.detail || 'Deletion failed');
      }

      setReceipts(prev => prev.filter(r => r.id !== receiptToDelete));
    } catch (err) {
      console.error("Error deleting receipt:", err);
      alert("Failed to delete the receipt. Please try again.");
    } finally {
      setDeleteDialogOpen(false);
      setReceiptToDelete(null);
    }
  };

  return (
    <>
      <Header />
      <div className="bg-gray-50 min-h-screen p-4 sm:p-8">
        <div className="max-w-7xl mx-auto bg-white p-6 sm:p-8 rounded-2xl border border-gray-200">
          <header className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
            <h1 className="text-3xl font-bold text-gray-800 mb-4 sm:mb-0">Receipts History</h1>
          </header>

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
                  onDelete={() => confirmDelete(receipt.id)}
                />
              ))
            )}
          </div>

          <footer className="flex flex-col sm:flex-row justify-start items-center gap-4 mt-8">
            <button
              className="bg-white text-gray-800 font-bold py-3 px-6 rounded-lg border-2 border-gray-300 hover:bg-gray-100 transition-colors"
              onClick={() => navigate('/dashboard')}
            >
              View Dashboard
            </button>
            <button className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors">
              Export Receipts
            </button>
          </footer>
        </div>

        <ReceiptDetailsDialog
          isOpen={dialogOpen}
          onClose={() => setDialogOpen(false)}
          receipt={selectedReceipt}
        />

        <DeleteConfirmationDialog
          isOpen={deleteDialogOpen}
          onCancel={() => {
            setDeleteDialogOpen(false);
            setReceiptToDelete(null);
          }}
          onConfirm={handleDeleteConfirmed}
        />
      </div>
    </>
  );
};

export default History;
