import React from "react";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

const ReceiptDetailsDialog = ({ isOpen, onClose, receipt }) => {
  if (!isOpen || !receipt) return null;

  const data = receipt.fullData || {};

  const titlecase = (str) => {
    if (typeof str !== "string") return "N/A";
    return str
      .toLowerCase()
      .replace(/_/g, " ")
      .split(" ")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const handleViewImage = async () => {
    try {
      const res = await fetch(
        `${BACKEND_URL}/api/get-signed-url?gcs_path=${encodeURIComponent(receipt.gcsUri)}`
      );

      const data = await res.json();
      if (data.signed_url) {
        window.open(data.signed_url, "_blank");
      } else {
        alert("Unable to fetch image URL.");
      }
    } catch (err) {
      console.error("Error fetching signed URL:", err);
      alert("Something went wrong while fetching the image.");
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-40">
      <div className="bg-white rounded-lg p-6 max-w-2xl w-full shadow-lg overflow-y-auto max-h-[90vh] hide-scrollbar">
        <h2 className="text-xl font-bold mb-4">Receipt Details</h2>

        <div className="space-y-2 text-sm text-gray-700">
          <p>
            <strong>Merchant:</strong> {data.merchantName || "Unknown"}
          </p>
          <p>
            <strong>Total Amount:</strong> ₹{(data.totalAmount || 0).toFixed(2)}
          </p>
          <p>
            <strong>Date:</strong> {receipt.date}
          </p>
          <p>
            <strong>Category:</strong> {receipt.category}
          </p>
          <p>
            <strong>Payment Method:</strong> {titlecase(data.paymentMethod)}
          </p>
          <p>
            <strong>Tax:</strong> ₹{(data.tax || 0).toFixed(2)}
          </p>
          <p>
            <strong>Items:</strong>
          </p>
          <ul className="list-disc pl-6">
            {Array.isArray(data.items) && data.items.length > 0 ? (
              data.items.map((item, index) => (
                <li key={index}>
                  {item.name} — ₹{item.price?.toFixed(2)} × {item.quantity}
                </li>
              ))
            ) : (
              <li>No item data available</li>
            )}
          </ul>

        </div>

        <div className="mt-6 text-right space-x-1 md:space-x-3">
          {receipt.walletLink && (
            <button
              onClick={() => {
                window.open(receipt.walletLink, "_blank");
              }}
              className="px-4 py-2 bg-gray-700 text-white bg-green-500 rounded hover:bg-green-800"
            >
              Wallet Pass
            </button>
          )}
          <button
            onClick={handleViewImage}
            className="px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-800"
          >
            Receipt
          </button>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Okay
          </button>
        </div>
      </div>
    </div>
  );
};

export default ReceiptDetailsDialog;
