import React, { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { FaArrowLeft } from "react-icons/fa";
import { useAuth } from "../context/AuthContext";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

const ReceiptResult = () => {
  const navigate = useNavigate();
  const { imagePreview, analysisStatus, analysisResult } = useLocation().state || {};
  const { user } = useAuth();

  const [walletStatus, setWalletStatus] = useState("idle"); // idle | adding | success | error
  const [dotCount, setDotCount] = useState(1);

  useEffect(() => {
    if (walletStatus === "adding") {
      const interval = setInterval(() => {
        setDotCount((prev) => (prev % 3) + 1);
      }, 500);
      return () => clearInterval(interval);
    }
  }, [walletStatus]);

  const toTitleCase = (str) =>
    str ? str.charAt(0).toUpperCase() + str.slice(1).toLowerCase() : "N/A";

  const handleAddToWallet = async () => {
    setWalletStatus("adding");
    try {
      const token = await user.getIdToken();

      const response = await fetch(`${BACKEND_URL}/shopping-list/create-shopping-pass`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          items: analysisResult?.ocrData?.extractedData?.items || [],
          merchant: analysisResult?.ocrData?.extractedData?.merchantName || "",
          totalAmount: analysisResult?.ocrData?.extractedData?.totalAmount || 0,
        }),
      });

      if (response.ok) {
        setWalletStatus("success");
      } else {
        setWalletStatus("error");
      }
    } catch (err) {
      console.error("Add to wallet failed:", err);
      setWalletStatus("error");
    }
  };

  const handleReset = () => {
    navigate("/scanreceipt");
  };

  return (
    <div className="bg-gray-50 min-h-screen p-4">
      <button
        onClick={() => navigate("/scanreceipts")}
        className="flex items-center text-gray-600 hover:text-blue-600 mb-4"
      >
        <FaArrowLeft className="mr-2" /> Back
      </button>

      <div className="bg-white p-8 sm:p-12 rounded-2xl border border-gray-200 shadow-lg text-center max-w-4xl w-full mx-auto">
        <img
          src={imagePreview}
          alt="Selected"
          className="max-w-md rounded-xl shadow-md mb-6"
        />

        {walletStatus === "adding" && (
          <p className="text-blue-600 font-semibold mb-4">
            Adding to wallet{".".repeat(dotCount)}
          </p>
        )}
        {walletStatus === "success" && (
          <p className="text-green-600 font-semibold mb-4">
            Successfully added to your Google Wallet!
          </p>
        )}
        {walletStatus === "error" && (
          <p className="text-red-600 font-semibold mb-4">
            Failed to add to wallet. Please try again.
          </p>
        )}

        {analysisStatus === "success" && analysisResult?.ocrData?.extractedData ? (
          <div className="w-full bg-gray-100 rounded-xl p-6 text-left mb-4 shadow">
            <h3 className="text-xl font-bold mb-4 text-gray-800">
              Extracted Receipt Details
            </h3>
            <ul className="space-y-2 text-gray-700">
              <li><strong>Merchant:</strong> {analysisResult.ocrData.extractedData.merchantName || "N/A"}</li>
              <li><strong>Date:</strong> {analysisResult.ocrData.extractedData.date || "N/A"}</li>
              <li><strong>Total Amount:</strong> ₹{analysisResult.ocrData.extractedData.totalAmount || "N/A"}</li>
              <li><strong>Payment Method:</strong> {toTitleCase(analysisResult.ocrData.extractedData.paymentMethod)}</li>
              <li><strong>Category:</strong> {toTitleCase(analysisResult.ocrData.extractedData.category)}</li>
              <li>
                <strong>Items:</strong>
                <ul className="ml-4 list-disc">
                  {Array.isArray(analysisResult.ocrData.extractedData.items) &&
                  analysisResult.ocrData.extractedData.items.length > 0 ? (
                    analysisResult.ocrData.extractedData.items.map((item, index) => (
                      <li key={index}>
                        {item.name} — ₹{item.price} × {item.quantity || 1}
                      </li>
                    ))
                  ) : (
                    <li>No item data found</li>
                  )}
                </ul>
              </li>
            </ul>
          </div>
        ) : (
          <p className="text-red-600 font-semibold mb-4">Error analyzing receipt.</p>
        )}

        <div className="flex gap-4 mt-4 justify-center">
          <button
            onClick={handleReset}
            className="bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded-lg hover:bg-gray-400"
          >
            Upload Another
          </button>
          <button
            onClick={handleAddToWallet}
            className="bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700"
            disabled={walletStatus === "adding"}
          >
            Add to Wallet
          </button>
        </div>
      </div>
    </div>
  );
};

export default ReceiptResult;
