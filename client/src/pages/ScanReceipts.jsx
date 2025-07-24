import React, { useRef, useState, useEffect } from 'react';
import { FiUpload, FiCamera } from 'react-icons/fi';
import { SiGoogledrive } from 'react-icons/si';
import { useAuth } from '../context/AuthContext'; // adjust path if needed

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL;

const OptionCard = ({ icon, title, description, buttonText, onClick }) => (
  <div className="bg-white p-8 rounded-xl border border-gray-200 shadow-sm flex flex-col items-center justify-between text-center w-full sm:w-72 h-80">
    <div className="flex-shrink-0 mb-6 flex items-center justify-center h-12">{icon}</div>
    <div className="flex-grow flex flex-col justify-center">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">{title}</h2>
      <p className="text-sm text-gray-500">{description}</p>
    </div>
    <button
      onClick={onClick}
      className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors w-full mt-auto"
    >
      {buttonText}
    </button>
  </div>
);

const ScanReceipts = () => {
  const fileInputRef = useRef(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [analysisStatus, setAnalysisStatus] = useState(null); // "success" | "error" | null
  const [analysisResult, setAnalysisResult] = useState(null);
  const { user } = useAuth();
  const [dotCount, setDotCount] = useState(0);

  useEffect(() => {
    if (loading || analysisStatus === 'wallet add') {
      const interval = setInterval(() => {
        setDotCount(prev => (prev + 1) % 4);
      }, 500);
      return () => clearInterval(interval);
    }
  }, [loading, analysisStatus]);


  const sendToBackend = async (fileOrBlob) => {
    setLoading(true);
    setAnalysisStatus(null);
    setAnalysisResult(null);
    try {
      const token = await user.getIdToken();
      const formData = new FormData();
      formData.append('file', fileOrBlob);

      const response = await fetch(`${BACKEND_URL}/api/receipts/analyze`, {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
        },
        body: formData,
      });

      const result = await response.json();
      if (response.ok) {
        setAnalysisStatus('success');
        setAnalysisResult(result);
      } else {
        setAnalysisStatus('error');
      }
    } catch (err) {
      setAnalysisStatus('error');
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      sendToBackend(file);
    }
  };

  const handleCameraOpen = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const newWindow = window.open('', '_blank', 'width=640,height=480');

      const doc = newWindow.document;
      doc.body.style.margin = '0';
      doc.body.style.background = '#000';

      const video = doc.createElement('video');
      video.srcObject = stream;
      video.autoplay = true;
      video.playsInline = true;
      video.style.width = '100%';
      doc.body.appendChild(video);

      const captureButton = doc.createElement('button');
      captureButton.innerText = 'Capture';
      captureButton.style.cssText =
        'position:absolute;bottom:20px;left:50%;transform:translateX(-50%);padding:12px 24px;font-size:16px;background:#2196f3;color:white;border:none;border-radius:8px;cursor:pointer;';
      doc.body.appendChild(captureButton);

      captureButton.onclick = () => {
        const canvas = doc.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageUrl = canvas.toDataURL('image/jpeg');

        setImagePreview(imageUrl);
        canvas.toBlob((blob) => {
          sendToBackend(blob);
        }, 'image/jpeg');

        stream.getTracks().forEach((track) => track.stop());
        newWindow.close();
      };
    } catch (err) {
      console.error('Camera access denied:', err);
      alert('Unable to access camera. Please grant permission.');
    }
  };

  const handleGoogleDriveImport = () => {
    console.log('Google Drive integration coming soon');
  };

  const handleAddToWallet = async () => {
  try {
    const token = await user.getIdToken();
    setAnalysisStatus('wallet add');
    const response = await fetch(`${BACKEND_URL}/api/receipts/create-wallet-pass`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json', 
      },
      body: JSON.stringify({
        uuid: analysisResult['receiptId'],
      }),
    });

    const result = await response.json();

    if (response.ok) {
      window.open(result.wallet_link, '_blank');
      setAnalysisStatus('wallet success');
    } else {
      console.error("Error creating wallet pass:", result);
      setAnalysisStatus('wallet error');
    }
  } catch (err) {
    console.error("Fetch failed:", err);
    setAnalysisStatus('wallet error');
  }
};


  const handleReset = () => {
    setImagePreview(null);
    setAnalysisStatus(null);
    setLoading(false);
  };

  const uploadOptions = [
    {
      icon: <FiUpload className="text-blue-800" style={{ fontSize: '44px', strokeWidth: '1.5' }} />,
      title: 'Upload from Device',
      description: '(PNG, JPG, PDF, etc.)',
      buttonText: 'Click to Upload',
      onClick: handleFileUpload,
    },
    {
      icon: <SiGoogledrive className="text-4xl" />,
      title: 'Import from Google Drive',
      description: '(Connect and Select File)',
      buttonText: 'Connect Drive',
      onClick: handleGoogleDriveImport,
    },
    {
      icon: <FiCamera className="text-blue-800" style={{ fontSize: '44px', strokeWidth: '1.5' }} />,
      title: 'Scan Using Camera',
      description: '(Take a Live Snapshot)',
      buttonText: 'Open Camera',
      onClick: handleCameraOpen,
    },
  ];

  const toTitleCase = (str) => {
  return str ? str.charAt(0).toUpperCase() + str.slice(1).toLowerCase() : 'N/A';
};


  return (
    <div className="bg-gray-50 min-h-full flex items-center justify-center p-4">
      <div className="bg-white p-8 sm:p-12 rounded-2xl border border-gray-200 shadow-lg text-center max-w-4xl w-full">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Scan or Upload Your Receipt</h1>
        <p className="text-gray-600 mb-10">Choose one of the options below to start processing your receipt.</p>

        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept="image/*,.pdf"
          onChange={handleFileChange}
        />

        {!imagePreview ? (
          <div className="flex flex-col md:flex-row justify-center items-center gap-8">
            {uploadOptions.map((option) => (
              <OptionCard key={option.title} {...option} />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center">
            <img src={imagePreview} alt="Selected" className="max-w-md rounded-xl shadow-md mb-6" />
            {loading ? (
  <p className="text-blue-600 font-semibold mb-4">
    Analyzing receipt{'.'.repeat(dotCount)}
  </p>
) : analysisStatus === 'wallet add' ? (
  <p className="text-blue-600 font-semibold mb-4">
    Adding to wallet{'.'.repeat(dotCount)}
  </p>
) : analysisStatus === 'success' ? (
  <>
    <p className="text-green-600 font-semibold mb-4">Success! Receipt processed.</p>

    {analysisResult?.ocrData?.extractedData && (
  <div className="w-full bg-gray-100 rounded-xl p-6 text-left mb-4 shadow">
    <h3 className="text-xl font-bold mb-4 text-gray-800">Extracted Receipt Details</h3>
    <ul className="space-y-2 text-gray-700">
      <li><strong>Merchant:</strong> {analysisResult.ocrData.extractedData.merchantName || 'N/A'}</li>
      <li><strong>Date:</strong> {analysisResult.ocrData.extractedData.date || 'N/A'}</li>
      <li><strong>Total Amount:</strong> 
        {analysisResult.ocrData.extractedData.totalAmount ? `₹${analysisResult.ocrData.extractedData.totalAmount}` : 'N/A'}
      </li>
      <li><strong>Payment Method:</strong> {toTitleCase(analysisResult.ocrData.extractedData.paymentMethod) || 'N/A'}</li>
      <li><strong>Category:</strong> {toTitleCase(analysisResult.ocrData.extractedData.category) || 'N/A'}</li>
      <li><strong>Items:</strong>
        <ul className="ml-4 list-disc">
          {Array.isArray(analysisResult.ocrData.extractedData.items) && analysisResult.ocrData.extractedData.items.length > 0 ? (
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
)}

  </>
) : analysisStatus === 'error' ? (
  <p className="text-red-600 font-semibold mb-4">Error analyzing receipt.</p>
) : analysisStatus === 'wallet success' ? (
  <p className="text-green-600 font-semibold mb-4">Success! Receipt added to wallet.</p>
) : analysisStatus === 'wallet error' ? (
  <p className="text-red-600 font-semibold mb-4">Error adding to wallet.</p>
) : null}



            <div className="flex gap-4 mt-2">
              <button
                onClick={handleReset}
                className="bg-gray-300 text-gray-800 font-semibold py-2 px-4 rounded-lg hover:bg-gray-400"
              >
                Upload Another
              </button>
              <button
                onClick={handleAddToWallet}
                className="bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg hover:bg-blue-700"
              >
                Add to Wallet
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ScanReceipts;