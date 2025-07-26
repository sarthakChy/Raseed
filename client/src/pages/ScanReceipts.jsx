import React, { useRef, useState, useEffect } from 'react';
import { FiUpload, FiCamera } from 'react-icons/fi';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header';

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
  const videoRef = useRef(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [showCameraModal, setShowCameraModal] = useState(false);
  const [cameraStream, setCameraStream] = useState(null);
  const navigate = useNavigate();

  const handleFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      navigate('/receipt-result', { state: { file } });
    }
  };

  const handleCameraOpen = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      setCameraStream(stream);
      setShowCameraModal(true);
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error('Camera access denied:', err);
      alert('Unable to access camera. Please grant permission.');
    }
  };

  const handleCapture = () => {
    const video = videoRef.current;
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);
    const imageUrl = canvas.toDataURL('image/jpeg');

    setImagePreview(imageUrl);

    canvas.toBlob((blob) => {
      navigate('/receipt-result', { state: { file: blob } });
    }, 'image/jpeg');

    handleCloseCamera();
  };

  const handleCloseCamera = () => {
    if (cameraStream) {
      cameraStream.getTracks().forEach((track) => track.stop());
    }
    setShowCameraModal(false);
  };

  useEffect(() => {
    if (videoRef.current && cameraStream) {
      videoRef.current.srcObject = cameraStream;
    }
  }, [cameraStream]);

  const uploadOptions = [
    {
      icon: <FiUpload className="text-blue-800" style={{ fontSize: '44px', strokeWidth: '1.5' }} />,
      title: 'Upload from Device',
      description: '(PNG, JPG, PDF, etc.)',
      buttonText: 'Click to Upload',
      onClick: handleFileUpload,
    },
    {
      icon: <FiCamera className="text-blue-800" style={{ fontSize: '44px', strokeWidth: '1.5' }} />,
      title: 'Scan Using Camera',
      description: '(Take a Live Snapshot)',
      buttonText: 'Open Camera',
      onClick: handleCameraOpen,
    },
  ];

  return (
    <>
      <Header />
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
              <p className="text-blue-600 font-semibold mb-4">Redirecting to analyze...</p>
            </div>
          )}
        </div>
      </div>

      {showCameraModal && (
        <div className="fixed inset-0 bg-black bg-opacity-80 z-50 flex items-center justify-center">
          <div className="relative w-full max-w-lg p-4 bg-white rounded-xl shadow-xl">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="w-full rounded-xl"
              style={{ maxHeight: '400px', objectFit: 'cover' }}
            />
            <div className="flex justify-center mt-4 gap-4">
              <button
                className="bg-blue-600 text-white px-6 py-2 rounded-lg"
                onClick={handleCapture}
              >
                Capture
              </button>
              <button
                className="bg-red-500 text-white px-6 py-2 rounded-lg"
                onClick={handleCloseCamera}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ScanReceipts;


// {
//       icon: <SiGoogledrive className="text-4xl" />,
//       title: 'Import from Google Drive',
//       description: '(Connect and Select File)',
//       buttonText: 'Connect Drive',
//       onClick: handleGoogleDriveImport,
//     },