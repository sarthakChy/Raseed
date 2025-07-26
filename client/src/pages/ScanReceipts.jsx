import React, { useRef, useState } from 'react';
import { FiUpload, FiCamera } from 'react-icons/fi';
import { SiGoogledrive } from 'react-icons/si';
import { useNavigate } from 'react-router-dom';
import Header from '../components/Header'; // ✅ Add this line

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
          navigate('/receipt-result', { state: { file: blob } });
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

  const uploadOptions = [
    {
      icon: <FiUpload className="text-blue-800" style={{ fontSize: '44px', strokeWidth: '1.5' }} />,
      title: 'Upload from Device',
      description: '(PNG, JPG, PDF, etc.)',
      buttonText: 'Click to Upload',
      onClick: handleFileUpload,
    },
    // {
    //   icon: <SiGoogledrive className="text-4xl" />,
    //   title: 'Import from Google Drive',
    //   description: '(Connect and Select File)',
    //   buttonText: 'Connect Drive',
    //   onClick: handleGoogleDriveImport,
    // },
    // {
    //   icon: <FiCamera className="text-blue-800" style={{ fontSize: '44px', strokeWidth: '1.5' }} />,
    //   title: 'Scan Using Camera',
    //   description: '(Take a Live Snapshot)',
    //   buttonText: 'Open Camera',
    //   onClick: handleCameraOpen,
    // },
  ];

  return (
    <>
      <Header /> {/* ✅ Add this inside the return */}
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
    </>
  );
}
export default ScanReceipts;