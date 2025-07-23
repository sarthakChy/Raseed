import React from 'react';
// 1. Import the icons you need
import { FiUpload, FiCamera } from 'react-icons/fi';
import { SiGoogledrive } from 'react-icons/si'; // <-- Correct import for Google Drive logo

// Reusable Option Card Component
type OptionCardProps = {
  icon: React.ReactNode;
  title: string;
  description: string;
  buttonText: string;
};

const OptionCard = ({ icon, title, description, buttonText }: OptionCardProps) => (
  <div className="bg-white p-8 rounded-xl border border-gray-200 shadow-sm flex flex-col items-center justify-between text-center w-full sm:w-72 h-80">
    <div className="flex-shrink-0 mb-6 flex items-center justify-center h-12">
      {icon}
    </div>
    <div className="flex-grow flex flex-col justify-center">
      <h2 className="text-xl font-semibold text-gray-800 mb-2">{title}</h2>
      <p className="text-sm text-gray-500">{description}</p>
    </div>
    <button className="bg-blue-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-blue-700 transition-colors w-full mt-auto">
      {buttonText}
    </button>
  </div>
);

// --- Main ScanRecipts Component ---
const ScanRecipts = () => {
  const uploadOptions = [
    {
      icon: <FiUpload className="text-blue-800" style={{ fontSize: '44px', strokeWidth: '1.5' }} />,
      title: 'Upload from Device',
      description: '(PNG, JPG, PDF, etc.)',
      buttonText: 'Click to Upload',
    },
    {
      // 2. Use the imported icon component here
      icon: <SiGoogledrive className="text-4xl" />,
      title: 'Import from Google Drive',
      description: '(Connect and Select File)',
      buttonText: 'Connect Drive',
    },
    {
      icon: <FiCamera className="text-blue-800" style={{ fontSize: '44px', strokeWidth: '1.5' }} />,
      title: 'Scan Using Camera',
      description: '(Take a Live Snapshot)',
      buttonText: 'Open Camera',
    },
  ];

  return (
    <div className="bg-gray-50 min-h-screen flex items-center justify-center p-4">
      <div className="bg-white p-8 sm:p-12 rounded-2xl border border-gray-200 shadow-lg text-center max-w-4xl w-full">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Scan or Upload Your Receipt
        </h1>
        <p className="text-gray-600 mb-10">
          Choose one of the options below to start processing your receipt.
        </p>
        
        <div className="flex flex-col md:flex-row justify-center items-center gap-8">
          {uploadOptions.map((option) => (
            <OptionCard
              key={option.title}
              icon={option.icon}
              title={option.title}
              description={option.description}
              buttonText={option.buttonText}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

export default ScanRecipts;