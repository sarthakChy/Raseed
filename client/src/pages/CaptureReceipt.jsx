import React, { useState } from 'react';
import { ArrowLeft, Upload } from 'lucide-react';

function CaptureReceiptPage({ onBack }) {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
        }
    };

    const handleAnalyze = () => {
        if (!selectedFile) return;
        setIsAnalyzing(true);
        setTimeout(() => {
            setIsAnalyzing(false);
            alert("Receipt analysis complete! (Simulation)");
        }, 2000);
    };

    return (
        <div className="max-w-md mx-auto">
            <header className="flex items-center py-4">
                <button onClick={onBack} className="p-2 rounded-full hover:bg-slate-200 mr-2">
                    <ArrowLeft className="h-6 w-6 text-slate-600" />
                </button>
                <h1 className="text-xl font-bold text-slate-800">Capture Receipt</h1>
            </header>
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                <input
                    type="file"
                    id="receiptUpload"
                    className="hidden"
                    accept="image/*"
                    onChange={handleFileChange}
                />
                {!previewUrl ? (
                    <label htmlFor="receiptUpload" className="flex flex-col items-center justify-center border-2 border-dashed border-slate-300 rounded-xl p-12 text-center cursor-pointer hover:bg-slate-100 transition">
                        <Upload className="h-12 w-12 text-slate-400 mb-2" />
                        <span className="font-semibold text-slate-600">Click to upload a receipt</span>
                        <span className="text-sm text-slate-500 mt-1">PNG, JPG, etc.</span>
                    </label>
                ) : (
                    <div className="mb-4">
                        <img src={previewUrl} alt="Receipt Preview" className="w-full h-auto rounded-lg shadow-inner" />
                    </div>
                )}

                {selectedFile && (
                    <div className="text-center mt-4">
                        <button
                            onClick={handleAnalyze}
                            disabled={isAnalyzing}
                            className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:bg-blue-300 flex items-center justify-center"
                        >
                            {isAnalyzing && (
                                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                            )}
                            {isAnalyzing ? "Analyzing..." : "Analyze Receipt"}
                        </button>
                    </div>
                )}
            </div>
        </div>
    );
}

export default CaptureReceiptPage;