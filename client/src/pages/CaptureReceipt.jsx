import { useState, useEffect } from 'react';
import { ArrowLeft, Upload, Loader2 } from 'lucide-react';
import AnalysisResultCard from '../components/AnalysisResultCard';

function CaptureReceiptPage({ user, onBack }) {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [analysisResult, setAnalysisResult] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        if (selectedFile && !analysisResult) {
            handleAnalyze();
        }
    }, [selectedFile]);

    const handleFileChange = (event) => {
        const file = event.target.files[0];
        if (file) {
            setAnalysisResult(null);
            setError(null);
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
        }
    };

    const handleAnalyze = async () => {
        if (!selectedFile) return;
        setIsAnalyzing(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', selectedFile);

        try {
            const token = localStorage.getItem('Token');
            const response = await fetch('http://localhost:8000/api/receipts/analyze', {
                method: 'POST',
                headers: { 'Authorization': `Bearer ${token}` },
                body: formData,
            });
            
            

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || `Request failed`);
            }
         
            const result = await response.json(); 
            localStorage.setItem('uuid', result.uuid); // Store the UUID in localStorage
            setAnalysisResult(result);
        } catch (err) {
            setError(err.message);
        } finally {
            setIsAnalyzing(false);
        }   
    };

    const handleScanAnother = () => {
        setAnalysisResult(null);
        setSelectedFile(null);
        setPreviewUrl(null);
        setError(null);
    };

    // If we have a result, show the new AnalysisResultCard.
    if (analysisResult) {
        return (
            <div className="max-w-md mx-auto">
                <header className="flex items-center py-4">
                    <button onClick={onBack} className="p-2 rounded-full hover:bg-slate-200 mr-2">
                        <ArrowLeft className="h-6 w-6 text-slate-600" />
                    </button>
                    <h1 className="text-xl font-bold text-slate-800">Receipt Analysis</h1>
                </header>
                <AnalysisResultCard 
                    analysisResult={analysisResult}
                    onScanAnother={handleScanAnother}
                />
            </div>
        );
    }

    // Otherwise, show the upload screen.
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
                    <div className="relative">
                        <div className="mb-4 max-h-96 overflow-y-auto rounded-lg border border-slate-200 shadow-inner">
                            <img src={previewUrl} alt="Receipt Preview" className={`w-full h-auto transition-opacity duration-300 ${isAnalyzing ? 'opacity-20' : 'opacity-100'}`} />
                        </div>
                        {isAnalyzing && (
                            <div className="absolute inset-0 flex flex-col items-center justify-center bg-white/50">
                                <Loader2 className="h-10 w-10 animate-spin text-blue-600" />
                                <p className="mt-2 font-semibold text-slate-700">Analyzing...</p>
                            </div>
                        )}
                    </div>
                )}
                {error && <p className="text-red-500 text-sm mt-4 text-center">{error}</p>}
            </div>
        </div>
    );
}

export default CaptureReceiptPage;