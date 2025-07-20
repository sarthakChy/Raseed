import { useState } from 'react';
import { WandSparkles, Loader2, CheckCircle, AlertCircle } from 'lucide-react';

function AnalysisResultCard({ analysisResult,onScanAnother, onAddToWallet }) {
    const [isAddingToWallet, setIsAddingToWallet] = useState(false);
    const [walletResult, setWalletResult] = useState(null);
    const [walletError, setWalletError] = useState(null);

    const receiptData = analysisResult
    // A fallback in case the data is not available yet
    if (!receiptData) {
        return <p>Loading results...</p>;
    }

    const handleAddToWallet = async () => {
        setIsAddingToWallet(true);
        setWalletError(null);
        setWalletResult(null);

        try {
            const token = localStorage.getItem('Token');
            const uuid = localStorage.getItem('uuid');

            const response = await fetch('http://localhost:8000/api/receipts/create-wallet-pass', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${token}`
                },
                body: JSON.stringify({uuid:uuid})
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to create wallet pass');
            }

            const result = await response.json();
            setWalletResult(result);
            
            // If we have a wallet link, open it automatically
            if (result.wallet_link) {
                window.open(result.wallet_link, '_blank');
            }
            
        } catch (err) {
            console.error('Wallet pass creation error:', err);
            setWalletError(err.message);
        } finally {
            setIsAddingToWallet(false);
        }
    };

    return (
        <div className="w-full max-w-md mx-auto font-sans bg-white rounded-2xl shadow-lg p-6">
            <div className="flex flex-col gap-6">

                {/* --- Header --- */}
                <header className="flex items-center gap-3">
                    <WandSparkles className="w-8 h-8 text-purple-600" />
                    <div>
                        <h1 className="text-2xl font-bold text-slate-800">{receiptData.merchantName || "Merchant"}</h1>
                        <p className="text-sm text-slate-500">
                            {receiptData.transactionDate 
                                ? new Date(receiptData.transactionDate).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
                                : "Date not found"}
                        </p>
                    </div>
                </header>

                {/* --- Items List --- */}
                <div>
                    <h2 className="font-bold text-slate-700 mb-2">Items</h2>
                    <ul className="space-y-2 text-sm">
                        {receiptData.items?.map((item, i) => (
                            <ListItem key={i} name={item.name} value={`₹${item.price?.toFixed(2)}`} />
                        ))}
                    </ul>
                </div>

                {/* --- Total --- */}
                <div className="border-t border-slate-200 pt-4">
                     <div className="flex justify-between items-center font-bold text-xl">
                        <span className="text-slate-800">Total</span>
                        <span className="text-slate-900">₹{receiptData.total?.toFixed(2)}</span>
                    </div>
                </div>

                {/* --- Insights Section (Placeholder) --- */}
                <div>
                    <h2 className="font-bold text-slate-700 mb-2">Insights</h2>
                    <div className="bg-slate-50 p-3 rounded-lg text-sm text-slate-600">
                        <p>Category: <span className="font-semibold">{receiptData.category || "Groceries"}</span></p>
                        <p>Future AI-powered insights will appear here.</p>
                    </div>
                </div>

                {/* --- Wallet Status Messages --- */}
                {walletResult && (
                    <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                        <div className="flex items-center gap-2 text-green-800">
                            <CheckCircle className="w-5 h-5" />
                            <span className="font-semibold">Success!</span>
                        </div>
                        <p className="text-sm text-green-700 mt-1">
                            Your receipt has been added to Google Wallet. 
                            {walletResult.wallet_link && (
                                <a 
                                    href={walletResult.wallet_link} 
                                    target="_blank" 
                                    rel="noopener noreferrer"
                                    className="underline ml-1"
                                >
                                    Open wallet link
                                </a>
                            )}
                        </p>
                    </div>
                )}

                {walletError && (
                    <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                        <div className="flex items-center gap-2 text-red-800">
                            <AlertCircle className="w-5 h-5" />
                            <span className="font-semibold">Error</span>
                        </div>
                        <p className="text-sm text-red-700 mt-1">{walletError}</p>
                    </div>
                )}

                {/* --- Action Buttons --- */}
                <div className="grid grid-cols-2 gap-4">
                    <button 
                        onClick={onScanAnother}
                        className="w-full bg-slate-200 text-slate-800 font-bold py-3 px-4 rounded-lg hover:bg-slate-300 transition-colors"
                    >
                        Scan Another
                    </button>
                    
                    <button 
                        onClick={handleAddToWallet}
                        disabled={isAddingToWallet || walletResult}
                        className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                        {isAddingToWallet ? (
                            <>
                                <Loader2 className="w-4 h-4 animate-spin" />
                                Adding...
                            </>
                        ) : walletResult ? (
                            <>
                                <CheckCircle className="w-4 h-4" />
                                Added to Wallet
                            </>
                        ) : (
                            <>
                                <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M21 8V7a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v10a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3V9a1 1 0 0 0-1-1h-8.5L10 7V6a1 1 0 0 1 1-1h8a1 1 0 0 1 1 1v2z"/>
                                </svg>
                                Add to Wallet
                            </>
                        )}
                    </button>
                </div>
            </div>
        </div>
    );
}

// Helper sub-component for a cleaner structure
const ListItem = ({ name, value }) => (
    <li className="flex justify-between text-slate-600">
        <p className="truncate pr-4">{name}</p>
        <p className="font-mono">{value}</p>
    </li>
);

export default AnalysisResultCard;
