// import React from 'react';
// import { WandSparkles } from 'lucide-react';

// function AnalysisResultCard({ analysisResult, onScanAnother, onAddToWallet }) {
//     // This component now directly uses the structure from your backend API response.
//     const receiptData = analysisResult;

//     // A fallback in case the data is not available yet
//     if (!receiptData) {
//         return <p>Loading results...</p>;
//     }

//     return (
//         <div className="w-full max-w-md mx-auto font-sans bg-white rounded-2xl shadow-lg p-6">
//             <div className="flex flex-col gap-6">

//                 {/* --- Header --- */}
//                 <header className="flex items-center gap-3">
//                     <WandSparkles className="w-8 h-8 text-purple-600" />
//                     <div>
//                         <h1 className="text-2xl font-bold text-slate-800">{receiptData.merchantName || "Merchant"}</h1>
//                         <p className="text-sm text-slate-500">
//                             {receiptData.transactionDate 
//                                 ? new Date(receiptData.transactionDate).toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })
//                                 : "Date not found"}
//                         </p>
//                     </div>
//                 </header>

//                 {/* --- Items List --- */}
//                 <div>
//                     <h2 className="font-bold text-slate-700 mb-2">Items</h2>
//                     <ul className="space-y-2 text-sm">
//                         {receiptData.items?.map((item, i) => (
//                             <ListItem key={i} name={item.name} value={`₹${item.price?.toFixed(2)}`} />
//                         ))}
//                     </ul>
//                 </div>

//                 {/* --- Total --- */}
//                 <div className="border-t border-slate-200 pt-4">
//                      <div className="flex justify-between items-center font-bold text-xl">
//                         <span className="text-slate-800">Total</span>
//                         <span className="text-slate-900">₹{receiptData.total?.toFixed(2)}</span>
//                     </div>
//                 </div>

//                 {/* --- Insights Section (Placeholder) --- */}
//                 <div>
//                     <h2 className="font-bold text-slate-700 mb-2">Insights</h2>
//                     <div className="bg-slate-50 p-3 rounded-lg text-sm text-slate-600">
//                         <p>Category: <span className="font-semibold">{receiptData.category || "Groceries"}</span></p>
//                         <p>Future AI-powered insights will appear here.</p>
//                     </div>
//                 </div>

//                 {/* --- Action Buttons --- */}
//                 <div className="grid grid-cols-2 gap-4">
//                     <button 
//                         onClick={onScanAnother}
//                         className="w-full bg-slate-200 text-slate-800 font-bold py-3 px-4 rounded-lg hover:bg-slate-300 transition-colors"
//                     >
//                         Scan Another
//                     </button>
//                     <button 
//                         onClick={onAddToWallet}
//                         className="w-full bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 transition-colors"
//                     >
//                         Add to Wallet
//                     </button>
//                 </div>
//             </div>
//         </div>
//     );
// }

// // Helper sub-component for a cleaner structure
// const ListItem = ({ name, value }) => (
//     <li className="flex justify-between text-slate-600">
//         <p className="truncate pr-4">{name}</p>
//         <p className="font-mono">{value}</p>
//     </li>
// );

// export default AnalysisResultCard;
