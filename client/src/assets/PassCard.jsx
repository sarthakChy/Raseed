// import React from 'react';
// import { BarChart2, AlertCircle } from 'lucide-react';

// function PassCard({ pass }) {
//     const getStatusColor = (status) => {
//         switch (status) {
//             case 'Updated': return 'bg-blue-100 text-blue-800';
//             case 'Action': return 'bg-yellow-100 text-yellow-800';
//             case 'Info': return 'bg-green-100 text-green-800';
//             default: return 'bg-gray-100 text-gray-800';
//         }
//     };

//     return (
//         <div className="bg-white p-4 rounded-2xl shadow-sm border border-slate-200 flex items-start gap-4">
//             <div className="flex-shrink-0 h-12 w-12 bg-slate-100 rounded-full flex items-center justify-center">
//                 {pass.status === 'Action' ? <AlertCircle className="h-6 w-6 text-yellow-500" /> : <BarChart2 className="h-6 w-6 text-slate-500" />}
//             </div>
//             <div className="flex-grow">
//                 <div className="flex justify-between items-center">
//                     <h6 className="font-bold text-slate-900">{pass.title}</h6>
//                     <span className={`text-xs font-semibold px-2.5 py-1 rounded-full ${getStatusColor(pass.status)}`}>
//                         {pass.status}
//                     </span>
//                 </div>
//                 <p className="text-sm text-slate-600 mt-1">{pass.description}</p>
//                 <div className="flex justify-between items-center mt-2">
//                     <p className="text-xs text-slate-400">{pass.time}</p>
//                     <a href="#" className="text-sm font-semibold text-blue-600 hover:underline">View details ▸</a>
//                 </div>
//             </div>
//         </div>
//     );
// }

// export default PassCard;