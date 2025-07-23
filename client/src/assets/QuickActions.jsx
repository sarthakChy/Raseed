// import React from 'react';
// import { useNavigate } from 'react-router-dom';
// import { Camera, MessageSquare } from 'lucide-react';
// import ActionButton from './ActionButton';
// import { PAGES } from '../constants/pages';

// function QuickActions() {
//     const navigate = useNavigate();

//     return (
//         <section className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8">
//             <h4 className="font-bold text-xl text-slate-700 mb-4">Quick Actions</h4>
//             <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-1 gap-4">
//                 <ActionButton
//                     icon={<Camera className="h-7 w-7 text-blue-600" />}
//                     title="Capture Receipt"
//                     subtitle="Photo • Video • Stream"
//                     onClick={() => navigate('/capture')}
//                 />
//                 <ActionButton
//                     icon={<MessageSquare className="h-7 w-7 text-purple-600" />}
//                     title="Ask RASEED"
//                     subtitle="Any language supported"
//                     onClick={() => navigate('/ask')}
//                 />
//             </div>
//         </section>
//     );
// }

// export default QuickActions;
