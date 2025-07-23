// import React from 'react';
// import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
// import PrivateRoute from './components/PrivateRoute';
// import DashboardPage from './pages/Dashboard';
// import CaptureReceiptPage from './pages/CaptureReceipt';
// import AskRaseedPage from './pages/AskRaseed';
// import SignInPage from './pages/SignIn';
// import SignUpPage from './pages/SignUp';
// import Home from './pages/Home';
// import ReceiptPassPreview from './components/AnalysisResultCard';

// export default function App() {
//   return (
//     <div className="bg-slate-50 min-h-screen font-sans text-gray-800">
//       <Router>
//         <Routes>
//           {/* Public Routes */}
//           <Route path="/" element={<Home />} />
//           <Route path="/signin" element={<SignInPage />} />
//           <Route path="/signup" element={<SignUpPage />} />

//           {/* Private Routes */}
//           <Route
//             path="/dashboard"
//             element={
//               <PrivateRoute>
//                 <DashboardPage />
//               </PrivateRoute>
//             }
//           />
//           <Route
//             path="/capture"
//             element={
//               <PrivateRoute>
//                 <div className="max-w-md mx-auto">
//                   <CaptureReceiptPage />
//                 </div>
//               </PrivateRoute>
//             }
//           />
//           <Route
//             path="/ask"
//             element={
//               <PrivateRoute>
//                 <div className="max-w-md mx-auto">
//                   <AskRaseedPage />
//                 </div>
//               </PrivateRoute>
//             }
//           />

//           <Route path="*" element={<Navigate to="/" />} />
//         </Routes>
//       </Router>
//     </div>
//   );
// }

// src/App.tsx - CORRECTED

import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Header from './components/Header';
import Hero from './components/Hero';
import About from './pages/About';
import FAQs from './pages/FAQs';
import Contact from './pages/Contact';
import UpgradeComponent from './pages/Upgrade'; 
import GetStarted from './pages/GetStarted';
import Chatbot from "./pages/Chatbot";
import DashBoard from "./pages/Dash.tsx";


const Upgrade = () => <UpgradeComponent />; // Using the corrected UpgradePlan component
const App: React.FC = () => {
  return (
    <Router>
      <div className="bg-[#FEFBF6] min-h-screen text-gray-900 overflow-x-hidden">
        <div className="container mx-auto px-6 lg:px-8">
          <Header />
          <main>
            <Routes>
              <Route path="/" element={<Hero />} />
              <Route path="/about" element={<About />} />
              <Route path="/getstarted" element={<GetStarted />} />
              <Route path="/upgrade" element={<Upgrade />} />
              <Route path="/faq" element={<FAQs />} />
              <Route path="/contact" element={<Contact />} />
              <Route path="/chatbot" element={<Chatbot />} />
              <Route path="/dashboard" element={<DashBoard />} />
            </Routes>
          </main>
        </div>
      </div>
    </Router>
  );
};

export default App;
