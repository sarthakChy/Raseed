import React, { useState } from 'react';

// Import pages and constants
import DashboardPage from './pages/Dashboard';
import CaptureReceiptPage from './pages/CaptureReceipt';
import AskRaseedPage from './pages/AskRaseed';
import { PAGES } from './constants/pages';

// Make sure you have removed all styles from src/index.css and src/App.css
// before adding the 3 tailwind lines to index.css

function App() {
    const [currentPage, setCurrentPage] = useState(PAGES.DASHBOARD);

    const navigateTo = (page) => setCurrentPage(page);

    const renderContent = () => {
        switch (currentPage) {
            case PAGES.CAPTURE_RECEIPT:
                // This wrapper will center the capture page correctly
                return (
                    <div className="max-w-md mx-auto">
                        <CaptureReceiptPage onBack={() => navigateTo(PAGES.DASHBOARD)} />
                    </div>
                );
            case PAGES.ASK_RASEED:
                 // This wrapper will center the ask page correctly
                return (
                    <div className="max-w-md mx-auto">
                         <AskRaseedPage onBack={() => navigateTo(PAGES.DASHBOARD)} />
                    </div>
                );
            case PAGES.DASHBOARD:
            default:
                // The Dashboard component itself will manage its own width
                return <DashboardPage onNavigate={navigateTo} />;
        }
    };

    return (
        <div className="bg-slate-50 min-h-screen font-sans text-gray-800">
            <div className="w-full p-4 lg:p-8">
                {renderContent()}
            </div>
        </div>
    );
}

export default App;