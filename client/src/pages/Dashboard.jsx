import React from 'react';
import StatsCard from '../components/StatsCard';
import QuickActions from '../components/QuickActions';
import RecentPasses from '../components/RecentPasses';
import { useAuth } from '../context/AuthContext'; // <-- Your context hook
import { useNavigate } from 'react-router-dom';     // For redirect after logout

function DashboardPage({ onNavigate }) {
    const { logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = async () => {
        try {
            await logout();
            navigate('/signin'); // Redirect after logout
        } catch (error) {
            console.error("Logout failed", error);
        }
    };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* --- Navbar --- */}
            <nav className="bg-white border-b border-slate-200 px-4 sm:px-6 lg:px-8 py-4 flex items-center justify-between">
                <h2 className="text-2xl font-bold text-slate-800">RASEED</h2>
                <button
                    onClick={handleLogout}
                    className="bg-red-600 px-4 p-2 rounded-xl text-white font-medium hover:bg-red-500"
                >
                    Logout
                </button>
            </nav>

            {/* --- Main Content --- */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6 pb-12">
                <header className="text-center pb-6">
                    <h2 className="text-3xl font-bold text-slate-800">Dashboard</h2>
                    <p className="text-slate-500"><small>Smarter Receipts. Smarter Spending.</small></p>
                </header>

                <div className="grid grid-cols-1 lg:grid-cols-12 lg:gap-8">
                    <div className="lg:col-span-5 space-y-6">
                        <StatsCard />
                        <QuickActions onNavigate={onNavigate} />
                    </div>

                    <div className="lg:col-span-7 mt-10 lg:mt-0">
                        <RecentPasses />
                    </div>
                </div>
            </div>
        </div>
    );
}

export default DashboardPage;
