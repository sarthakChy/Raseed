import React from 'react';
import StatsCard from '../components/StatsCard';
import QuickActions from '../components/QuickActions';
import RecentPasses from '../components/RecentPasses';

function DashboardPage({ onNavigate }) {
    // This main div no longer has max-width or mx-auto, allowing it to be wide.
    return (
        <div className="max-w-7xl mx-auto">
            <header className="text-center pb-4">
                <h2 className="text-3xl font-bold text-slate-800">RASEED</h2>
                <p className="text-slate-500"><small>Smarter Receipts. Smarter Spending.</small></p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-12 lg:gap-8">
                {/* --- Left Column --- */}
                <div className="lg:col-span-5">
                    <StatsCard />
                    <QuickActions onNavigate={onNavigate} />
                </div>

                {/* --- Right Column --- */}
                <div className="lg:col-span-7 mt-8 lg:mt-0">
                    <RecentPasses />
                </div>
            </div>
        </div>
    );
}

export default DashboardPage;