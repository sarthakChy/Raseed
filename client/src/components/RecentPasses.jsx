import React from 'react';
import PassCard from './PassCard';

function RecentPasses() {
    const mockRecentPasses = [
        { id: 1, title: 'Grocery Insights', description: 'Spending ↑12% this month • Top category: Snacks', time: '3 hours ago', status: 'Updated' },
        { id: 2, title: 'Smart Alert', description: 'Detergent runs out in 2 days • Add to shopping list?', time: '1 day ago', status: 'Action' },
        { id: 3, title: 'Dining Out', description: 'You spent ₹2,500 on dining last week.', time: '2 days ago', status: 'Info' }
    ];

    return (
        <section className="bg-white border border-slate-200 rounded-2xl p-6 sm:p-8">
            <h4 className="font-bold text-xl text-slate-700 mb-4">Recent Wallet Passes</h4>
            <div className="space-y-3">
                {mockRecentPasses.map(pass => <PassCard key={pass.id} pass={pass} />)}
            </div>
        </section>
    );
}

export default RecentPasses;
