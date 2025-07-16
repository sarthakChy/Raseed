import React from 'react';

function StatsCard() {
    return (
        <div className="bg-white shadow-sm border border-slate-200 rounded-2xl p-6 mb-8">
            <div className="flex justify-around text-center">
                <div><h5 className="font-bold text-2xl text-slate-800">₹7.8K</h5><small className="text-slate-500">This month</small></div>
                <div><h5 className="font-bold text-2xl text-slate-800">15</h5><small className="text-slate-500">Receipts</small></div>
                <div><h5 className="font-bold text-2xl text-green-600">₹300</h5><small className="text-slate-500">Saved</small></div>
            </div>
        </div>
    );
}

export default StatsCard;